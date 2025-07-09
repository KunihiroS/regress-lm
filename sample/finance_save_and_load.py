import logging
import os
import time
import argparse
from datetime import datetime
from typing import Dict, List
import glob
import yaml

import matplotlib.pyplot as plt
import numpy as np
from regress_lm import core
from regress_lm import rlm

# --- パス設定 ---
# このスクリプトの場所を基準にプロジェクトルートを決定
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- ロギング設定 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

# --- 設定値・定数 ---
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "saved_models")
NUM_SAMPLES = 128
ABNORMAL_THRESHOLD = 10
DEFAULT_MODEL_NAME = "finance_model.pth"

"""
異常値の定義:
・ここでは「±10」を超える値を異常値と定義します。
・理由: このタスクのyのスケール（例: 0.3, -0.3）から大きく外れており、
  物理的にも不自然な値であるためです。
・運用: 必要に応じて閾値を変更してください。
"""


def sanitize_label_for_filename(label: str) -> str:
    """ファイル名用にラベルをサニタイズし、長さを制限する。"""
    # ファイル名に使えない可能性のある文字をアンダースコアに置換
    sanitized = "".join(c if c.isalnum() else "_" for c in label)
    # 連続するアンダースコアを1つにまとめる
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    # 先頭と末尾のアンダースコアを削除
    sanitized = sanitized.strip('_')
    # 50文字に制限
    return sanitized[:50]


def calculate_statistics(samples: List[float], threshold: float) -> Dict[str, float]:
    """サンプリング結果から各種統計量を計算する。"""
    samples = np.array(samples)
    total_count = len(samples)

    # 異常値の計算
    abnormal_mask = np.abs(samples) > threshold
    abnormal_count = np.sum(abnormal_mask)

    # ヒストグラムベースのエントロピー計算
    hist, bin_edges = np.histogram(samples, bins=20, density=True)
    hist = hist + 1e-12  # log(0)回避
    entropy = -np.sum(hist * np.log(hist)) * (bin_edges[1] - bin_edges[0])

    return {
        "mean": np.mean(samples),
        "std": np.std(samples),
        "entropy": entropy,
        "abnormal_count": abnormal_count,
        "total_count": total_count,
        "abnormal_ratio": abnormal_count / total_count if total_count > 0 else 0,
    }


def plot_distribution(samples: List[float], label: str, stats: Dict[str, float], output_dir: str, timestamp: str):
    """計算された統計量を用いて分布をプロットし、画像を保存する。"""
    samples = np.array(samples)
    plt.figure()
    plt.hist(samples, bins=20, density=True, alpha=0.7)
    plt.title(f"{label} samples (mean={stats['mean']:.3f}, std={stats['std']:.3f})")
    plt.xlabel("Predicted y")
    plt.ylabel("Density")
    plt.grid(True)

    safe_label = sanitize_label_for_filename(label)
    fname = f"{output_dir}/{timestamp}_{safe_label}_distribution.png"

    plt.savefig(fname)
    plt.close()
    logging.info(f"分布ヒストグラム画像を {fname} に保存しました。")


def save_statistics_to_file(label: str, stats: Dict[str, float], timing_data: Dict[str, float], output_dir: str, timestamp: str):
    """計算された統計量と処理時間をテキストファイルに保存する。"""
    safe_label = sanitize_label_for_filename(label)
    fname = f"{output_dir}/{timestamp}_{safe_label}_result.txt"
    with open(fname, 'w', encoding='utf-8') as f:
        f.write(f"Query: {label}\n\n")
        f.write("--- Statistics ---\n")
        for key, value in stats.items():
            if isinstance(value, float):
                f.write(f"- {key}: {value:.4f}\n")
            else:
                f.write(f"- {key}: {value}\n")
        
        f.write("\n--- Performance ---\n")
        for key, value in timing_data.items():
            f.write(f"- {key}: {value:.2f}\n")

    logging.info(f"統計情報と処理時間を {fname} に保存しました。")


def print_samples(label: str, samples: List[float], max_print: int = 32):
    """推測値をprintで出力する（長すぎる場合は先頭・末尾のみ）。"""
    samples = np.array(samples)
    total = len(samples)
    if total <= max_print:
        print(f"{label} samples: {samples}")
    else:
        head = samples[:max_print // 2]
        tail = samples[-max_print // 2:]
        print(f"{label} samples: {head.tolist()} ... {tail.tolist()} (total {total})")


def load_finetuning_data(data_dir: str = None) -> List[core.Example]:
    """ファインチューニングデータを読み込む"""
    if data_dir is None:
        data_dir = os.path.join(PROJECT_ROOT, "test/data/finance_01/finetuning/")
    else:
        data_dir = os.path.join(PROJECT_ROOT, data_dir)
    
    logging.info(f"ディレクトリからファインチューニングデータを検索中: {data_dir}")
    finetune_files = glob.glob(os.path.join(data_dir, "*.yml"))
    
    examples = []
    for f_path in finetune_files:
        logging.info(f"  - 読み込み中: {f_path}")
        with open(f_path, 'r') as f:
            data = yaml.safe_load(f)
            examples.append(core.Example(x=data['news'], y=data['change_rate']))
    
    logging.info(f"{len(examples)}件のファインチューニングデータを読み込みました。")
    return examples


def load_test_data() -> List[core.ExampleInput]:
    """テストデータを読み込む"""
    test_file_path = os.path.join(PROJECT_ROOT, "test/data/finance_01/test_target.yml")
    logging.info(f"テスト用クエリを読み込み中: {test_file_path}")
    with open(test_file_path, 'r') as f:
        test_data = yaml.safe_load(f)
    
    queries = [core.ExampleInput(x=test_data['news'])]
    logging.info("クエリの準備が完了しました。")
    return queries


def save_model(reg_lm: rlm.RegressLM, model_name: str = DEFAULT_MODEL_NAME):
    """モデルを保存する"""
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_SAVE_DIR, model_name)
    
    logging.info(f"モデルを保存中: {model_path}")
    reg_lm.save_model(model_path)
    
    # ファイルサイズを確認
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        logging.info(f"モデル保存完了: {model_path} ({file_size:.2f} MB)")
    
    return model_path


def load_model(reg_lm: rlm.RegressLM, model_name: str = DEFAULT_MODEL_NAME):
    """モデルを読み込む"""
    model_path = os.path.join(MODEL_SAVE_DIR, model_name)
    
    if not os.path.exists(model_path):
        logging.error(f"モデルファイルが見つかりません: {model_path}")
        return False
    
    logging.info(f"モデルを読み込み中: {model_path}")
    reg_lm.load_model(model_path)
    logging.info("モデル読み込み完了")
    return True


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(
        description='金融ニュース分析 - セーブ・ロード機能付き',
        epilog="""
使用例:
  # 新規学習してモデルを保存
  python test/finance_save_and_load.py

  # 保存されたモデルを読み込んで推論のみ
  python test/finance_save_and_load.py --load-model finance_model.pth --no-finetune

  # モデルを保存しない（テスト用）
  python test/finance_save_and_load.py --no-save

  # 別の名前でモデルを保存
  python test/finance_save_and_load.py --save-model my_finance_model.pth

  # 既存モデルに新しいデータを追加学習
  python test/finance_save_and_load.py --load-model old_model.pth --additional-data test/data/finance_02/ --save-model updated_model.pth


        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--load-model', type=str, default=None,
                       help='読み込むモデルファイル名（例: finance_model.pth）\n'
                            '指定しない場合は新規学習を実行')
    parser.add_argument('--save-model', type=str, default=DEFAULT_MODEL_NAME,
                       help='保存するモデルファイル名（デフォルト: %s）' % DEFAULT_MODEL_NAME)
    parser.add_argument('--no-save', action='store_true',
                       help='モデルを保存しない（テスト用）')
    parser.add_argument('--no-finetune', action='store_true',
                       help='ファインチューニングをスキップ（--load-modelと併用時のみ有効）\n'
                            '読み込んだモデルで推論のみを実行')
    parser.add_argument('--additional-data', type=str, default=None,
                       help='追加学習用のデータディレクトリ（例: test/data/finance_02/）\n'
                            '指定すると既存モデルに新しいデータを追加学習')

    
    args = parser.parse_args()
    
    # 画像保存ディレクトリとタイムスタンプ
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timing_data = {}

    # 1. モデルのインスタンスを作成
    logging.info("ステップ1: RegressLMモデルのインスタンスを作成します...")
    start_time = time.time()
    reg_lm = rlm.RegressLM.from_default(max_input_len=2048)
    timing_data['model_instantiation_sec'] = time.time() - start_time
    logging.info(f"モデルのインスタンス作成が完了しました。({timing_data['model_instantiation_sec']:.2f}秒)")

    # 2. モデル読み込み（指定された場合）
    if args.load_model:
        logging.info("ステップ2: 保存されたモデルを読み込みます...")
        if load_model(reg_lm, args.load_model):
            logging.info("モデル読み込み完了")
        else:
            logging.error("モデル読み込みに失敗しました。新規学習を開始します。")
            args.load_model = None  # 読み込み失敗時は新規学習
    
    # 3. ファインチューニング（読み込みが失敗した場合、または読み込みなしの場合）
    if not args.load_model or not args.no_finetune:
        logging.info("ステップ3: ファインチューニング用のデータを準備し、モデルを微調整します...")
        start_time = time.time()
        
        # 学習モードの判定
        if args.additional_data and args.load_model:
            # 追加学習モード
            logging.info(f"追加学習モード: {args.additional_data}のデータを使用")
            examples = load_finetuning_data(args.additional_data)
        else:
            # 新規学習モード
            logging.info("新規学習モード: 初期モデルから学習開始")
            examples = load_finetuning_data()
        
        logging.info("ファインチューニングを開始します...")
        reg_lm.fine_tune(examples, batch_size=1)
        timing_data['fine_tuning_sec'] = time.time() - start_time
        logging.info(f"モデルのファインチューニングが完了しました。({timing_data['fine_tuning_sec']:.2f}秒)")
        
        # モデルを保存（--no-saveが指定されていない場合）
        if not args.no_save:
            model_path = save_model(reg_lm, args.save_model)
            timing_data['model_save_sec'] = time.time() - start_time - timing_data.get('fine_tuning_sec', 0)
    else:
        logging.info("ステップ3: ファインチューニングをスキップします（読み込んだモデルを使用）")

    # 4. 推論
    logging.info("ステップ4: 推論用のクエリを準備します...")
    queries = load_test_data()

    logging.info(f"ステップ5: モデルを使って予測（サンプリング数: {NUM_SAMPLES}）を実行します...")
    start_time = time.time()
    results = reg_lm.sample(queries, num_samples=NUM_SAMPLES)
    timing_data['inference_sec'] = time.time() - start_time
    logging.info(f"サンプリングが完了しました。({timing_data['inference_sec']:.2f}秒)")

    # 5. 結果の分析と可視化
    logging.info("--- 予測結果の分析 ---")
    for query, samples in zip(queries, results):
        label = query.x
        logging.info(f"クエリ「{label[:30]}...」の結果を分析中...")

        # 推測値をprintで出力
        print_samples(label, samples, max_print=32)

        stats = calculate_statistics(samples, ABNORMAL_THRESHOLD)

        logging.info(
            f"[{label}]: 平均={stats['mean']:.3f}, 標準偏差={stats['std']:.3f}, "
            f"エントロピー={stats['entropy']:.3f}, "
            f"異常値={stats['abnormal_count']}/{stats['total_count']} ({stats['abnormal_ratio']:.1%}) "
            f"[閾値: |y| > {ABNORMAL_THRESHOLD}]"
        )

        plot_distribution(samples, label, stats, OUTPUT_DIR, timestamp)
        save_statistics_to_file(label, stats, timing_data, OUTPUT_DIR, timestamp)

    # 6. 実行サマリー
    logging.info("=== 実行サマリー ===")
    if args.load_model:
        logging.info(f"読み込みモデル: {args.load_model}")
    if not args.no_save:
        logging.info(f"保存モデル: {args.save_model}")
    logging.info(f"処理時間: モデル初期化={timing_data.get('model_instantiation_sec', 0):.2f}s, "
                f"ファインチューニング={timing_data.get('fine_tuning_sec', 0):.2f}s, "
                f"推論={timing_data.get('inference_sec', 0):.2f}s")


if __name__ == "__main__":
    main() 