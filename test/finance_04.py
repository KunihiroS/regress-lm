import logging
import os
import time
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
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

# --- 設定値・定数 ---
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
NUM_SAMPLES = 128
ABNORMAL_THRESHOLD = 10
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


def main():
    """メイン実行関数"""
    # --- ディレクトリとパスの設定 ---
    # このテストの名前（チェックポイントや出力のサブディレクトリ名になる）
    TEST_NAME = "finance_test_01"
    
    # チェックポイントと出力のルートディレクトリ
    CHECKPOINT_ROOT = os.path.join(PROJECT_ROOT, "work/model_checkpoints")
    OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "work/output")

    # このテスト専用のディレクトリを作成
    checkpoint_dir = os.path.join(CHECKPOINT_ROOT, TEST_NAME)
    output_dir = os.path.join(OUTPUT_ROOT, TEST_NAME)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # タイムスタンプと処理時間記録用の辞書
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timing_data = {}

    # --- モデルの準備 ---
    logging.info("--- ステップ1: モデルの準備 ---")
    start_time = time.time()
    
    # 1. モデルのインスタンスを作成
    reg_lm = rlm.RegressLM.from_default(max_input_len=2048)

    # 2. 最新のチェックポイントを探して読み込む
    checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
    if os.path.exists(checkpoint_path):
        logging.info(f"既存のチェックポイントを読み込みます: {checkpoint_path}")
        reg_lm.load_checkpoint(checkpoint_path)
    else:
        logging.info("既存のチェックポイントが見つかりません。新しいモデルから開始します。")

    timing_data['model_preparation_sec'] = time.time() - start_time
    logging.info(f"モデルの準備が完了しました。 ({timing_data['model_preparation_sec']:.2f}秒)")

    # 2. ファインチューニング
    logging.info("ステップ2: ファインチューニング用のデータを準備し、モデルを微調整します...")
    start_time = time.time()
    
    # YAMLファイルからファインチューニングデータを読み込む
    finetune_data_dir = os.path.join(PROJECT_ROOT, "test/data/finance_04/finetuning/")
    logging.info(f"ディレクトリからファインチューニングデータを検索中: {finetune_data_dir}")
    finetune_files = glob.glob(os.path.join(finetune_data_dir, "*.yml"))
    
    examples = []
    for f_path in finetune_files:
        logging.info(f"  - 読み込み中: {f_path}")
        with open(f_path, 'r') as f:
            data = yaml.safe_load(f)
            examples.append(core.Example(x=data['news'], y=data['change_rate']))
    
    logging.info(f"{len(examples)}件のファインチューニングデータを読み込みました。")

    logging.info("ファインチューニングを開始します...")
    reg_lm.fine_tune(examples, batch_size=1)
    timing_data['fine_tuning_sec'] = time.time() - start_time
    logging.info(f"モデルのファインチューニングが完了しました。({timing_data['fine_tuning_sec']:.2f}秒)")

    # 2b. ファインチューニング後のモデルを保存
    logging.info("ステップ2b: ファインチューニング後のモデルをチェックポイントとして保存します...")
    start_time = time.time()
    reg_lm.save_checkpoint(checkpoint_path)
    timing_data['save_checkpoint_sec'] = time.time() - start_time
    logging.info(f"チェックポイントの保存が完了しました。 ({timing_data['save_checkpoint_sec']:.2f}秒)")

    # 3. 推論
    logging.info("ステップ3: 推論用のクエリを準備します...")
    
    # テスト用YAMLファイルからクエリを読み込む
    test_file_path = os.path.join(PROJECT_ROOT, "test/data/finance_04/test_target.yml")
    logging.info(f"テスト用クエリを読み込み中: {test_file_path}")
    with open(test_file_path, 'r') as f:
        test_data = yaml.safe_load(f)
    
    queries = [
        core.ExampleInput(x=test_data['news'])
    ]
    logging.info("クエリの準備が完了しました。")

    logging.info(f"ステップ4: モデルを使って予測（サンプリング数: {NUM_SAMPLES}）を実行します...")
    start_time = time.time()
    results = reg_lm.sample(queries, num_samples=NUM_SAMPLES)
    timing_data['inference_sec'] = time.time() - start_time
    logging.info(f"サンプリングが完了しました。({timing_data['inference_sec']:.2f}秒)")

    # 4. 結果の分析と可視化
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

        # このテスト実行に紐づく出力サブディレクトリを作成
        current_output_dir = os.path.join(output_dir, timestamp)
        os.makedirs(current_output_dir, exist_ok=True)

        plot_distribution(samples, label, stats, current_output_dir, timestamp)
        save_statistics_to_file(label, stats, timing_data, current_output_dir, timestamp)


if __name__ == "__main__":
    main()