import logging
import os
from datetime import datetime
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from regress_lm import core
from regress_lm import rlm

# --- ロギング設定 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

# --- 設定値・定数 ---
OUTPUT_DIR = "output"
NUM_SAMPLES = 128
ABNORMAL_THRESHOLD = 10
"""
異常値の定義:
・ここでは「±10」を超える値を異常値と定義します。
・理由: このタスクのyのスケール（例: 0.3, -0.3）から大きく外れており、
  物理的にも不自然な値であるためです。
・運用: 必要に応じて閾値を変更してください。
"""


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
    fname = f"{output_dir}/{timestamp}_{label}_distribution.png"
    plt.savefig(fname)
    plt.close()
    logging.info(f"分布ヒストグラム画像を {fname} に保存しました。")


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
    # 画像保存ディレクトリとタイムスタンプ
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. モデルのインスタンスを作成
    logging.info("ステップ1: RegressLMモデルのインスタンスを作成します...")
    reg_lm = rlm.RegressLM.from_default(max_input_len=2048)
    logging.info("モデルのインスタンス作成が完了しました。")

    # 2. ファインチューニング
    logging.info("ステップ2: ファインチューニング用のデータを準備し、モデルを微調整します...")
    examples = [
        core.Example(x='King', y=0.3),
        core.Example(x='Queen', y=-0.3)
    ]
    reg_lm.fine_tune(examples)
    logging.info("モデルのファインチューニングが完了しました。")

    # 3. 推論
    logging.info("ステップ3: 推論用のクエリを準備します...")
    queries = [
        core.ExampleInput(x='men'),
        core.ExampleInput(x='female')
    ]
    logging.info(f"ステップ4: モデルを使って予測（サンプリング数: {NUM_SAMPLES}）を実行します...")
    results = reg_lm.sample(queries, num_samples=NUM_SAMPLES)
    logging.info("サンプリングが完了しました。")

    # 4. 結果の分析と可視化
    logging.info("--- 予測結果の分析 ---")
    for query, samples in zip(queries, results):
        label = query.x

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


if __name__ == "__main__":
    main()