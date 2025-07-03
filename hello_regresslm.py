import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from regress_lm import core
from regress_lm import rlm

# --- ロギング設定 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

# 画像保存ディレクトリとタイムスタンプ
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 1. モデルのインスタンスを作成します。
logging.info("ステップ1: RegressLMモデルのインスタンスを作成します...")
reg_lm = rlm.RegressLM.from_default(max_input_len=2048)
logging.info("モデルのインスタンス作成が完了しました。")

# 2. ファインチューニング（微調整）のための学習用サンプルデータ (入力テキスト x, 目標数値 y) を準備します。
logging.info("ステップ2: ファインチューニング用のデータを準備し、モデルを微調整します...")
examples = [
    core.Example(x='hello', y=0.3),
    core.Example(x='world', y=-0.3)
]
reg_lm.fine_tune(examples)
logging.info("モデルのファインチューニングが完了しました。")

# 3. 推論（予測）したい新しいテキスト入力を準備します。
logging.info("ステップ3: 推論用のクエリを準備します...")
query1 = core.ExampleInput(x='hi')
query2 = core.ExampleInput(x='bye')

# 4. sample() メソッドを使って、各入力に対する予測結果をサンプリングします。
logging.info("ステップ4: モデルを使って予測（サンプリング）を実行します...")
samples1, samples2 = reg_lm.sample([query1, query2], num_samples=128)
logging.info("サンプリングが完了しました。")

logging.info("--- 予測結果 ---")
print("hi samples:", samples1)
print("bye samples:", samples2)

# 5. 分布の可視化と確かさ(標準偏差・エントロピー)の計算
def plot_and_report(samples, label):
    mean = np.mean(samples)
    std = np.std(samples)
    # エントロピーも可視化指標として計算（ヒストグラム分布のエントロピー近似）
    hist, bin_edges = np.histogram(samples, bins=20, density=True)
    hist = hist + 1e-12  # log(0)回避
    entropy = -np.sum(hist * np.log(hist)) * (bin_edges[1] - bin_edges[0])

    logging.info(f"{label}: 平均={mean:.3f}, 標準偏差={std:.3f}, ヒストグラム近似エントロピー={entropy:.3f}")

    plt.figure()
    plt.hist(samples, bins=20, density=True, alpha=0.7)
    plt.title(f"{label} samples (mean={mean:.3f}, std={std:.3f}, entropy={entropy:.3f})")
    plt.xlabel("Predicted y")
    plt.ylabel("Density")
    plt.grid(True)
    fname = f"{output_dir}/{timestamp}_{label}_distribution.png"
    plt.savefig(fname)
    plt.close()
    logging.info(f"{label} の分布ヒストグラム画像を {fname} に保存しました。")

plot_and_report(samples1, "hi")
plot_and_report(samples2, "bye")