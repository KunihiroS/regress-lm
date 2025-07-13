import logging
import os
import glob
import yaml
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime
import time
from typing import Dict, List

from regress_lm import core
from regress_lm import rlm

# --- ロギング設定 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s',
)

# --- 定数・パス設定 ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED_DATA_DIR = os.path.join(PROJECT_ROOT, "test/cleaned_data")
WORK_DIR = os.path.join(PROJECT_ROOT, "work")
OUTPUT_DIR = os.path.join(WORK_DIR, "test_incremental_learning_output")
CHECKPOINT_PATH = os.path.join(WORK_DIR, "incremental_learning_checkpoint.pt")
NUM_SAMPLES = 64
ABNORMAL_THRESHOLD = 5.0

# --- ヘルパー関数 (finance_04.pyから移植) ---

def sanitize_label_for_filename(label: str) -> str:
    sanitized = "".join(c if c.isalnum() else "_" for c in label)
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized.strip('_')[:50]

def calculate_statistics(samples: List[float], threshold: float) -> Dict[str, float]:
    samples = np.array(samples)
    total_count = len(samples)
    abnormal_mask = np.abs(samples) > threshold
    abnormal_count = np.sum(abnormal_mask)
    hist, bin_edges = np.histogram(samples, bins=20, density=True)
    hist = hist + 1e-12
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
    plt.figure()
    plt.hist(samples, bins=20, density=True, alpha=0.7)
    plt.title(f"{label} (mean={stats['mean']:.3f}, std={stats['std']:.3f})")
    plt.xlabel("Predicted y")
    plt.ylabel("Density")
    plt.grid(True)
    safe_label = sanitize_label_for_filename(label)
    fname = f"{output_dir}/{timestamp}_{safe_label}_distribution.png"
    plt.savefig(fname)
    plt.close()
    logging.info(f"分布ヒストグラム画像を {fname} に保存しました。")

def save_statistics_to_file(label: str, stats: Dict[str, float], timing_data: Dict[str, float], output_dir: str, timestamp: str):
    safe_label = sanitize_label_for_filename(label)
    fname = f"{output_dir}/{timestamp}_{safe_label}_result.txt"
    with open(fname, 'w', encoding='utf-8') as f:
        f.write(f"--- Test Phase: {label} ---\n\n")
        f.write("--- Statistics ---\n")
        for key, value in stats.items():
            f.write(f"- {key}: {value:.4f}\n" if isinstance(value, float) else f"- {key}: {value}\n")
        f.write("\n--- Performance ---\n")
        for key, value in timing_data.items():
            f.write(f"- {key}: {value:.2f} sec\n")
    logging.info(f"統計情報と処理時間を {fname} に保存しました。")

# --- データおよび評価関連の関数 ---

def load_data_from_files(file_paths: list[str]) -> list[core.Example]:
    examples = []
    for f_path in file_paths:
        with open(f_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            examples.append(core.Example(x=data['news'], y=data['change_rate']))
    return examples

def evaluate_model(model: rlm.RegressLM, test_examples: list[core.Example]) -> dict:
    logging.info(f"{len(test_examples)}件のテストデータで評価を実行します...")
    true_ys = np.array([ex.y for ex in test_examples])
    
    # tune.pyでの成功事例に基づき、decodeを直接呼び出し、結果をflattenする
    inputs = model.model.convert_examples(test_examples)
    _, output_floats = model.model.decode(inputs, num_samples=1)
    pred_ys = output_floats.flatten()

    mse = mean_squared_error(true_ys, pred_ys)
    logging.info(f"評価完了。MSE: {mse:.6f}")

    stats = calculate_statistics(pred_ys, ABNORMAL_THRESHOLD)
    stats['mse'] = mse

    return {
        "stats": stats,
        "predictions": pred_ys,
    }

# --- メイン実行関数 ---

def main():
    """インクリメンタル学習のテストを実行し、詳細なレポートを生成する。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- データ準備 ---
    train_files = sorted(glob.glob(os.path.join(CLEANED_DATA_DIR, 'train', '*.yml')))
    test_files = sorted(glob.glob(os.path.join(CLEANED_DATA_DIR, 'test', '*.yml')))
    phase1_train_files = train_files[:len(train_files) // 2]
    phase2_train_files = train_files[len(train_files) // 2:]
    test_examples = load_data_from_files(test_files)

    # ==========================================================================
    # フェーズ1: 初期学習
    # ==========================================================================
    logging.info(f"--- フェーズ1: 初期学習を開始 (データ: {len(phase1_train_files)}件) ---")
    timing_data_p1 = {}
    start_time = time.time()

    phase1_examples = load_data_from_files(phase1_train_files)
    model_p1 = rlm.RegressLM.from_default()
    model_p1.fine_tune(phase1_examples, batch_size=1, max_epochs=1)
    model_p1.save_checkpoint(CHECKPOINT_PATH)
    timing_data_p1['training'] = time.time() - start_time

    start_time = time.time()
    eval_results_p1 = evaluate_model(model_p1, test_examples)
    timing_data_p1['evaluation'] = time.time() - start_time

    plot_distribution(eval_results_p1['predictions'], "Phase1_Initial_Training", eval_results_p1['stats'], OUTPUT_DIR, timestamp)
    save_statistics_to_file("Phase1_Initial_Training", eval_results_p1['stats'], timing_data_p1, OUTPUT_DIR, timestamp)
    loss1 = eval_results_p1['stats']['mse']
    del model_p1

    # ==========================================================================
    # フェーズ2: 追加学習
    # ==========================================================================
    logging.info(f"--- フェーズ2: 追加学習を開始 (データ: {len(phase2_train_files)}件) ---")
    timing_data_p2 = {}
    start_time = time.time()

    phase2_examples = load_data_from_files(phase2_train_files)
    model_p2 = rlm.RegressLM.from_default()
    model_p2.load_checkpoint(CHECKPOINT_PATH)
    model_p2.fine_tune(phase2_examples, batch_size=1, max_epochs=1)
    timing_data_p2['training'] = time.time() - start_time

    start_time = time.time()
    eval_results_p2 = evaluate_model(model_p2, test_examples)
    timing_data_p2['evaluation'] = time.time() - start_time

    plot_distribution(eval_results_p2['predictions'], "Phase2_Incremental_Training", eval_results_p2['stats'], OUTPUT_DIR, timestamp)
    save_statistics_to_file("Phase2_Incremental_Training", eval_results_p2['stats'], timing_data_p2, OUTPUT_DIR, timestamp)
    loss2 = eval_results_p2['stats']['mse']
    del model_p2

    # ==========================================================================
    # 最終検証
    # ==========================================================================
    logging.info("--- 最終検証 ---")
    logging.info(f"初期学習後の損失 (MSE): {loss1:.6f}")
    logging.info(f"追加学習後の損失 (MSE): {loss2:.6f}")
    assert loss2 < loss1, f"テスト失敗: 追加学習後に損失が悪化しました。({loss2:.6f} >= {loss1:.6f})"
    logging.info("テスト成功: 追加学習によってモデルの性能が向上しました！")

if __name__ == "__main__":
    main()
