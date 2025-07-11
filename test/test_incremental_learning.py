import logging
import os
import glob
import yaml
import numpy as np
from sklearn.metrics import mean_squared_error

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
CHECKPOINT_PATH = os.path.join(WORK_DIR, "incremental_learning_checkpoint.pt")
NUM_SAMPLES = 64 # テストの速度を考慮してサンプル数を調整

def load_data_from_files(file_paths: list[str]) -> list[core.Example]:
    """指定されたYAMLファイルのリストからデータを読み込む。"""
    examples = []
    for f_path in file_paths:
        with open(f_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            examples.append(core.Example(x=data['news'], y=data['change_rate']))
    return examples

def evaluate_model(model: rlm.RegressLM, test_examples: list[core.Example]) -> float:
    """テストデータでモデルを評価し、平均二乗誤差（MSE）を返す。"""
    logging.info(f"{len(test_examples)}件のテストデータで評価を実行します...")
    true_ys = [ex.y for ex in test_examples]
    queries = [core.ExampleInput(x=ex.x) for ex in test_examples]
    
    results = model.sample(queries, num_samples=NUM_SAMPLES)
    
    # 予測値はサンプルの平均値とする
    pred_ys = [np.mean(samples) for samples in results]
    
    mse = mean_squared_error(true_ys, pred_ys)
    logging.info(f"評価完了。MSE: {mse:.6f}")
    return mse

def main():
    """インクリメンタル学習のテストを実行するメイン関数。"""
    os.makedirs(WORK_DIR, exist_ok=True)

    # --- データ準備 ---
    train_files = sorted(glob.glob(os.path.join(CLEANED_DATA_DIR, 'train', '*.yml')))
    test_files = sorted(glob.glob(os.path.join(CLEANED_DATA_DIR, 'test', '*.yml')))
    
    # 学習データを2つのフェーズに分割
    phase1_train_files = train_files[:len(train_files) // 2]
    phase2_train_files = train_files[len(train_files) // 2:]
    
    test_examples = load_data_from_files(test_files)

    # ==========================================================================
    # フェーズ1: 初期学習
    # ==========================================================================
    logging.info("""
    --------------------------------------------------
    フェーズ1: 初期学習を開始します... (データ: {}件)
    --------------------------------------------------
    """.format(len(phase1_train_files)))
    
    phase1_examples = load_data_from_files(phase1_train_files)
    model_p1 = rlm.RegressLM.from_default()
    model_p1.fine_tune(phase1_examples, batch_size=1, max_epochs=1)
    model_p1.save_checkpoint(CHECKPOINT_PATH)
    
    loss1 = evaluate_model(model_p1, test_examples)
    del model_p1 # メモリを解放

    # ==========================================================================
    # フェーズ2: 追加学習（インクリメンタル学習）
    # ==========================================================================
    logging.info("""
    --------------------------------------------------
    フェーズ2: 追加学習を開始します... (データ: {}件)
    --------------------------------------------------
    """.format(len(phase2_train_files)))

    phase2_examples = load_data_from_files(phase2_train_files)
    model_p2 = rlm.RegressLM.from_default()
    model_p2.load_checkpoint(CHECKPOINT_PATH) # フェーズ1の学習状態を復元
    model_p2.fine_tune(phase2_examples, batch_size=1, max_epochs=1)
    
    loss2 = evaluate_model(model_p2, test_examples)
    del model_p2 # メモリを解放

    # ==========================================================================
    # 結果の検証
    # ==========================================================================
    logging.info("""
    --------------------------------------------------
    最終検証
    --------------------------------------------------
    """)
    logging.info(f"初期学習後の損失 (MSE): {loss1:.6f}")
    logging.info(f"追加学習後の損失 (MSE): {loss2:.6f}")

    assert loss2 < loss1, f"テスト失敗: 追加学習後に損失が悪化しました。({loss2:.6f} >= {loss1:.6f})"

    logging.info("テスト成功: 追加学習によってモデルの性能が向上しました！")

if __name__ == "__main__":
    main()
