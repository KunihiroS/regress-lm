import os
import torch
import logging
import sys

# プロジェクトのルートディレクトリをPythonのパスに追加
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from regress_lm import rlm
from regress_lm import core

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compare_models(model1: rlm.RegressLM, model2: rlm.RegressLM) -> bool:
    """2つのモデルのstate_dictとオプティマイザの状態を比較する"""
    # 1. モデルパラメータの比較（完全一致）
    logging.info("Comparing model parameters...")
    param_state1 = model1.model.state_dict()
    param_state2 = model2.model.state_dict()

    if param_state1.keys() != param_state2.keys():
        logging.error("Model parameter keys do not match.")
        return False

    for key in param_state1:
        if not torch.equal(param_state1[key], param_state2[key]):
            logging.error(f"Model parameter tensor for key '{key}' does not match.")
            return False
    logging.info("Model parameters match successfully.")

    # 2. オプティマイザ状態の比較（テンソルは近似一致）
    logging.info("Comparing optimizer states...")
    opt_state1 = model1.model.optimizer.state_dict()
    opt_state2 = model2.model.optimizer.state_dict()

    def compare_optimizer_states(s1, s2):
        if s1.keys() != s2.keys():
            logging.error("Optimizer state keys do not match.")
            return False
        for k in s1:
            if isinstance(s1[k], dict):
                if not compare_optimizer_states(s1[k], s2[k]):
                    return False
            elif isinstance(s1[k], torch.Tensor):
                if not torch.allclose(s1[k], s2[k]):
                    logging.error(f"Optimizer tensor for key '{k}' does not match.")
                    logging.error(f"State 1: {s1[k]}")
                    logging.error(f"State 2: {s2[k]}")
                    return False
            elif s1[k] != s2[k]:
                logging.error(f"Optimizer value for key '{k}' does not match: {s1[k]} vs {s2[k]}.")
                return False
        return True

    if not compare_optimizer_states(opt_state1, opt_state2):
        logging.error("Optimizer states do not match.")
        return False
        
    logging.info("Optimizer states match successfully.")

    return True

def main():
    """チェックポイントの保存・読み込み機能の軽量テスト"""
    logging.info("--- Starting Checkpoint I/O Test ---")

    checkpoint_dir = os.path.join(PROJECT_ROOT, "work", "temp_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "test_checkpoint.pth")

    try:
        # --- 1. 元のモデルを作成し、状態を保存 --- #
        logging.info("Step 1: Creating the original model and saving checkpoint.")
        model_orig = rlm.RegressLM.from_default()
        
        # オプティマイザの状態を初期化するために、ダミーデータで一度だけ学習ステップを実行
        dummy_example = [core.Example(x="dummy", y=1.0)]
        model_orig.fine_tune(dummy_example, max_epochs=1, batch_size=1)

        model_orig.save_checkpoint(checkpoint_path)
        logging.info(f"Original model saved to {checkpoint_path}")

        # --- 2. 新しいモデルを作成し、状態を読み込み --- #
        logging.info("Step 2: Creating a new model and loading from checkpoint.")
        model_loaded = rlm.RegressLM.from_default()
        model_loaded.load_checkpoint(checkpoint_path)
        logging.info("New model loaded from checkpoint.")

        # --- 3. 2つのモデルの状態を比較 --- #
        logging.info("Step 3: Comparing the original and loaded models.")
        if compare_models(model_orig, model_loaded):
            logging.info("[SUCCESS] The loaded model's state perfectly matches the original.")
        else:
            logging.error("[FAILURE] The loaded model's state does not match the original.")

    finally:
        # --- 4. クリーンアップ --- #
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            logging.info(f"Cleaned up temporary checkpoint file: {checkpoint_path}")

if __name__ == "__main__":
    main()
