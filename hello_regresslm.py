import logging
from regress_lm import core
from regress_lm import rlm

# --- ロギング設定 ---
# ログのフォーマットとレベルを設定します。これにより、どの処理がいつ実行されたかが分かりやすくなります。
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

# 1. モデルのインスタンスを作成します。
# from_default() は事前学習済みのデフォルトモデルを読み込みます。
logging.info("ステップ1: RegressLMモデルのインスタンスを作成します...")
reg_lm = rlm.RegressLM.from_default(max_input_len=2048)
logging.info("モデルのインスタンス作成が完了しました。")

# 2. ファインチューニング（微調整）のための学習用サンプルデータ (入力テキスト x, 目標数値 y) を準備します。
logging.info("ステップ2: ファインチューニング用のデータを準備し、モデルを微調整します...")
examples = [
    core.Example(x='hello', y=0.3),
    core.Example(x='world', y=-0.3)
]
reg_lm.fine_tune(examples)  # 準備したデータでモデルをファインチューニングします。
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