from regress_lm import core
from regress_lm import rlm

# 1. モデルのインスタンスを作成します。
# from_default() は事前学習済みのデフォルトモデルを読み込みます。
reg_lm = rlm.RegressLM.from_default(max_input_len=2048)

# 2. ファインチューニング（微調整）のための学習用サンプルデータ (入力テキスト x, 目標数値 y) を準備します。
examples = [
    core.Example(x='hello', y=0.3),
    core.Example(x='world', y=-0.3)
]
reg_lm.fine_tune(examples) # 準備したデータでモデルをファインチューニングします。

# 3. 推論（予測）したい新しいテキスト入力を準備します。
query1 = core.ExampleInput(x='hi')
query2 = core.ExampleInput(x='bye')

# 4. sample() メソッドを使って、各入力に対する予測結果をサンプリングします。
samples1, samples2 = reg_lm.sample([query1, query2], num_samples=128)

print("hi samples:", samples1)
print("bye samples:", samples2)