# RegressLM: Easy Text-to-Text Regression
[![Continuous Integration](https://github.com/google-deepmind/regress-lm/actions/workflows/core_test.yml/badge.svg)](https://github.com/google-deepmind/regress-lm/actions?query=branch%3Amain)

## Overview
RegressLM is a library for text-to-text regression, applicable to any input
string representation and allows pretraining and fine-tuning over multiple
regression tasks.

<figure>
<p align="center" width=65%>
<img src="https://raw.githubusercontent.com/akhauriyash/figures_placeholder/refs/heads/main/teaser_rlm_compressed.gif" alt="RegressLM decoding a numerical performance metric from text."/>
  <br>
  <figcaption style="text-align: center;"><em><b><a href="https://arxiv.org/abs/2506.21718">Example Application</a>: Directly regressing performance metrics from unstructured, textually represented system states from Google's massive compute clusters.</b></em></figcaption>
</p>
</figure>

## Environment Setup
- Dev container setup.
- venv setup
+Note: このプロジェクトでは `python3 -m venv` と `uv` CLI を使って仮想環境作成と依存管理を行います。
1. 仮想環境を .venv ディレクトリに作成
```sh
python3 -m venv .venv
```
2. 仮想環境を有効化
```sh
source .venv/bin/activate
```
3. 開発モードでインストール
   - **CPU版をインストールする場合:**
     ```sh
     uv pip install -e ".[extras]"
     ```
   - **GPU版 (CUDA 12.1) をインストールする場合:**
     PyTorchのGPU対応版をインストールするために、追加のフラグが必要です。
     ```sh
     uv pip install -e ".[extras]" --extra-index-url https://download.pytorch.org/whl/cu121
     ```
   - .[extras] は requirements.txt と requirements-extras.txt の依存関係をインストールします。

## TODO

### Phase 1: PoCと基本機能の検証 (完了)
- [x] 60Mパラメータモデルでの動作確認
- [x] GPU OOMエラーの解決（学習・推論時のバッチ処理化）
- [x] 金融ニュース（ポジティブ）でのPoCテスト (`finance_01`)
- [x] データリーク問題を特定・修正しての再テストと評価

### Phase 2: 汎化能力向上のためのテスト設計と実践
- [ ] **テストケースの再設計と方針策定**
    - **基本方針**: モデルの真の文脈理解能力を測定するため、安易な数値マッピング（例: `売上5%減` -> `y=-0.05`）を避け、より現実世界の複雑さを反映したデータセットを構築する。これにより、モデルが安易なヒューリスティックに頼る「ショートカット学習」を防ぎ、汎化能力を高める。
    - **具体的アプローチ**:
        1.  **市場期待値との乖離**: 「予想を上回る売上」「コンセンサス未達」など、市場の期待値との比較を含むニュースを導入する。
        2.  **文脈依存性の導入**: 同じニュースでも企業規模や市況によって影響度が変わるケースを反映させる。
        3.  **定性的表現の強化**: 直接的な数値を含まない、より定性的な表現（例: 「画期的な新技術の発表」「CEOの突然の辞任」）の比重を増やす。

### Phase 3: データオーグメンテーションの検討
- [ ] **データオーグメンテーションによる精度・汎化性能向上の検証**
  - **目的**: 少量データ環境下での回帰精度・文脈強弱推定能力の向上。
  - **手法例**:
    1.  変動の強弱を多様な形容詞・副詞（slightly, sharply, significantly, modestly など）で表現し、同じchange_rateでも複数の言い換えパターンを作成。
    2.  比較表現や程度表現（"slight improvement", "record-breaking surge" など）を導入。
    3.  変動理由や背景文脈を加え、強弱ニュアンスを補強。
  - **考慮点**:
    - オーグメンテーションには手作業や工数がかかるため、データ件数を増やす場合とのコストパフォーマンス比較が重要。
    - データ件数が十分に多い場合は、前処理の必要性が下がる可能性がある。
    - ユースケースや運用要件に応じて、最適なバランスを検討する。
- [ ] **方針に基づくネガティブ・ニュースデータセットの構築とテスト (`finance_02`)**
- [ ] **異なるドメインでの汎化性能テスト**
    - 例：製品レビューの星評価予測、不動産の説明文からの価格予測など、金融以外のドメインでモデルの適応力を評価する。
- [ ] **ハイパーパラメータの体系的なチューニング**
    - 学習率、エポック数、バッチサイズなどを調整し、性能への影響を体系的に評価する。
  - [ ] **テスト結果の自動集計とレポート機能**
    - 複数テストの結果を自動で集計し、比較・分析を容易にするスクリプトを整備する。


## High-Level Design Overview
- Centralize all path definitions in a new module `regress_lm/config.py`:
  - `PROJECT_ROOT`, `OUTPUT_ROOT`, `MODEL_ROOT`, etc.
- CLI entrypoints separated or unified via subcommands (`train` / `infer`), with generic names (e.g. `run_finetuning.py`).
- Dynamic data-labeling: derive `data_label` from the basename of `--data-dir`, not hard-coded.
- Consistent directory structure per run:
  ```
  output/<data_label>/<timestamp>/
  saved_models/<data_label>/<timestamp>/
  ```
- Domain-agnostic script names and arguments (`--data-dir`, `--model-dir`, `--output-dir`).

## Pending Implementation TODO
- [ ] Create `regress_lm/config.py` to hold common path constants and environment overrides
- [ ] Refactor sample scripts into generic CLI with subcommands for training and inference
- [ ] Implement incremental fine-tuning support (`--additional-data` or `train --resume`) in training workflow
- [ ] Update argument parsing to accept `--data-dir`, `--model-dir`, `--output-dir` and remove finance-specific defaults
- [ ] Automate `data_label` extraction from provided data directory name
- [ ] Update documentation and usage examples in README to reflect new design

## Implementation Details

### Model Initialization
- Calling `RegressLM.from_default(max_input_len=...)` always creates a new `PyTorchModel` with standard PyTorch weight initialization.

### Incremental Fine-Tuning Support
- After fine-tuning, save the model state:
  ```python
  torch.save(reg_lm._model.state_dict(), "path/to/checkpoint.pt")
  ```
- To resume on additional data:
  ```python
  reg_lm = RegressLM.from_default(max_input_len=...)
  state = torch.load("path/to/checkpoint.pt")
  reg_lm._model.load_state_dict(state)
  reg_lm.fine_tune(additional_examples)
  ```

## Usage
There are two main stages: **inference** and **pretraining** (optional).

## Inference
The intended use-case is to import a RegressLM class, which can decode
floating-point predictions from a given input, and also fine-tune against new
data.

```python
from regress_lm import core
from regress_lm import rlm

# Create RegressLM with max input token length.
reg_lm = rlm.RegressLM.from_default(max_input_len=2048)

# Example (x,y) pairs, which can be fine-tuned against.
examples = [core.Example(x='hello', y=0.3), core.Example(x='world', y=-0.3)]
reg_lm.fine_tune(examples)

# Query inputs.
query1, query2 = core.ExampleInput(x='hi'), core.ExampleInput(x='bye')
samples1, samples2 = reg_lm.sample([query1, query2], num_samples=128)
```

## Pretraining
To produce better initial checkpoints for transfer learning, we recommend
the user pretrains over large amounts of their own training data. Example
pseudocode with PyTorch:

```python
from torch import optim
from regress_lm.models.pytorch import model as torch_model_lib

model = torch_model_lib.PyTorchModel(...)
optimizer = optim.Adafactor(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
)
for _ in range(...):
  examples = [Example(x=..., y=...), ...]
  tensor_examples = model.convert(examples)
  optimizer.zero_grad()
  loss, _ = model.compute_loss_and_metrics(tensor_examples)
  loss.backward()
  optimizer.step()
```

## Contributors and Citation

The codebase was written by: Xingyou Song, Yash Akhauri, Dara Bahri, Michal
Lukasik, Arissa Wongpanich, Adrian N. Reyes, and Bryan Lewandowski.

If you find this project useful, please consider citing our work as follows:

```
@article{akhauri2025performancepredictionlargesystems,
      title={Performance Prediction for Large Systems via Text-to-Text Regression},
      author={Yash Akhauri and Bryan Lewandowski and Cheng-Hsi Lin and Adrian N. Reyes and Grant C. Forbes and Arissa Wongpanich and Bangding Yang and Mohamed S. Abdelfattah and Sagi Perel and Xingyou Song},
      journal={arXiv preprint arXiv:2506.21718},
      year={2025}
}
```

**Disclaimer:** This is not an officially supported Google product.