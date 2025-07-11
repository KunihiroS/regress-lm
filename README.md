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
- venv setup  
Note: このプロジェクトでは `python3 -m venv` と `uv` CLI を使って仮想環境作成と依存管理を行います。
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

## 開発ロードマップ (Development Roadmap)

### ステップ1: モデル保存と継続的ファインチューニング基盤の構築 (最優先)

モデルのライフサイクルを管理し、継続的な改善を可能にするための基盤を整備します。

-   **タスクリスト:**
    -   [ ] **汎用的なCLIの作成**:
        -   `run_cli.py` のような単一のエントリーポイントを作成し、`train` と `infer` のサブコマンドを実装する。
    -   [ ] **引数の標準化**:
        -   `--data-dir`, `--output-dir`, `--resume-from-checkpoint <path>` のような、ドメインに依存しない標準的な引数を整備する。
    -   [ ] **一貫したディレクトリ構造の導入**:
        -   学習済みモデルは `saved_models/<data_label>/<timestamp>/` の形式で保存する。
        -   `data_label` は `--data-dir` のディレクトリ名から自動で抽出する。
    -   [ ] **安全な追加学習の実現**:
        -   追加学習時、読み込むチェックポイントのパス（例: `.../finance_positive/...`）と、新しいデータのパス（例: `.../finance_positive/`）からデータラベルを比較し、**不一致の場合はエラーで停止**する検証ロジックを実装する。
    -   [ ] **ドキュメントの更新**:
        -   新しいCLIの使用方法を `Usage` セクションに反映する。

### ステップ2: モデルの汎化能力向上と評価

基盤の上で、モデルの性能を本格的に向上させるための実験を行います。

-   **タスクリスト:**
    -   [ ] **ショートカット学習を回避するテストケースの設計**:
        -   市場期待値との乖離や、定性的な表現を含む、より複雑なデータセットを構築・評価する。
    -   [ ] **異種ドメインでの性能テスト**:
        -   金融ニュース以外のデータ（例: 製品レビュー）でモデルの適応力を評価する。
    -   [ ] **ハイパーパラメータの体系的なチューニング**:
        -   学習率やバッチサイズなどを調整し、性能への影響を評価する。

### ステップ3: データ拡張と応用範囲の拡大

より少ないデータで高い性能を引き出すための手法を検討します。

-   **タスクリスト:**
    -   [ ] **データオーグメンテーション手法の検証**:
        -   同義語の言い換えや背景情報の付与など、データの多様性を増すアプローチを評価する。

## On processing task
- **安全な追加学習の実現 (チェックポイント機能の安定化)**:
    - **課題**: モデルとファインチューナー間でオプティマイザの状態が一致せず、学習の再開に失敗する問題があった。
    - **修正**:
        - オプティマイザの管理を`PyTorchModel`に一元化し、設計上の不整合を解消。
        - チェックポイント読み込み時に、モデルとファインチューナーのオプティマイザが必ず同期するように修正。
    - **検証**:
        - `Adafactor`オプティマイザの非決定的な挙動（浮動小数点誤差）を許容する、信頼性の高いテスト (`test_checkpoint_io.py`) を新規に作成し、合格を確認。
    - **その他**:
        - テストで生成される巨大なチェックポイントファイル (`~300MB`) がGitの追跡対象外となるよう`.gitignore`を更新し、リポジトリへのpush失敗問題を解決。

## 完了済みタスク (Completed Tasks)

### Phase 1: PoCと基本機能の検証
- [x] 60Mパラメータモデルでの動作確認
- [x] GPU OOMエラーの解決（学習・推論時のバッチ処理化）
- [x] 金融ニュース（ポジティブ）でのPoCテスト (`finance_01`)
- [x] データリーク問題を特定・修正しての再テストと評価

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

## Repository Structure

-   `/regress_lm`: Main source code for the `regress_lm` package.
    -   `rlm.py`: Core `RegressLM` class.
    -   `models/pytorch`: PyTorch model implementations.
-   `/test`: Test scripts and data.
    -   `test_checkpoint_io.py`: Unit test for checkpoint functionality.
    -   `finance_*.py`: End-to-end tests for different scenarios.
-   `/work`: Working directory for checkpoints and other artifacts (ignored by Git).
-   `/output`: Directory for inference results like graphs (ignored by Git).
-   `README.md`: This file.

**Disclaimer:** This is not an officially supported Google product.