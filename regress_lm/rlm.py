# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Very high-level class for users."""

import logging
from typing import Sequence
import numpy as np
from regress_lm import core
from regress_lm import tokenizers
from regress_lm import vocabs
from regress_lm.models import base as model_base
from regress_lm.models.pytorch import model as pytorch_model
import torch
import yaml
import matplotlib.pyplot as plt


def calculate_statistics(samples: np.ndarray, threshold: float = 10.0) -> dict:
    """サンプリング結果から各種統計量を計算する。"""
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
        "abnormal_count": int(abnormal_count),
        "total_count": int(total_count),
        "abnormal_ratio": abnormal_count / total_count if total_count > 0 else 0,
    }
 
def load_examples_from_yaml(yaml_path: str) -> list[core.Example]:
    """YAMLファイルからExampleオブジェクトのリストを読み込む。"""
    with open(yaml_path, 'r') as f:
        records = yaml.safe_load(f)
    examples: list[core.Example] = []
    for rec in records:
        examples.append(core.Example(x=rec['x'], y=rec.get('y', None)))
    return examples


def save_statistics_and_plot(
    query_text: str,
    samples: np.ndarray,
    output_dir: str,
    performance_stats: dict | None = None
):
    """統計情報をテキストファイルに保存し、分布グラフをプロットする。"""
    stats = calculate_statistics(samples)

    # ファイル名の衝突を避けるため、クエリテキストをサニタイズ
    safe_filename = "".join(c for c in query_text if c.isalnum() or c in (' ', '_')).rstrip()
    if len(safe_filename) > 50: # ファイル名が長くなりすぎないように切り詰める
        safe_filename = safe_filename[:50]
    
    base_filename = f"{output_dir}/{safe_filename}"

    # 統計情報のテキストファイルを作成
    with open(f"{base_filename}_result.txt", "w") as f:
        f.write(f"Query: {query_text}\n\n")
        f.write("--- Statistics ---\n")
        for key, value in stats.items():
            if isinstance(value, float):
                f.write(f"- {key}: {value:.4f}\n")
            else:
                f.write(f"- {key}: {value}\n")
        
        if performance_stats:
            f.write("\n--- Performance ---\n")
            for key, value in performance_stats.items():
                f.write(f"- {key}: {value:.2f}\n")

    # 分布グラフをプロットして保存
    plt.figure()
    plt.hist(samples, bins=20, density=True, alpha=0.7)
    plt.title(f"{query_text[:30]}... samples (mean={stats['mean']:.3f}, std={stats['std']:.3f})")
    plt.xlabel("Predicted y")
    plt.ylabel("Density")
    plt.grid(True)
    plt.savefig(f"{base_filename}_distribution.png")
    plt.close()


class RegressLM:
  """User-facing API for RegressLM."""

  def __init__(self, model: model_base.Model, fine_tuner: model_base.FineTuner):
    self.model = model
    self.fine_tuner = fine_tuner

  def fine_tune(
      self,
      examples: Sequence[core.Example],
      validation_examples: Sequence[core.Example] | None = None,
      **kwargs,
  ):
    self.fine_tuner.fine_tune(examples, validation_examples, **kwargs)

  @classmethod
  def from_default(cls, **kwargs) -> "RegressLM":
    """Creates a RegressLM with default model and finetuner (~60M parameters)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    encoder_vocab = vocabs.SentencePieceVocab.from_t5()
    decoder_vocab = vocabs.DecoderVocab(tokenizers.P10Tokenizer())
    model = pytorch_model.PyTorchModel(
        encoder_vocab=encoder_vocab,
        decoder_vocab=decoder_vocab,
        max_input_len=kwargs.get("max_input_len", 2048),
        learning_rate=kwargs.get("learning_rate", 1e-4),
        d_model=kwargs.get("d_model", 512),
        nhead=kwargs.get("nhead", 8),
        # 変更: 論文規模60Mパラメータに合わせて層数を2→4に増加
        # num_encoder_layers=kwargs.get("num_encoder_layers", 2),  # 旧30M設定
        # num_decoder_layers=kwargs.get("num_decoder_layers", 2),  # 旧30M設定
        num_encoder_layers=kwargs.get("num_encoder_layers", 4),  # 新60M設定
        num_decoder_layers=kwargs.get("num_decoder_layers", 4),  # 新60M設定
        dim_feedforward=kwargs.get("dim_feedforward", 2048),
        dropout=kwargs.get("dropout", 0.0),
        device=device,
    )
    model.to(device)

    fine_tuner = pytorch_model.PyTorchFineTuner(model)
    return cls(model, fine_tuner)

  def sample(
      self, xs: Sequence[core.ExampleInput], num_samples: int
  ) -> Sequence[core.Example]:
    """Samples from the model."""
    examples = self.model.convert_inputs(xs)
    if self.model.device.type == 'cuda':
        prop = torch.cuda.get_device_properties(self.model.device)
        total_mem = prop.total_memory
        mem_gib = total_mem / (1024**3)
        logging.info(f"GPU VRAM total: {mem_gib:.2f} GiB")
        if total_mem <= 8 * 1024**3:
            chunk_size = 32
            logging.info(f"VRAM <=8 GiB, sampling in chunks of {chunk_size}")
            chunks = []
            for start in range(0, num_samples, chunk_size):
                c = min(chunk_size, num_samples - start)
                logging.debug(f"Sampling chunk: start={start}, size={c}")
                _, chunk_out = self.model.decode(examples, c)
                chunks.append(chunk_out)
            output_floats = np.concatenate(chunks, axis=1)
        else:
            logging.info(f"VRAM >8 GiB, sampling all {num_samples} samples at once")
            _, output_floats = self.model.decode(examples, num_samples)
    else:
        logging.info(f"CPU mode, sampling all {num_samples} samples")
        _, output_floats = self.model.decode(examples, num_samples)
    
    # y_samplesを各ExampleInputに割り当てる
    y_samples_list = [y.squeeze(axis=0) for y in np.split(output_floats, len(xs), axis=0)]
    
    results = []
    for example_input, y_samples in zip(xs, y_samples_list):
        results.append(core.Example(x=example_input.x, y=None, y_samples=y_samples))
        
    return results
