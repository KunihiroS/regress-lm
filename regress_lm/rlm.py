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

    fine_tuner = pytorch_model.PyTorchFineTuner(model=model, optimizer=model.optimizer)
    return cls(model, fine_tuner)

  def save_checkpoint(self, path: str):
    """モデルの状態とオプティマイザの状態をファイルに保存する。"""
    logging.info(f"チェックポイントを {path} に保存しています...")
    torch.save({
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self.model.optimizer.state_dict(),
        'model_kwargs': self.model.encoder_decoder.hparams
    }, path)
    logging.info("チェックポイントの保存が完了しました。")

  def load_checkpoint(self, path: str):
    """ファイルからモデルの状態とオプティマイザの状態を読み込む。"""
    logging.info(f"チェックポイントを {path} から読み込んでいます...")
    checkpoint = torch.load(path, map_location=self.model.device)
    
    # モデルのアーキテクチャが保存されたものと一致するかを簡易的に確認
    saved_kwargs = checkpoint.get('model_kwargs', {})
    current_kwargs = self.model.encoder_decoder.hparams
    if saved_kwargs and saved_kwargs != current_kwargs:
        logging.warning("警告: モデルのパラメータがチェックポイントと一致しません。")
        logging.warning(f"  - 保存されたパラメータ: {saved_kwargs}")
        logging.warning(f"  - 現在のモデルパラメータ: {current_kwargs}")

    self.model.load_state_dict(checkpoint['model_state_dict'])
    self.model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # fine_tunerがロードされたオプティマイザを確実に参照するように設定
    self.fine_tuner.optimizer = self.model.optimizer
    # fine_tunerがロードされたオプティマイザを確実に参照するように設定
    self.fine_tuner.optimizer = self.model.optimizer
    self.model.to(self.model.device)
    logging.info("チェックポイントの読み込みが完了しました。")

  def sample(
      self, xs: Sequence[core.ExampleInput], num_samples: int
  ) -> Sequence[np.ndarray]:
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
    return [y.squeeze(axis=0) for y in np.split(output_floats, len(xs), axis=0)]
