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
  <figcaption style="text-align: center;"><em><b>Example Application: Directly regressing performance metrics from unstructured, textually represented system states from Google's compute clusters.</b></em></figcaption>
</p>
</figure>

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
samples1, samples2 = model.sample([query1, query2], num_samples=128)
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

## Authors and Contributors
Xingyou Song, Yash Akhauri, Dara Bahri, Michal Lukasik, Arissa Wongpanich,
Adrian N. Reyes, Bryan Lewandowski

**Disclaimer:** This is not an officially supported Google product.