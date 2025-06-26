# RegressLM: Easy Text-to-Text Regression
[![Continuous Integration](https://github.com/google-deepmind/regress-lm/actions/workflows/core_test.yml/badge.svg)](https://github.com/google-deepmind/regress-lm/actions?query=branch%3Amain)

## Overview
RegressLM is a library for text-to-text regression, applicable to any input
string representation and allows pretraining and fine-tuning over multiple
regression tasks.

## Usage
There are two main stages: **inference** and **pretraining** (optional).

## Inference
The overall use-case is to import a RegressLM class, which can decode
floating-point predictions from a given input, and also gradient update against
new data.

```python
from regress_lm import core
from regress_lm import rlm

# Create RegressLM with max input token lengths.
reg_lm = rlm.RegressLM.from_default()

# Example (x,y) pairs, which can be fine-tuned against.
examples = [core.Example(x='hello', y=0.3), core.Example(x='world', y=-0.3)]
reg_lm.fine_tune(examples)

# Query inputs.
query1, query2 = core.ExampleInput(x='hi'), core.ExampleInput(x='bye')
samples1, samples2 = model.sample([query1, query2], num_samples=128)
```

## Pretraining
To start with a better initial checkpoint for transfer learning, we recommend
the user writes their own standard pretraining script over large amounts of
training data. Example pseudocode with PyTorch:

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

**Disclaimer:** This is not an officially supported Google product.