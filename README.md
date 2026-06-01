# SkyworkRewardModel Wrapper

A clean, production-ready Python wrapper for evaluating text responses using **Skywork Reward Models**. It features automatic hardware placement and built-in memory cleanup to prevent GPU memory leaks.

## Installation

You can install this package directly from GitHub:

```bash
pip install git+agentlans/skywork-reward-model
```

## Quick Start

The recommended way to use the wrapper is with a Python `with` statement. This ensures all GPU memory is automatically freed immediately after use.

```python
from skywork_reward_model import SkyworkRewardModel

# 1. Define your data
prompt = "Explain gravity in one sentence."
responses = [
    "Gravity is the force by which a planet or other body draws objects toward its center.",
    "Gravity is what makes things float away into deep space."
]

# 2. Evaluate with automatic VRAM cleanup
model_path = "Skywork/Skywork-Reward-Llama-3.1-8B"

with SkyworkRewardModel(model_path) as rm:
    scores = rm.evaluate(prompt, responses)

# 3. Print results
for response, score in zip(responses, scores):
    print(f"[{score:+.4f}] {response}")
```

## Manual Cleanup

If you can't use a context manager, you can instantiate the class normally and manually clear the GPU memory when finished:

```python
rm = SkyworkRewardModel("Skywork/Skywork-Reward-Llama-3.1-8B")
scores = rm.evaluate(prompt, responses)

# Explicitly free VRAM, collect garbage, and clear CUDA cache
rm.close()
```

## Licence

MIT
