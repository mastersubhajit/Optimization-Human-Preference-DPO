---
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
tags:
  - dpo
  - alignment
  - truthfulness
  - lora
  - peft
  - qwen2
datasets:
  - jondurbin/truthy-dpo-v0.1
pipeline_tag: text-generation
library_name: peft
---

# Qwen2.5-1.5B-Instruct — DPO Fine-tuned for Truthfulness

This is a LoRA adapter for [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), fine-tuned using Direct Preference Optimization (DPO) to reduce hallucinations and improve factual accuracy.

I trained this as part of my NLU course assignment (A5) at AIT, where the goal was to align a language model to prefer truthful responses over hallucinated ones.

## What does this model do differently?

The base Qwen2.5-1.5B-Instruct model occasionally generates plausible-sounding but incorrect answers. By training with DPO on the [truthy-dpo](https://huggingface.co/datasets/jondurbin/truthy-dpo-v0.1) dataset, this adapter nudges the model toward choosing factually grounded responses instead of making things up.

It's not a massive overhaul — it's a lightweight LoRA adapter that adjusts the attention layers to be slightly more cautious and accurate.

## Training Details

| Parameter | Value |
|-----------|-------|
| Base model | Qwen/Qwen2.5-1.5B-Instruct |
| Training method | DPO (Direct Preference Optimization) |
| Dataset | jondurbin/truthy-dpo-v0.1 (1016 samples) |
| Adapter type | LoRA (rank=8, alpha=16) |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Trainable params | ~2.18M (0.14% of total) |
| Learning rate | 1e-4 |
| Beta (DPO) | 0.3 |
| Training steps | 50 |
| Batch size | 1 (with gradient accumulation = 4) |
| Precision | float32 |
| Hardware | Apple M2 (MPS) |

I ran two experiments — one with a conservative setup (beta=0.1, lr=5e-5) and another with stronger preference signal (beta=0.3, lr=1e-4). The second experiment showed better loss reduction, so that's what I saved here.

## How to Use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

# load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype="auto",
    trust_remote_code=True,
)

# load the DPO adapter on top
model = PeftModel.from_pretrained(base_model, "mastersubhajit/DPO")
model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained("mastersubhajit/DPO")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

response = pipe("What is the capital of France?", max_new_tokens=256, do_sample=False)
print(response[0]["generated_text"])
```

## Evaluation

I evaluated this model using AlpacaEval with an LLM-as-a-Judge setup. I took 15 random samples from the `helpful_base` subset and had both the base model and this DPO model generate responses. Then I used Llama 3.3 70B (via Groq) as a judge to compare them.

**DPO Win Rate: ~60%**

The DPO model won more comparisons than the base model, especially on factual and knowledge-based questions. On creative or open-ended prompts the difference was smaller, which makes sense since the training data specifically targets hallucination reduction rather than general helpfulness.

## Limitations

- This is a LoRA adapter, not a full fine-tune. The changes are subtle.
- Trained for only 50 steps due to hardware constraints (M2 Mac). More training would likely help.
- The truthy-dpo dataset focuses on a specific kind of truthfulness — the model won't suddenly become perfect at everything.
- float32 training (MPS doesn't fully support fp16 for all ops), so training was slower than it would be on a GPU with fp16/bf16.

## Acknowledgements

- Base model by [Qwen Team](https://huggingface.co/Qwen)
- Training dataset by [Jon Durbin](https://huggingface.co/jondurbin)
- Built using [TRL](https://huggingface.co/docs/trl) and [PEFT](https://huggingface.co/docs/peft) libraries