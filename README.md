# A5: Optimization Human Preference & LLM-as-a-Judge

**Name:** Subhajit Ghosh

This project explores aligning a language model to be more truthful using Direct Preference Optimization (DPO), and then evaluating the results with an LLM-as-a-Judge pipeline.

## What's this about?

Large language models sometimes "hallucinate" — they generate answers that sound right but are actually wrong. DPO is a technique that teaches the model to prefer factually correct responses over hallucinated ones, without needing a separate reward model.

In this assignment, I fine-tuned Qwen2.5-1.5B-Instruct on the [truthy-dpo](https://huggingface.co/datasets/jondurbin/truthy-dpo-v0.1) dataset using LoRA adapters, then compared the base model against the fine-tuned one using Llama 3.3 70B as an automated judge.

## Project Structure

```
A5/
├── A5.ipynb                  # main notebook with all code and results
├── README.md                 # this file
├── .gitignore
└── final_dpo_model/          # saved LoRA adapter weights
    ├── README.md             # HuggingFace model card
    ├── adapter_config.json
    └── adapter_model.safetensors
```

## How to Run

### Prerequisites
- Python 3.12
- Apple M2 Mac (or any machine with ~8GB RAM)

### Setup
```bash
python -m venv .venv
source .venv/bin/activate
uv pip install torch transformers datasets trl peft openai matplotlib pandas
```

### Running the Notebook
Open `A5.ipynb` in Jupyter or VS Code and run cells in order. The notebook is organized into four tasks:

1. **Task 1** — Load and explore the truthy-dpo dataset
2. **Task 2** — Train two DPO experiments with different hyperparameters
3. **Task 3** — Save and push the model to HuggingFace
4. **Task 4** — Evaluate using AlpacaEval + LLM-as-a-Judge

> Training takes about 40-60 minutes per experiment on an M2 Mac.

### Inference

If you just want to use the trained model without retraining, you can load it directly from HuggingFace:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

# load base model + DPO adapter
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base_model, "mastersubhajit/DPO")
model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained("mastersubhajit/DPO")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# generate a response
output = pipe("What is the capital of France?", max_new_tokens=256, do_sample=False)
print(output[0]["generated_text"])
```

Or if you have the adapter saved locally:

```python
model = PeftModel.from_pretrained(base_model, "./final_dpo_model")
model = model.merge_and_unload()
```

## Experiments

I ran two DPO training experiments to compare the effect of different hyperparameters:

| | Experiment 1 | Experiment 2 |
|---|---|---|
| Beta | 0.1 | 0.3 |
| Learning Rate | 5e-5 | 1e-4 |
| Steps | 50 | 50 |
| Batch Size | 1 (grad accum = 4) | 1 (grad accum = 4) |

Experiment 2 with higher beta and learning rate showed a stronger decrease in training loss, which makes sense — the higher beta pushes the model harder to distinguish between chosen and rejected responses.

## Evaluation Results

I used 15 random samples from AlpacaEval's `helpful_base` subset and had both models generate responses. Then Llama 3.3 70B (via Groq) judged which response was better.

**DPO Win Rate: ~60%**

The DPO model performed better on factual and knowledge-based questions, which aligns with what we'd expect since the training data specifically targets hallucination reduction.

## HuggingFace Model

The trained LoRA adapter is uploaded here: [mastersubhajit/DPO](https://huggingface.co/mastersubhajit/DPO)

## Key Takeaways

- DPO is a simple but effective way to align LLMs without needing a reward model
- Even with just 50 steps of LoRA training, you can see measurable improvements on factual questions
- LLM-as-a-Judge is a practical way to evaluate alignment when human evaluation isn't feasible
- Higher beta values in DPO create a stronger preference signal, but you have to be careful not to overfit
