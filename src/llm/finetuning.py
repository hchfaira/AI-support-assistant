"""QLoRA fine-tuning of an instruction-tuned LLM for customer support response
generation, using Hugging Face Transformers + PEFT + TRL + BitsAndBytes.

Intended to run on a single GPU (e.g. a T4/A10/A100 cloud instance). Produces
a LoRA adapter under `output_dir`; merge it into the base weights with
`src.llm.merge_adapter` before exporting to ONNX/Triton.

Usage:
    python -m src.llm.finetuning \
        --base-model Qwen/Qwen2.5-3B-Instruct \
        --data data/sample/sft_dataset.json \
        --output checkpoints/qlora-support-assistant
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_sft_dataset(path: str | Path) -> list[dict]:
    """Load instruction/context/query/response records and format them as a
    single training prompt-completion pair per example.
    """
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    formatted = []
    for r in records:
        prompt = (
            f"{r['instruction']}\n\nContext:\n{r['context']}\n\n"
            f"Customer question: {r['query']}\n\nJSON response:"
        )
        formatted.append({"prompt": prompt, "completion": r["response"]})
    return formatted


def train(
    base_model: str = "Qwen/Qwen2.5-3B-Instruct",
    data_path: str = "data/sample/sft_dataset.json",
    output_dir: str = "checkpoints/qlora-support-assistant",
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    use_4bit: bool = True,
) -> None:
    """Fine-tune `base_model` with QLoRA on (prompt, completion) pairs.

    Requires: transformers, peft, trl, bitsandbytes, accelerate, torch (CUDA).
    This is the full production recipe; it needs a GPU to actually execute.
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    examples = load_sft_dataset(data_path)
    dataset = Dataset.from_list(
        [{"text": ex["prompt"] + ex["completion"]} for ex in examples]
    )

    bnb_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        if use_4bit
        else None
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb_config, device_map="auto"
    )

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        max_seq_length=1024,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=lora_config,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)
    print(f"Saved LoRA adapter to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--data", default="data/sample/sft_dataset.json")
    parser.add_argument("--output", default="checkpoints/qlora-support-assistant")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    train(
        base_model=args.base_model,
        data_path=args.data,
        output_dir=args.output,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_4bit=not args.no_4bit,
    )


if __name__ == "__main__":
    main()
