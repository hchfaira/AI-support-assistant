"""Merge a trained LoRA adapter into the base model weights, producing a
single standalone model ready for ONNX export / Triton deployment.

Usage:
    python -m src.llm.merge_adapter \
        --base-model Qwen/Qwen2.5-3B-Instruct \
        --adapter checkpoints/qlora-support-assistant \
        --output checkpoints/merged-support-assistant
"""
from __future__ import annotations

import argparse


def merge_adapter(base_model: str, adapter_path: str, output_dir: str) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    merged = PeftModel.from_pretrained(base, adapter_path)
    merged = merged.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    merged.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Merged model saved to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    merge_adapter(args.base_model, args.adapter, args.output)


if __name__ == "__main__":
    main()
