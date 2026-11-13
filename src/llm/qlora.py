"""QLoRA configuration helpers shared by `finetuning.py` and inference code.

Centralizes the BitsAndBytes 4-bit quantization config and LoRA hyperparameters
so experiments (rank, alpha, target modules, compute dtype) stay consistent
and easy to sweep.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class QLoRAConfig:
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    def to_bitsandbytes_config(self):
        import torch
        from transformers import BitsAndBytesConfig

        dtype = getattr(torch, self.bnb_4bit_compute_dtype)
        return BitsAndBytesConfig(
            load_in_4bit=self.load_in_4bit,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
        )

    def to_lora_config(self):
        from peft import LoraConfig

        return LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=self.target_modules,
        )


# Pre-defined sweeps useful for an ablation table in the comparison notebook.
LORA_RANK_SWEEP = [8, 16, 32, 64]
