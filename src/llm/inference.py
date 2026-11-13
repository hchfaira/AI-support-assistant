"""Inference wrapper for the base and fine-tuned LLM, used by the RAG pipeline
and by the base-vs-fine-tuned comparison notebook.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GenerationResult:
    text: str
    latency_ms: float


class FineTunedLLM:
    """Thin wrapper around a Transformers causal LM (optionally with a LoRA
    adapter attached) for greedy/structured JSON generation.

    Falls back to `None` backend (caller should use a heuristic/template
    response, see `src.rag.generation.heuristic_generate`) when `transformers`
    is not installed or no GPU/model weights are available, so the rest of the
    pipeline keeps working in constrained environments.
    """

    def __init__(self, model_name_or_path: str, adapter_path: str | None = None):
        self.model_name_or_path = model_name_or_path
        self.adapter_path = adapter_path
        self._model = None
        self._tokenizer = None
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto"
            )
            if adapter_path:
                from peft import PeftModel

                self._model = PeftModel.from_pretrained(self._model, adapter_path)
        except Exception:
            self._model = None
            self._tokenizer = None

    @property
    def is_real_backend(self) -> bool:
        return self._model is not None

    def generate(self, prompt: str, max_new_tokens: int = 256) -> GenerationResult:
        import time

        start = time.perf_counter()
        if self._model is None or self._tokenizer is None:
            raise RuntimeError(
                "No LLM backend available. Install transformers+torch and "
                "provide a valid model_name_or_path, or use "
                "src.rag.generation.heuristic_generate as a fallback."
            )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        output_ids = self._model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
        text = self._tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        latency_ms = (time.perf_counter() - start) * 1000
        return GenerationResult(text=text, latency_ms=latency_ms)
