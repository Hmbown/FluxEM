"""
Baseline runners for FluxEM + Qwen3-4B benchmarks.

Provides pluggable baseline backends that can be used without tool-calling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import time


@dataclass
class BaselineResult:
    response: str
    time_ms: float
    backend: str


class BaselineRunner:
    backend: str

    def generate(self, prompt: str) -> BaselineResult:
        raise NotImplementedError

    def close(self) -> None:
        return None


class WrapperBaseline(BaselineRunner):
    """Adapter for Qwen3MLXWrapper baseline generation."""

    def __init__(self, wrapper) -> None:
        self.wrapper = wrapper
        self.backend = "mlx"

    def generate(self, prompt: str) -> BaselineResult:
        result = self.wrapper.generate_baseline(prompt)
        return BaselineResult(
            response=result.get("response", ""),
            time_ms=result.get("time_ms", 0.0),
            backend=self.backend,
        )


class TransformersBaseline(BaselineRunner):
    """Baseline using HuggingFace transformers (local files only)."""

    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        device: str = "cpu",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        trust_remote_code: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "transformers and torch are required for transformers baseline."
            ) from exc

        self.backend = "transformers"
        self.device = torch.device(device)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path or model_path,
            local_files_only=True,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.float32,
            trust_remote_code=trust_remote_code,
        )
        self.model.to(self.device)
        self.model.eval()

    def generate(self, prompt: str) -> BaselineResult:
        import torch

        start = time.time()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
            )
        generated = output[0][inputs["input_ids"].shape[1] :]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        return BaselineResult(
            response=response,
            time_ms=(time.time() - start) * 1000,
            backend=self.backend,
        )

