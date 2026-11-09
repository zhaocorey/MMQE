"""Local student-model inference for evaluation.

Two backends:
- "vllm" (preferred for throughput): requires the vllm package.
- "hf"   (transformers, batched greedy generation): always available.

Both expose StudentPredictor.predict(texts) -> list of raw output strings.
Parsing into scores uses lib.json_extractor (same path as teacher scoring,
so metric computation is backend-independent).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class StudentPredictor:
    def __init__(self,
                 model_path: str,
                 rubric_prompt: str,
                 backend: str = "hf",
                 max_input_len: int = 4096,
                 max_output_len: int = 256,
                 batch_size: int = 16,
                 gpu_memory_utilization: float = 0.9,
                 temperature: float = 0.1,
                 dtype: str = "bfloat16"):
        self.model_path = model_path
        self.rubric = rubric_prompt
        self.backend = backend
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.batch_size = batch_size
        self.temperature = temperature
        self._impl = None
        self._gpu_util = gpu_memory_utilization
        self._dtype = dtype
        self.last_stats: Dict[str, Any] = {}

    # -- lazy backend construction ----------------------------------------- #
    def _load(self):
        if self._impl is not None:
            return self._impl
        if self.backend == "vllm":
            from vllm import LLM
            self._impl = LLM(model=self.model_path,
                             max_model_len=self.max_input_len + self.max_output_len,
                             gpu_memory_utilization=self._gpu_util,
                             dtype=self._dtype,
                             trust_remote_code=True)
        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(self.model_path,
                                                trust_remote_code=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=getattr(torch, self._dtype),
                trust_remote_code=True,
                device_map="auto",
            )
            model.eval()
            self._impl = {"model": model, "tokenizer": tok, "torch": torch}
        return self._impl

    def _messages(self, text: str) -> List[Dict[str, str]]:
        return [{"role": "system", "content": self.rubric},
                {"role": "user", "content": text[:self.max_input_len * 3]}]

    # -- prediction --------------------------------------------------------- #
    def predict(self, texts: List[str]) -> List[Optional[str]]:
        t0 = time.monotonic()
        if self.backend == "vllm":
            outs = self._predict_vllm(texts)
        else:
            outs = self._predict_hf(texts)
        elapsed = time.monotonic() - t0
        n_out_tokens = sum(len(o or "") for o in outs) // 4
        self.last_stats = {
            "n_samples": len(texts),
            "elapsed_s": round(elapsed, 1),
            "approx_output_ktok_per_s": round(n_out_tokens / elapsed / 1000, 2)
            if elapsed > 0 else 0.0,
        }
        return outs

    def _predict_vllm(self, texts: List[str]) -> List[Optional[str]]:
        from vllm import SamplingParams
        impl = self._load()
        tok = impl.get_tokenizer()
        prompts = [tok.apply_chat_template(self._messages(t),
                                           tokenize=False,
                                           add_generation_prompt=True)
                   for t in texts]
        # Paper Sec 5.2: temperature 0.1 on vLLM inference
        params = SamplingParams(temperature=self.temperature,
                                max_tokens=self.max_output_len)
        outputs = impl.generate(prompts, params)
        by_prompt = {}
        for o in outputs:
            by_prompt[o.prompt] = o.outputs[0].text
        return [by_prompt.get(p) for p in prompts]

    def _predict_hf(self, texts: List[str]) -> List[Optional[str]]:
        impl = self._load()
        model, tok, torch = impl["model"], impl["tokenizer"], impl["torch"]
        results: List[Optional[str]] = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            prompts = [tok.apply_chat_template(self._messages(t),
                                               tokenize=False,
                                               add_generation_prompt=True)
                       for t in batch_texts]
            enc = tok(prompts, return_tensors="pt", padding=True,
                      truncation=True, max_length=self.max_input_len)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            with torch.no_grad():
                out = model.generate(
                    **enc, max_new_tokens=self.max_output_len,
                    do_sample=False, pad_token_id=tok.pad_token_id)
            input_len = enc["input_ids"].shape[1]
            gen = out[:, input_len:]
            results.extend(tok.batch_decode(gen, skip_special_tokens=True))
            if (i // self.batch_size) % 20 == 0:
                logger.info("hf predict %d/%d", i + len(batch_texts), len(texts))
        return results
