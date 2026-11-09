"""Async OpenAI-compatible chat client for teacher/student API inference.

Features (per MTDD skill best practices):
- asyncio + aiohttp with a concurrency semaphore (default 32 workers)
- exponential-backoff retries (1-2-4s, 3 retries)
- low temperature (0.1) for deterministic outputs
- thinking/reasoning mode disabled via provider-specific payload fields
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional

try:
    import aiohttp
except ImportError:  # pragma: no cover - allow import without aiohttp for tests
    aiohttp = None

logger = logging.getLogger(__name__)


class AsyncLLMClient:
    """Minimal async chat-completions client for OpenAI-compatible APIs."""

    def __init__(self,
                 teacher_cfg: Dict[str, Any],
                 inference_cfg: Optional[Dict[str, Any]] = None):
        self.name: str = teacher_cfg["name"]
        self.model_id: str = teacher_cfg["model_id"]
        self.api_url: str = teacher_cfg["api_url"]
        api_key_env = teacher_cfg.get("api_key_env", "")
        self.api_key: str = os.environ.get(api_key_env, "") if api_key_env else ""
        self.max_workers: int = int(teacher_cfg.get("concurrent_workers", 32))
        self.max_retries: int = int(teacher_cfg.get("max_retries", 3))
        self.timeout: float = float(teacher_cfg.get("request_timeout", 60))
        self.disable_thinking: bool = bool(teacher_cfg.get("disable_thinking", True))
        self.extra_body: Dict[str, Any] = dict(teacher_cfg.get("extra_body", {}))

        inf = inference_cfg or {}
        self.temperature: float = float(inf.get("temperature", 0.1))
        self.top_p: float = float(inf.get("top_p", 0.95))
        self.max_output_tokens: int = int(inf.get("max_output_tokens", 256))
        self.retry_base_delay: float = float(inf.get("retry_base_delay_s", 1.0))

        self._semaphore = asyncio.Semaphore(self.max_workers)
        self._session: Optional["aiohttp.ClientSession"] = None

        # Cost/throughput accounting
        self.stats = {"requests": 0, "successes": 0, "failures": 0,
                      "retries": 0, "prompt_chars": 0, "completion_chars": 0,
                      "wall_time_s": 0.0}

    async def __aenter__(self) -> "AsyncLLMClient":
        if aiohttp is None:
            raise RuntimeError("aiohttp is required for AsyncLLMClient; "
                               "pip install aiohttp")
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self

    async def __aexit__(self, *exc) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    def _build_payload(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_output_tokens,
            "stream": False,
        }
        payload.update(self.extra_body)
        # Common provider flags to disable reasoning/thinking mode (10x faster).
        if self.disable_thinking:
            payload.setdefault("enable_thinking", False)
            payload.setdefault("reasoning_effort", "none")
        return payload

    async def chat(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """One chat completion with retries. Returns text content or None."""
        assert self._session is not None, "Use client inside `async with`"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = self._build_payload(messages)

        last_err = ""
        for attempt in range(self.max_retries + 1):
            async with self._semaphore:
                try:
                    t0 = time.monotonic()
                    async with self._session.post(
                            self.api_url, json=payload, headers=headers) as resp:
                        elapsed = time.monotonic() - t0
                        self.stats["wall_time_s"] += elapsed
                        if resp.status == 200:
                            data = await resp.json()
                            content = (data.get("choices") or [{}])[0] \
                                .get("message", {}).get("content", "")
                            usage = data.get("usage", {}) or {}
                            self.stats["requests"] += 1
                            self.stats["successes"] += 1
                            self.stats["prompt_chars"] += \
                                int(usage.get("prompt_tokens", 0)) * 4
                            self.stats["completion_chars"] += \
                                int(usage.get("completion_tokens", 0)) * 4
                            return content
                        # 429 / 5xx are retryable; 4xx (except 429) are not
                        body = await resp.text()
                        last_err = f"HTTP {resp.status}: {body[:300]}"
                        if resp.status not in (408, 425, 429) and resp.status < 500:
                            self.stats["requests"] += 1
                            self.stats["failures"] += 1
                            logger.warning("[%s] non-retryable %s", self.name, last_err)
                            return None
                except (asyncio.TimeoutError, aiohttp.ClientError, OSError) as e:
                    last_err = f"{type(e).__name__}: {e}"

            if attempt < self.max_retries:
                delay = self.retry_base_delay * (2 ** attempt)  # 1-2-4s
                delay += random.uniform(0, delay * 0.25)        # jitter
                self.stats["retries"] += 1
                logger.debug("[%s] retry %d in %.1fs (%s)",
                             self.name, attempt + 1, delay, last_err)
                await asyncio.sleep(delay)

        self.stats["requests"] += 1
        self.stats["failures"] += 1
        logger.warning("[%s] giving up after %d retries: %s",
                       self.name, self.max_retries, last_err)
        return None


def build_messages(rubric_prompt: str, text: str,
                   max_input_chars: int = 16000) -> List[Dict[str, str]]:
    """Compose system=rubric / user=document messages for one sample."""
    if len(text) > max_input_chars:
        text = text[:max_input_chars]
    return [
        {"role": "system", "content": rubric_prompt},
        {"role": "user", "content": text},
    ]
