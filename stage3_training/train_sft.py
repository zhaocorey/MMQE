"""Stage 3 — Full-parameter SFT of the student model on the MTDD dataset.

Paper setting: full SFT of all parameters — the paper contrasts this with a
frozen embedder + MLP heads approach, and MMQE gains +13pp Spearman rho over
its base model (paper abstract, Sec 5.1/5.3.1). 3 epochs,
max_input_len=4096, max_output_len=256, cosine LR with warmup, bf16.

Only the assistant response tokens contribute to the loss (prompt tokens
are masked with -100), so the student learns to emit the compact JSON.

Launch (single GPU):
    python -m stage3_training.train_sft --config config/run_config.json \
        --train-file <data_root>/sft/train.jsonl \
        --eval-file <data_root>/sft/eval.jsonl

Launch (multi-GPU, recommended for 1.69M samples):
    torchrun --nproc_per_node=8 -m stage3_training.train_sft \
        --config config/run_config.json \
        --train-file <data_root>/sft/train.jsonl

Requires: torch, transformers>=4.43, accelerate, datasets.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.config_loader import apply_overrides, load_config

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("stage3.train_sft")

IGNORE_INDEX = -100


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

class SFTDataset:
    """Chat-format JSONL -> tokenized (input_ids, attention_mask, labels).

    Tokenization strategy: apply the tokenizer's chat template to the full
    conversation and to the prompt-only prefix; labels are IGNORE_INDEX for
    all prefix tokens and the true ids for the assistant segment.
    """

    def __init__(self, path: str | Path, tokenizer,
                 max_input_len: int = 4096, max_output_len: int = 256):
        import torch  # local import so module loads without torch for tests
        self.torch = torch
        self.tokenizer = tokenizer
        self.max_len = max_input_len + max_output_len
        self.examples: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))
        logger.info("loaded %d training examples from %s",
                    len(self.examples), path)

    def __len__(self) -> int:
        return len(self.examples)

    def _encode(self, messages: List[Dict[str, str]]):
        tok = self.tokenizer
        full_ids = tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False)
        prompt_ids = tok.apply_chat_template(
            messages[:-1], tokenize=True, add_generation_prompt=True)
        # Guard against template edge cases where prefix >= full
        if len(prompt_ids) >= len(full_ids):
            prompt_ids = full_ids[:max(0, len(full_ids) - 1)]
        input_ids = full_ids[:self.max_len]
        labels = ([IGNORE_INDEX] * len(prompt_ids)) + input_ids[len(prompt_ids):]
        labels = labels[:self.max_len]
        return input_ids, labels

    def __getitem__(self, idx: int):
        torch = self.torch
        ex = self.examples[idx]
        input_ids, labels = self._encode(ex["messages"])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


@dataclass
class Collator:
    pad_token_id: int

    def __call__(self, batch: List[Dict[str, Any]]):
        import torch
        max_len = max(b["input_ids"].size(0) for b in batch)

        def pad(t, value):
            out = torch.full((max_len,), value, dtype=t.dtype)
            out[:t.size(0)] = t
            return out

        return {
            "input_ids": torch.stack(
                [pad(b["input_ids"], self.pad_token_id) for b in batch]),
            "attention_mask": torch.stack(
                [pad(b["attention_mask"], 0) for b in batch]),
            "labels": torch.stack(
                [pad(b["labels"], IGNORE_INDEX) for b in batch]),
        }


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def train(cfg: Dict[str, Any], args) -> None:
    import torch
    from torch.utils.data import DataLoader
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              get_cosine_schedule_with_warmup)

    student = cfg.get("student", {})
    model_name = args.base_model or student.get("base_model")
    out_dir = Path(args.output_dir or student.get("output_dir") or
                   Path(cfg.get("data_root", ".")) / "student_model")
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = int(args.epochs or student.get("epochs", 3))
    lr = float(args.learning_rate or student.get("learning_rate", 1e-5))
    bs = int(student.get("per_device_train_batch_size", 2))
    grad_accum = int(student.get("gradient_accumulation_steps", 8))
    warmup_ratio = float(student.get("warmup_ratio", 0.03))
    weight_decay = float(student.get("weight_decay", 0.01))
    max_input = int(student.get("max_input_len", 4096))
    max_output = int(student.get("max_output_len", 256))
    use_bf16 = bool(student.get("bf16", True))
    grad_ckpt = bool(student.get("gradient_checkpointing", True))
    seed = int(student.get("seed", 42))
    log_every = int(args.log_every)
    save_every = int(args.save_every_steps)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{local_rank}"
                          if torch.cuda.is_available() else "cpu")
    if world_size > 1:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    torch.manual_seed(seed)

    logger.info("loading tokenizer/model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        trust_remote_code=True,
    )
    if grad_ckpt:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.to(device)

    train_ds = SFTDataset(args.train_file, tokenizer, max_input, max_output)
    collator = Collator(pad_token_id=tokenizer.pad_token_id)
    sampler = None
    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            train_ds, num_replicas=world_size, rank=local_rank, shuffle=True,
            seed=seed)
    loader = DataLoader(train_ds, batch_size=bs, shuffle=(sampler is None),
                        sampler=sampler, collate_fn=collator,
                        num_workers=int(args.dataloader_workers),
                        pin_memory=True)

    total_steps = math.ceil(len(loader) / grad_accum) * epochs
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    logger.info("total optimizer steps=%d warmup=%d world_size=%d",
                total_steps, warmup_steps, world_size)

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}], lr=lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps,
                                                total_steps)

    scaler = None  # bf16 needs no loss scaler; fp16 users can add GradScaler
    global_step = 0
    running_loss = 0.0
    model.train()
    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for step, batch in enumerate(loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / grad_accum
            loss.backward()
            running_loss += outputs.loss.item()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % log_every == 0 and local_rank == 0:
                    avg = running_loss / (log_every * grad_accum)
                    logger.info("epoch %d step %d/%d loss=%.4f lr=%.2e",
                                epoch + 1, global_step, total_steps, avg,
                                scheduler.get_last_lr()[0])
                    running_loss = 0.0
                if save_every and global_step % save_every == 0 and local_rank == 0:
                    ckpt = out_dir / f"checkpoint-{global_step}"
                    model.save_pretrained(ckpt)
                    tokenizer.save_pretrained(ckpt)
                    logger.info("saved %s", ckpt)

    if local_rank == 0:
        logger.info("training complete — saving final model to %s", out_dir)
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
    if world_size > 1:
        torch.distributed.destroy_process_group()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/run_config.json")
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--eval-file", default=None,
                    help="(optional) held-out split; perplexity eval only")
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--learning-rate", type=float, default=None)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every-steps", type=int, default=2000)
    ap.add_argument("--dataloader-workers", type=int, default=2)
    ap.add_argument("-o", "--override", action="append", default=[])
    args = ap.parse_args(argv)

    cfg = apply_overrides(load_config(args.config), args.override)
    train(cfg, args)


if __name__ == "__main__":
    main()
