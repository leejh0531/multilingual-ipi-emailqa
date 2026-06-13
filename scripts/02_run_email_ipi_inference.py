#!/usr/bin/env python3
"""Run LLM inference for prepared multilingual IPI EmailQA prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def make_key(item: dict[str, Any]) -> str:
    if item.get("id") is not None:
        return f"id:{item['id']}"
    raw = str(item.get("system", "")) + "\n" + str(item.get("user", ""))
    return "sha1:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()


def done_keys(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(make_key(json.loads(line)))
            except json.JSONDecodeError:
                continue
    return done


def decide_prompt_format(model_name: str, requested: str) -> str:
    if requested != "auto":
        return requested
    return "chat" if "instruct" in model_name.lower() or "chat" in model_name.lower() else "plain"


def format_plain(system: str, user: str) -> str:
    return f"### System\n{system}\n\n### User\n{user}\n\n### Assistant\n"


def format_chat(tokenizer, system: str, user: str) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
        return format_plain(system, user)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_model(model_name: str, load_in_4bit: bool):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[run] device={device}")
    print(f"[run] model={model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = None
    if load_in_4bit and device == "cuda":
        try:
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
            )
            print("[run] loaded in 4-bit")
        except Exception as e:
            print(f"[run] 4-bit loading failed: {e}")
            print("[run] falling back")

    if model is None:
        if device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
            model.to(device)
    model.eval()
    return tokenizer, model, device


def generate_one(tokenizer, model, device: str, system: str, user: str, prompt_format: str, max_new_tokens: int, max_input_tokens: int | None) -> str:
    prompt = format_chat(tokenizer, system, user) if prompt_format == "chat" else format_plain(system, user)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=max_input_tokens is not None,
        max_length=max_input_tokens,
        padding=False,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen = out[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--max-input-tokens", type=int, default=None)
    p.add_argument("--prompt-format", choices=["auto", "plain", "chat"], default="auto")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    prompts = read_jsonl(args.input)
    if args.limit is not None:
        prompts = prompts[: args.limit]
    print(f"[run] prompts={len(prompts)} from {args.input}")

    tokenizer, model, device = load_model(args.model, args.load_in_4bit)
    fmt = decide_prompt_format(args.model, args.prompt_format)
    print(f"[run] prompt_format={fmt}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = done_keys(args.output) if args.resume else set()
    print(f"[run] resume={args.resume} done={len(done)}")

    mode = "a" if args.resume else "w"
    with args.output.open(mode, encoding="utf-8") as out:
        for item in tqdm(prompts, desc="generate"):
            key = make_key(item)
            if key in done:
                continue
            response = generate_one(
                tokenizer,
                model,
                device,
                str(item.get("system", "")),
                str(item.get("user", "")),
                fmt,
                args.max_new_tokens,
                args.max_input_tokens,
            )
            obj = {**item, "response": response, "model": args.model, "prompt_format": fmt}
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            out.flush()

    print(f"[run] saved={args.output}")


if __name__ == "__main__":
    main()
