#!/usr/bin/env python3
"""Run chat-style LLM inference for prepared multilingual IPI EmailQA prompts.

This runner is intended for the paper/thesis setting, where instruction-tuned
models are evaluated under defense_off / defense_on conditions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluation import read_records
from run_state import item_key as content_key, run_config, unique_keys, check_resume, open_output


REQUIRED_FIELDS = ("system", "user", "orig_ideal")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return read_records(path, REQUIRED_FIELDS)


def make_key(item: dict[str, Any]) -> str:
    return content_key(item)


def format_chat_prompt(tokenizer, system: str, user: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    raise ValueError("The selected tokenizer has no chat template; supply an instruction model with a template.")


def load_model(model_name: str, load_in_4bit: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[run] device={device}")
    print(f"[run] model={model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if load_in_4bit and device != "cuda":
        raise ValueError("--load-in-4bit requires CUDA in this runner; omit it for CPU execution")
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
            raise RuntimeError("4-bit loading failed; fix the environment or start a separate run without --load-in-4bit") from e

    if model is None:
        if device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
            model.to(device)
    model.eval()
    return tokenizer, model, device


def generate_one(
    tokenizer,
    model,
    device: str,
    system: str,
    user: str,
    max_new_tokens: int,
    max_input_tokens: int | None,
) -> str:
    import torch

    prompt = format_chat_prompt(tokenizer, system, user)
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
            eos_token_id=(model.generation_config.eos_token_id
                          if model.generation_config.eos_token_id is not None else tokenizer.eos_token_id),
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
    p.add_argument("--prompt-format", choices=["chat"], default="chat", help="Paper setting uses chat formatting for instruction-tuned models.")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true", help="Explicitly replace an existing output instead of resuming")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    config = run_config(args, "prepared")

    prompts = read_jsonl(args.input)
    if args.limit is not None:
        prompts = prompts[: args.limit]
    print(f"[run] prompts={len(prompts)} from {args.input}")
    print("[run] prompt_format=chat")

    expected = unique_keys(prompts)
    done = check_resume(args.output, config, expected, args.resume, args.overwrite)
    if done == expected:
        print(f"All {len(done)} items are already complete: {args.output}")
        return
    tokenizer, model, device = load_model(args.model, args.load_in_4bit)
    from tqdm import tqdm

    with open_output(args.output, args.resume, args.overwrite) as out:
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
                args.max_new_tokens,
                args.max_input_tokens,
            )
            obj = {**item, "response": response, "model": args.model, "prompt_format": "chat", "run_config": config, "run_key": key,
                   "loaded_in_4bit": bool(getattr(model, "is_loaded_in_4bit", False))}
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            out.flush()
            done.add(key)

    print(f"[run] saved={args.output}")


if __name__ == "__main__":
    main()
