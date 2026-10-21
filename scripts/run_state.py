"""Prevent accidental output reuse, silent resume skips, and broken JSONL appends."""
from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

from evaluation import read_records


def item_key(item: dict) -> str:
    fields = ("id", "system", "user", "context", "question", "ideal", "orig_ideal",
              "context_lang", "qa_lang", "injection_lang", "position", "defense",
              "placement", "is_injection", "lang")
    content = {field: item[field] for field in fields if field in item}
    raw = json.dumps(content, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def run_config(args, kind: str) -> dict:
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Input and output must be different files")
    if args.max_new_tokens <= 0 or (args.max_input_tokens is not None and args.max_input_tokens <= 0):
        raise ValueError("Token limits must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    return {"runner": kind, "run_version": 2, "model": args.model,
            "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
            "max_new_tokens": args.max_new_tokens, "max_input_tokens": args.max_input_tokens,
            "prompt_format": args.prompt_format, "load_in_4bit_requested": args.load_in_4bit,
            "lang": getattr(args, "lang", None)}


def check_resume(path: Path, config: dict, expected: set[str], resume: bool, overwrite: bool) -> set[str]:
    if resume and overwrite:
        raise ValueError("--resume and --overwrite cannot be combined")
    if not path.exists():
        return set()
    if not resume:
        if not overwrite:
            raise FileExistsError(f"{path} exists. Use --resume, a new path, or explicit --overwrite.")
        return set()
    if path.stat().st_size == 0:
        return set()
    done = set()
    for row in read_records(path, ("response",)):
        if row.get("run_config") != config:
            raise ValueError("Resume configuration differs or is absent. Keep the old output and choose a new path.")
        key = item_key(row)
        if row.get("run_key") != key or key not in expected:
            raise ValueError("Resume output does not match the current input items")
        if key in done:
            raise ValueError("Duplicate items in resume output")
        done.add(key)
    return done


def unique_keys(items: list[dict]) -> set[str]:
    keys = [item_key(item) for item in items]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate input items; assign distinct IDs if repeats are intentional")
    return set(keys)


@contextmanager
def open_output(path: Path, resume: bool, overwrite: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_newline = False
    if resume and path.exists() and path.stat().st_size:
        with path.open("rb") as source:
            source.seek(-1, 2)
            needs_newline = source.read(1) != b"\n"
    mode = "a" if resume else ("w" if overwrite else "x")
    with path.open(mode, encoding="utf-8") as output:
        if needs_newline:
            output.write("\n")
        yield output
