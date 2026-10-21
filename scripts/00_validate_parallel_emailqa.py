#!/usr/bin/env python3
"""Validate parallel multilingual EmailQA JSONL files.

This script is intentionally conservative. It checks JSONL validity, row counts,
required fields, known/unknown distribution, money-answer alignment, and rough
language hints for context/question fields.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from evaluation import read_records, validate_parallel, money_values, unknown_label

MONEY_RE = re.compile(r"\$\s*[\d\u09E6-\u09EF,\.]+")
UNKNOWN_TOKENS = {"unknown", "不明", "অজানা", "unk", "n/a", "none", "모름", "불명"}


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    return read_records(path, ("context", "question", "ideal")), 0


def ascii_digit(ch: str) -> str:
    try:
        return str(unicodedata.digit(ch))
    except Exception:
        return ch


def normalize_money(s: str) -> list:
    return money_values(s)


def norm_text(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def is_unknown(ideal: str) -> bool:
    return unknown_label(ideal)


def ideal_kind(ideal: str) -> str:
    if normalize_money(ideal):
        return "money"
    if is_unknown(ideal):
        return "unknown"
    if norm_text(ideal):
        return "other_known"
    return "empty"


def char_counts(text: str) -> Counter[str]:
    c = Counter()
    for ch in text or "":
        o = ord(ch)
        if 0x3040 <= o <= 0x30FF or 0x4E00 <= o <= 0x9FFF:
            c["ja"] += 1
        elif 0x0980 <= o <= 0x09FF:
            c["bn"] += 1
        elif "a" <= ch.lower() <= "z":
            c["en"] += 1
    return c


def guess_lang(text: str) -> str:
    c = char_counts(text)
    if not c:
        return "unk"
    return c.most_common(1)[0][0]


def summarize_lang(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    required_missing = Counter()
    kinds = Counter()
    guesses_context = Counter()
    guesses_question = Counter()
    for r in rows:
        for k in ["context", "question", "ideal"]:
            if k not in r:
                required_missing[k] += 1
        kinds[ideal_kind(str(r.get("ideal", "")))] += 1
        guesses_context[guess_lang(str(r.get("context", "")))] += 1
        guesses_question[guess_lang(str(r.get("question", "")))] += 1
    return {
        "lang": name,
        "n": len(rows),
        "missing": dict(required_missing),
        "ideal_kinds": dict(kinds),
        "context_lang_guess": dict(guesses_context),
        "question_lang_guess": dict(guesses_question),
    }


def compare_alignment(base_lang: str, base_rows: list[dict[str, Any]], other_lang: str, other_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    n = min(len(base_rows), len(other_rows))
    for i in range(n):
        b = base_rows[i]
        o = other_rows[i]
        b_ideal = str(b.get("ideal", ""))
        o_ideal = str(o.get("ideal", ""))
        b_kind = ideal_kind(b_ideal)
        o_kind = ideal_kind(o_ideal)
        b_money = normalize_money(b_ideal)
        o_money = normalize_money(o_ideal)

        bad = False
        reason = ""
        if b_kind == "money" or o_kind == "money":
            if b_money != o_money:
                bad = True
                reason = f"money mismatch: {b_money} vs {o_money}"
        elif b_kind != o_kind:
            bad = True
            reason = f"kind mismatch: {b_kind} vs {o_kind}"

        if bad:
            mismatches.append({
                "row_index": i,
                "base_lang": base_lang,
                "other_lang": other_lang,
                "base_ideal": b_ideal,
                "other_ideal": o_ideal,
                "reason": reason,
                "base_question_preview": str(b.get("question", ""))[:80],
                "other_question_preview": str(o.get("question", ""))[:80],
            })
    return mismatches


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--en", type=Path, required=True)
    p.add_argument("--ja", type=Path, required=True)
    p.add_argument("--bn", type=Path, required=True)
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--max-mismatches", type=int, default=20)
    args = p.parse_args()
    if args.max_mismatches < 0:
        p.error("--max-mismatches must be nonnegative")

    paths = {"en": args.en, "ja": args.ja, "bn": args.bn}
    data: dict[str, list[dict[str, Any]]] = {}
    report: dict[str, Any] = {"files": {}, "alignment": {}}

    for lang, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        rows, bad = read_jsonl(path)
        data[lang] = rows
        report["files"][lang] = summarize_lang(lang, rows)
        report["files"][lang]["path"] = str(path)
        report["files"][lang]["bad_json_lines"] = bad

    print("=== File summary ===")
    for lang in ["en", "ja", "bn"]:
        s = report["files"][lang]
        print(f"[{lang}] n={s['n']} bad_json={s['bad_json_lines']} ideals={s['ideal_kinds']}")
        print(f"     context_guess={s['context_lang_guess']} question_guess={s['question_lang_guess']}")
        if s["missing"]:
            print(f"     missing={s['missing']}")

    lens = {lang: len(rows) for lang, rows in data.items()}
    if len(set(lens.values())) != 1:
        print(f"\n[WARN] length mismatch: {lens}")
    else:
        print(f"\n[OK] all files have {next(iter(lens.values()))} rows")

    print("\n=== Alignment check against en ===")
    for lang in ["ja", "bn"]:
        mm = compare_alignment("en", data["en"], lang, data[lang])
        report["alignment"][f"en_vs_{lang}"] = {"mismatch_count": len(mm), "examples": mm[: args.max_mismatches]}
        if mm:
            print(f"[WARN] en vs {lang}: {len(mm)} potential mismatches")
            for ex in mm[: args.max_mismatches]:
                print(f"  row={ex['row_index']} {ex['reason']} | en={ex['base_ideal']} {lang}={ex['other_ideal']}")
        else:
            print(f"[OK] en vs {lang}: no money/kind mismatch")

    validation_error = None
    try:
        validate_parallel(data)
    except ValueError as exc:
        validation_error = str(exc)
    report["valid"] = validation_error is None
    report["validation_error"] = validation_error

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved report: {args.out_json}")
    if validation_error:
        p.exit(1, f"Validation failed: {validation_error}\n")


if __name__ == "__main__":
    main()
