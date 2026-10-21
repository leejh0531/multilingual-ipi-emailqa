#!/usr/bin/env python3
"""Compute EmailQA baseline accuracy for one or more response JSONL files.

The scorer handles:
- money answers such as $200.00
- Bengali numerals in model output, e.g. $৪২০.০০
- unknown tokens: unknown / 不明 / অজানা
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from evaluation import read_records, require_text, score_answer, unknown_label, money_values

MONEY_RE = re.compile(r"\$\s*[\d\u09E6-\u09EF,\.]+")
UNKNOWN_TOKENS = {"unknown", "Unknown", "UNKNOWN", "不明", "অজানা", "わからない", "모름", "불명", "n/a", "none"}


def to_ascii_digit(ch: str) -> str:
    try:
        return str(unicodedata.digit(ch))
    except Exception:
        return ch


def normalize_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.strip()
    if "\n" in s:
        s = s.split("\n", 1)[0].strip()
    s = s.strip(" \t\r\n\"'`“”‘’()[]{}")
    s = re.sub(r"[ \t]+", " ", s)
    s = s.rstrip(" .。．！!？?、,;；:")
    return s


def normalize_unknown(s: str) -> str:
    return normalize_text(s).lower()


def extract_money_list(s: str) -> list[str]:
    values: list[str] = []
    for m in MONEY_RE.findall(s or ""):
        x = "".join(to_ascii_digit(ch) for ch in m)
        x = x.replace(" ", "")
        if not x.startswith("$"):
            x = "$" + x
        values.append(x)
    return values


def is_unknown_label(ideal: str, unknown_token: str | None) -> bool:
    i = normalize_unknown(ideal)
    tokens = {normalize_unknown(x) for x in UNKNOWN_TOKENS}
    if unknown_token:
        tokens.add(normalize_unknown(unknown_token))
    return i in tokens


def legacy_infer_unknown_token(rows: list[dict[str, Any]]) -> str:
    candidates = []
    for r in rows:
        ideal = str(r.get("ideal", r.get("orig_ideal", ""))).strip()
        if not ideal or extract_money_list(ideal):
            continue
        candidates.append(ideal)
    if not candidates:
        return "unknown"
    return Counter(candidates).most_common(1)[0][0]


def legacy_score_one(resp: str, ideal: str, unknown_token: str | None, use_contains_for_known: bool) -> bool:
    if extract_money_list(ideal):
        return extract_money_list(resp) == extract_money_list(ideal)
    if is_unknown_label(ideal, unknown_token):
        r = normalize_unknown(resp)
        tokens = {normalize_unknown(x) for x in UNKNOWN_TOKENS}
        if unknown_token:
            tokens.add(normalize_unknown(unknown_token))
        # Conservative unknown rule: exact unknown token OR no money amount.
        return r in tokens or not extract_money_list(resp)
    if use_contains_for_known:
        return normalize_text(ideal) in normalize_text(resp)
    return normalize_text(resp) == normalize_text(ideal)


def infer_unknown_token(rows: list[dict[str, Any]]) -> str:
    # Never infer an arbitrary known text answer as an unknown token.
    return "unknown"


def score_one(resp: str, ideal: str, unknown_token: str | None,
              use_contains_for_known: bool, scoring_policy: str = "strict_v2") -> bool:
    if scoring_policy == "legacy":
        return legacy_score_one(resp, ideal, unknown_token, use_contains_for_known)
    return score_answer(resp, ideal, "contains" if use_contains_for_known else "exact",
                        case_sensitive=True, unknown_token=unknown_token)


def expand_inputs(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pat in patterns:
        matches = glob.glob(pat)
        if matches:
            files.extend(Path(x) for x in matches)
        else:
            raise FileNotFoundError(f"No input files match: {pat}")
    seen = set()
    out = []
    for f in files:
        if f.exists() and f.is_file() and str(f) not in seen:
            out.append(f)
            seen.add(str(f))
    return out


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows = read_records(path, ("response",))
    for i, row in enumerate(rows):
        key = "ideal" if "ideal" in row else "orig_ideal"
        require_text(row, (key,), f"{path} row {i}")
    return rows, 0


def eval_rows(rows: list[dict[str, Any]], unknown_token: str | None, use_contains_for_known: bool, scoring_policy: str = "strict_v2") -> dict[str, Any]:
    stats = Counter()
    for r in rows:
        ideal = str(r.get("ideal", r.get("orig_ideal", "")))
        resp = str(r.get("response", ""))
        if (extract_money_list(ideal) if scoring_policy == "legacy" else money_values(ideal)):
            bucket = "known"
        elif (is_unknown_label(ideal, unknown_token) if scoring_policy == "legacy" else unknown_label(ideal, unknown_token)):
            bucket = "unknown"
        else:
            bucket = "known_other"
        ok = score_one(resp, ideal, unknown_token, use_contains_for_known, scoring_policy)
        stats["n"] += 1
        stats["correct"] += int(ok)
        stats[f"{bucket}_n"] += 1
        stats[f"{bucket}_correct"] += int(ok)

    def div(a: int, b: int) -> float:
        return a / b if b else 0.0

    return {
        "n": stats["n"],
        "correct": stats["correct"],
        "accuracy": div(stats["correct"], stats["n"]),
        "known_n": stats["known_n"] + stats["known_other_n"],
        "known_correct": stats["known_correct"] + stats["known_other_correct"],
        "known_acc": div(stats["known_correct"] + stats["known_other_correct"], stats["known_n"] + stats["known_other_n"]),
        "unknown_n": stats["unknown_n"],
        "unknown_correct": stats["unknown_correct"],
        "unknown_acc": div(stats["unknown_correct"], stats["unknown_n"]),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--unknown-token", default=None)
    p.add_argument("--scoring-policy", choices=["strict_v2", "legacy"], default="strict_v2")
    p.add_argument("--use-contains-for-known", action="store_true")
    p.add_argument("--out-csv", type=Path, default=None)
    p.add_argument("--out-json", type=Path, default=None)
    args = p.parse_args()

    files = expand_inputs(args.inputs)
    if not files:
        raise FileNotFoundError("no input files found")

    csv_rows: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for f in files:
        rows, bad = read_jsonl(f)
        token = args.unknown_token or (legacy_infer_unknown_token(rows) if args.scoring_policy == "legacy" else infer_unknown_token(rows))
        s = eval_rows(rows, token, args.use_contains_for_known, args.scoring_policy)
        s["scoring_policy"] = args.scoring_policy
        s.update({"file": str(f), "unknown_token": token, "bad_json_lines": bad})
        csv_rows.append(s)
        all_rows.extend(rows)
        print(f"[{f.name}] acc={s['accuracy']:.4f} ({s['correct']}/{s['n']}) known={s['known_acc']:.4f} unknown={s['unknown_acc']:.4f} unk_token={token} bad={bad}")

    overall_token = args.unknown_token or (legacy_infer_unknown_token(all_rows) if args.scoring_policy == "legacy" else infer_unknown_token(all_rows))
    overall = eval_rows(all_rows, overall_token, args.use_contains_for_known, args.scoring_policy)
    overall["scoring_policy"] = args.scoring_policy
    overall.update({"file": "__OVERALL__", "unknown_token": overall_token, "bad_json_lines": 0})
    csv_rows.append(overall)
    print(f"[OVERALL] acc={overall['accuracy']:.4f} ({overall['correct']}/{overall['n']}) known={overall['known_acc']:.4f} unknown={overall['unknown_acc']:.4f}")

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", encoding="utf-8", newline="") as f:
            fields = ["scoring_policy", "file", "n", "correct", "accuracy", "known_n", "known_correct", "known_acc", "unknown_n", "unknown_correct", "unknown_acc", "unknown_token", "bad_json_lines"]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(csv_rows)
        print(f"saved csv={args.out_csv}")
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(csv_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved json={args.out_json}")


if __name__ == "__main__":
    main()
