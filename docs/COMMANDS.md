# Commands for new runs

Use Python 3.10+. Run from the repository root after installing requirements.txt and preparing the parallel input files described in [data/README.md](../data/README.md). These examples use the corrected v2 defaults and new filenames. They do not reproduce or overwrite historical tables.

## 0. Validate parallel data

```bash
python scripts/00_validate_parallel_emailqa.py \
  --en data/raw/email_qa_en.jsonl \
  --ja data/raw/email_qa_ja.jsonl \
  --bn data/raw/email_qa_bn.jsonl \
  --out-json results/metrics/parallel_validation_v2.json
```

Stop and fix any validation error before continuing. Passing validation does not establish translation quality.

## 1. Prepare an attack and matched clean control

Both files use exactly the same known-only selection, language combinations, system prompt and reminder condition. This example uses reminder off. All attacks are inside the email boundaries.

```bash
python scripts/01_prepare_email_ipi_multilingual.py \
  --en data/raw/email_qa_en.jsonl \
  --ja data/raw/email_qa_ja.jsonl \
  --bn data/raw/email_qa_bn.jsonl \
  --known-only --placement inside \
  --output data/prompts/email_v2_attack_off.jsonl

python scripts/01_prepare_email_ipi_multilingual.py \
  --en data/raw/email_qa_en.jsonl \
  --ja data/raw/email_qa_ja.jsonl \
  --bn data/raw/email_qa_bn.jsonl \
  --known-only --placement inside --no-injection \
  --output data/prompts/email_v2_clean_off.jsonl
```

For reminder on, add `--defense` to **both** commands and use distinct `_on` filenames throughout. If sampling, pass the same `--num-emails` and `--seed` to both. The count depends on the local files: 41 known source items yield 3,321 conditions per file. Clean conditions repeat for pairing across nominal attack-language/position factors; they are not independent observations.

## 2. Run both files through the same runner

The loop below uses Qwen with the same token limit for both conditions. For Llama, use `meta-llama/Llama-3.2-3B-Instruct` and distinct `llama32` output filenames. Model access and sufficient memory are required.

```bash
for condition in attack clean; do
  python scripts/02_run_email_ipi_inference.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --input "data/prompts/email_v2_${condition}_off.jsonl" \
    --output "results/responses/qwen25_v2_${condition}_off.jsonl" \
    --prompt-format chat --max-new-tokens 128 --resume
done
```

Use identical `--max-input-tokens` and quantization settings if supplied. Truncation can remove email content or the attack; inspect prompts and token lengths before interpreting such a run. `--load-in-4bit` requires CUDA and compatible bitsandbytes; failure stops the run rather than changing precision.

`--resume` accepts only outputs created with matching input hashes, model names, generation settings and item content. It rejects historical outputs lacking the new metadata. Choose a new output path for changed experiments. Use `--overwrite` only to intentionally replace an output; it cannot be combined with `--resume`. Remote model revisions and software versions must still be recorded separately.

## 3. Score ASR and paired QA utility

```bash
python scripts/04_compute_asr_ipi_multilingual.py \
  --input results/responses/qwen25_v2_attack_off.jsonl \
  --out-csv results/metrics/qwen25_v2_attack_off_asr.csv

for condition in attack clean; do
  python scripts/06_compute_accuracy_by_combo.py \
    --input "results/responses/qwen25_v2_${condition}_off.jsonl" \
    --gold-key orig_ideal --scoring-policy strict_v2 \
    --strip-marker --marker-mode remove \
    --out-csv "results/metrics/qwen25_v2_${condition}_off_accuracy.csv"
done

python scripts/07_compute_metrics_with_mixed.py \
  --input results/responses/qwen25_v2_attack_off.jsonl \
  --scoring-policy strict_v2 --strip-marker --marker-mode remove \
  --out-csv results/metrics/qwen25_v2_attack_off_mixed.csv

python scripts/08_summarize_hypotheses.py \
  --input results/responses/qwen25_v2_attack_off.jsonl \
  --model-tag qwen25_v2_attack_off \
  --out-csv results/metrics/qwen25_v2_attack_off_hypotheses.csv
```

Compare the QA tables by the same source items and conditions. Matching the workflow controls the documented prompt and generation differences; it does not supply statistical significance. Hypothesis comparisons with an empty group are unavailable, not zero. ASR measures marker occurrence, including mixed responses.

## 4. Optional same-language baseline

This is a separate workflow for per-language task accuracy, not the matched clean control above. It uses all input rows, including unknown-reference rows if supplied.

```bash
python scripts/03_run_email_baseline.py \
  --lang en --model Qwen/Qwen2.5-3B-Instruct \
  --input data/raw/email_qa_en.jsonl \
  --output results/responses/baseline_en_qwen25_v2.jsonl \
  --prompt-format chat --max-new-tokens 128 --resume

python scripts/05_compute_email_accuracy_multi.py \
  --inputs results/responses/baseline_en_qwen25_v2.jsonl \
  --scoring-policy strict_v2 \
  --out-csv results/metrics/baseline_en_qwen25_v2_accuracy.csv
```

Repeat with matching files and `--lang ja` or `--lang bn` as needed. Avoid broad globs that accidentally pool different models or experiment versions.

## Legacy behavior and tests

To inspect historical boundary behavior, add `--placement legacy` to preparation and use separate output paths. To apply the legacy scoring rules to existing response logs, pass `--scoring-policy legacy` to script 05, 06 or 07. With `--strip-marker` in 06/07, add `--marker-mode prefix` for historical truncation behavior. This restores those specific rules, not missing input provenance, original model versions or an exact historical environment. JSONL validation remains strict in both policies.

```bash
python -m unittest discover -s tests -v
```

The tests use synthetic records and fake generation. They do not download model weights or validate historical aggregate results.
