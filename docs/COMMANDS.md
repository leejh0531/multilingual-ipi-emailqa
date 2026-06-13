# Command order

## 0. Validate parallel data

```bash
python scripts/00_validate_parallel_emailqa.py \
  --en data/raw/email_qa_en.jsonl \
  --ja data/raw/email_qa_ja.jsonl \
  --bn data/raw/email_qa_bn.jsonl \
  --out-json results/metrics/parallel_validation.json
```

## 1. Build prompt dataset

Defense off:

```bash
python scripts/01_prepare_email_ipi_multilingual.py \
  --en data/raw/email_qa_en.jsonl \
  --ja data/raw/email_qa_ja.jsonl \
  --bn data/raw/email_qa_bn.jsonl \
  --known-only \
  --output data/prompts/email_prompts_inj_multilang_3321_defense_off.jsonl
```

Defense on:

```bash
python scripts/01_prepare_email_ipi_multilingual.py \
  --en data/raw/email_qa_en.jsonl \
  --ja data/raw/email_qa_ja.jsonl \
  --bn data/raw/email_qa_bn.jsonl \
  --known-only \
  --defense \
  --output data/prompts/email_prompts_inj_multilang_3321_defense_on.jsonl
```

## 2. Run IPI inference

```bash
python scripts/02_run_email_ipi_inference.py \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --input data/prompts/email_prompts_inj_multilang_3321_defense_off.jsonl \
  --output results/responses/llama32_3b_instruct_defense_off.jsonl \
  --prompt-format chat \
  --resume
```

```bash
python scripts/02_run_email_ipi_inference.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --input data/prompts/email_prompts_inj_multilang_3321_defense_off.jsonl \
  --output results/responses/qwen25_3b_instruct_defense_off.jsonl \
  --prompt-format chat \
  --resume
```

## 3. Compute ASR

```bash
python scripts/04_compute_asr_ipi_multilingual.py \
  --input results/responses/qwen25_3b_instruct_defense_off.jsonl \
  --out-csv results/metrics/qwen25_3b_instruct_defense_off_asr.csv
```

## 4. Baseline inference

```bash
python scripts/03_run_email_baseline.py \
  --lang en \
  --model Qwen/Qwen2.5-3B-Instruct \
  --input data/raw/email_qa_en.jsonl \
  --output results/responses/baseline_en_qwen25_3b_instruct.jsonl \
  --prompt-format chat
```

Repeat for `ja` and `bn`.

## 5. Baseline accuracy

```bash
python scripts/05_compute_email_accuracy_multi.py \
  --inputs 'results/responses/baseline_*_qwen25_3b_instruct.jsonl' \
  --out-csv results/metrics/baseline_qwen25_3b_instruct_accuracy.csv
```

## 6. QA accuracy after injection

```bash
python scripts/06_compute_accuracy_by_combo.py \
  --input results/responses/qwen25_3b_instruct_defense_off.jsonl \
  --gold-key orig_ideal \
  --strip-marker \
  --out-csv results/metrics/qwen25_3b_instruct_after_accuracy_by_combo.csv
```

## 7. Mixed outcome analysis

```bash
python scripts/07_compute_metrics_with_mixed.py \
  --input results/responses/qwen25_3b_instruct_defense_off.jsonl \
  --strip-marker \
  --out-csv results/metrics/qwen25_3b_instruct_mixed.csv
```

## 8. H1/H2 summary

```bash
python scripts/08_summarize_hypotheses.py \
  --input results/responses/qwen25_3b_instruct_defense_off.jsonl \
  --model-tag qwen25_3b_instruct_defense_off \
  --out-csv results/metrics/qwen25_3b_instruct_hypotheses.csv
```
