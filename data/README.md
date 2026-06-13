# Data

Raw EmailQA/BIPIA data, translated EmailQA files, generated prompt JSONL files, and model response JSONL files are **not included** in this repository.

This is intentional because those files may contain original email text, translated email text, questions, generated prompts, and full model outputs. They should be kept locally and should not be redistributed through this GitHub repository.

## Expected local files

To reproduce the experiment, prepare the following files locally:

```text
data/raw/email_qa_en.jsonl
data/raw/email_qa_ja.jsonl
data/raw/email_qa_bn.jsonl
```

Each JSONL row should contain at least:

```json
{"context": "...", "question": "...", "ideal": "..."}
```

The three language files must be parallel: row `i` in each file should correspond to the same EmailQA item.

## Generated local files

The scripts will generate files such as:

```text
data/prompts/email_prompts_inj_multilang_3321.jsonl
results/responses/*.jsonl
results/metrics/*.csv
```

These files are ignored by Git by default.

## What is included in the repository

This repository includes only:

- experimental code,
- documentation,
- aggregated result tables,
- figures used for reporting.

Raw data and full model outputs should remain local.
