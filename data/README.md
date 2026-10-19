# Data preparation

The repository includes code and aggregate results, but not raw BIPIA/EmailQA records, translated records, generated prompts, or full model responses. These remain local under the repository's current distribution policy.

The instructions below describe the input format for a new run. They do not provide enough information to reconstruct the exact historical dataset by themselves.

## Expected local files

~~~text
data/raw/email_qa_en.jsonl
data/raw/email_qa_ja.jsonl
data/raw/email_qa_bn.jsonl
~~~

Each non-empty line must be a JSON object with these fields:

~~~json
{"context": "...", "question": "...", "ideal": "..."}
~~~

| Field | Meaning |
| --- | --- |
| context | Email content in the file's language |
| question | Question corresponding to that email |
| ideal | Reference answer, preserving the intended value and known/unknown status |

Row i in all three files must refer to the same source item. The generator aligns files by row order rather than joining them by a stable source ID.

The documented experiment uses 41 known-answer items. The --known-only option filters using the English reference answer; it does not select a predefined list of 41 items. The resulting count depends on the supplied data.

## Validation

Run from the repository root:

~~~bash
python scripts/00_validate_parallel_emailqa.py \
  --en data/raw/email_qa_en.jsonl \
  --ja data/raw/email_qa_ja.jsonl \
  --bn data/raw/email_qa_bn.jsonl \
  --out-json results/metrics/parallel_validation.json
~~~

The validator inspects JSON parsing, row counts, required fields, answer types, money-answer alignment, and rough character-based language hints.

Inspect the report and console warnings before generation. The script can finish after reporting mismatches; a completed command is not evidence that every check passed. Matching answer amounts or types also does not prove that translated questions and emails are semantically aligned.

## Provenance needed for exact reproduction

The public documentation does not fully record:

- The upstream dataset version and stable IDs of selected source items.
- The translation tool/model, version, settings, and translation instructions.
- Whether emails, questions, and reference answers were translated or normalized separately.
- Human review, alignment corrections, and quality checks for each language.
- Input-file hashes and the exact run configuration associated with each published table.

Record these for a new experiment. A translation process or human validation step should not be assumed merely because expected language files are named here.

## Local generated files

Typical outputs include:

~~~text
data/prompts/email_prompts_inj_multilang_3321_defense_off.jsonl
data/prompts/email_prompts_inj_multilang_3321_defense_on.jsonl
results/responses/*.jsonl
results/metrics/*.csv
~~~

A filename containing 3321 is a naming convention, not an enforced item count. Check the generator's reported count.

Keep raw inputs, translations, prompts, and full responses local. Published materials under [results/](../results/) are aggregate tables and figures; their provenance and scoring limitations are explained in the [results guide](../results/README.md).
