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

All three fields must be nonempty strings. Row i in all three files must refer to the same source item. The generator aligns by row order; if any aligned row has an `id`, all corresponding rows must have the same `id`. This checks IDs but does not reorder or translate records.

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

Invalid JSON, missing/non-string/empty fields, unequal row counts, mismatched IDs, money values or known/unknown status stop validation with a nonzero exit status. The generator applies the same checks before writing prompts. Numerically equivalent amounts such as $200 and $200.00 agree. Language hints remain informational; matching amounts, IDs or answer types do not prove semantic translation equivalence.

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
data/prompts/email_v2_attack_off.jsonl
data/prompts/email_v2_clean_off.jsonl
results/responses/*.jsonl
results/metrics/*.csv
~~~

A filename containing 3321 is a naming convention, not an enforced item count. Check the generator's reported count.

Keep raw inputs, translations, prompts, and full responses local. Published materials under [results/](../results/) are aggregate tables and figures; their provenance and scoring limitations are explained in the [results guide](../results/README.md).
