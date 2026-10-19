# Stored results

This directory contains existing aggregate CSVs and figures. It includes main instruction-model summaries and additional accuracy outputs with base-model filenames. The files should not be treated as one uniformly configured evaluation.

The README clarification does not recompute results. The public materials do not document a complete reconciliation of current scripts, historical outputs, and report-derived summaries against the original response logs.

## File guide

| File or group | Contents and interpretation |
| --- | --- |
| [asr/asr_summary_by_condition.csv](asr/asr_summary_by_condition.csv) | Main stored ASR, success counts, and sample counts for two instruction models with reminder off/on |
| [asr/asr_by_combo.csv](asr/asr_by_combo.csv) | ASR by language/placement combination |
| [asr/asr_by_axis.csv](asr/asr_by_axis.csv) | Summaries by experimental axis |
| [asr/asr_by_context_lang.csv](asr/asr_by_context_lang.csv), [asr/asr_by_qa_lang.csv](asr/asr_by_qa_lang.csv), [asr/asr_by_injection_lang.csv](asr/asr_by_injection_lang.csv), [asr/asr_by_position.csv](asr/asr_by_position.csv) | Language- and placement-level aggregates |
| [accuracy/baseline_accuracy_summary.csv](accuracy/baseline_accuracy_summary.csv) | Stored known/unknown baseline summaries; cites a report not included here |
| [accuracy/accuracy_before_after.csv](accuracy/accuracy_before_after.csv) | Stored before/after summaries; not a fully matched causal comparison under the current workflow |
| [accuracy/accuracy_summary.csv](accuracy/accuracy_summary.csv), [accuracy/accuracy_summary_untranslated.csv](accuracy/accuracy_summary_untranslated.csv) | Additional baseline outputs whose source filenames identify base-model runs; do not pool with the main instruction-model summaries |
| [accuracy/accuracy_after_injection_summary_raw.csv](accuracy/accuracy_after_injection_summary_raw.csv) | Mixed run summaries, including base-model and instruction-model filenames |
| [accuracy/accuracy_after_injection_by_combo.csv](accuracy/accuracy_after_injection_by_combo.csv) | Detailed post-injection accuracy outputs; retain run identifiers when analyzing |
| [hypotheses/h1_h2_summary.csv](hypotheses/h1_h2_summary.csv) | Stored H1/H2 descriptive comparisons; supported labels are not significance tests |
| [figures/](figures/) | Existing reporting figures; interpret with the associated metric and provenance limitations |

Filenames containing translated, untranslated, or base do not by themselves establish the full translation or model configuration. Consult original run records before comparing them.

## Main stored ASR

| Model | Reminder | Successes | Prompts | Stored ASR |
| --- | --- | ---: | ---: | ---: |
| Llama-3.2-3B-Instruct | off | 1064 | 3321 | 0.320 |
| Llama-3.2-3B-Instruct | on | 1409 | 3321 | 0.424 |
| Qwen2.5-3B-Instruct | off | 699 | 3321 | 0.210 |
| Qwen2.5-3B-Instruct | on | 842 | 3321 | 0.254 |

Source: [asr_summary_by_condition.csv](asr/asr_summary_by_condition.csv). Values are reproduced as stored, including rounding. The prompt count represents combinations over 41 source items, not 3,321 independent source examples.

## Provenance

Some CSVs cite graduation_report.pdf sections or tables. That report is not included in this repository. Others identify local .txt or .jsonl source filenames without distributing those files.

The public mapping from historical files to current script versions, model revisions, translation versions, and generation settings is incomplete. Aggregate files can be inspected, but the public repository alone does not support full verification or exact regeneration of historical results.

There is no documented end-to-end command that regenerates every curated table and figure. Commands in [docs/COMMANDS.md](../docs/COMMANDS.md) write new local metrics; they do not automatically replace or validate these stored outputs.

## Interpretation limits

- **Reminder condition:** defense_on adds a short security-policy reminder. The recorded ASR increase should not be generalized to other defenses.
- **Position:** the current generator puts start/end instructions outside the email delimiters and middle instructions inside. Position and boundary placement are confounded.
- **Baseline:** current baseline and injection runners differ in language combinations, prompt templates, and default output-token limits. Before/after deltas should not be attributed solely to injection.
- **Unknown-answer scoring:** current scoring code can accept any response without a detected dollar amount, including an empty response. This is not strict semantic accuracy. Its effect on historical unknown-answer summaries requires checking their original scoring code and logs.
- **Money scoring:** extracted money strings are compared; equivalent numerical values in different formats can score differently.
- **Marker stripping:** post-injection scoring with --strip-marker discards the marker and everything after its first occurrence.
- **Hypotheses:** resource rank is a configured assumption. Reported differences and supported fields do not include confidence intervals or significance tests.

See the [main README](../README.md) for implementation details. Corrected metrics should be generated and labeled separately after validation, preserving the provenance of these historical files.

## Local-only artifacts

Raw inputs, translations, generated prompts, full responses, and new intermediate metrics remain local:

~~~text
data/raw/
data/prompts/
results/responses/
results/metrics/
~~~

See [data/README.md](../data/README.md) for preparation and provenance requirements.
