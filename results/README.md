# Results

This directory contains **public aggregated results** for the paper/thesis experiment.

The public results are organized around instruction-tuned models and defense on/off comparison:

```text
Llama 3.2-3B-Instruct
Qwen 2.5-3B-Instruct

defense_off
defense_on
```

Raw EmailQA/BIPIA data, generated prompt JSONL files, and full model response JSONL files are not included.

## Directory layout

```text
results/
├── asr/
├── accuracy/
├── figures/
└── hypotheses/
```

## `asr/`

ASR-related CSV files.

```text
asr_summary_by_condition.csv
asr_by_combo.csv
asr_by_axis.csv
asr_by_position.csv
asr_by_context_lang.csv
asr_by_qa_lang.csv
asr_by_injection_lang.csv
```

## `accuracy/`

QA accuracy-related CSV files.

```text
baseline_accuracy_summary.csv
accuracy_before_after.csv
accuracy_summary.csv
accuracy_summary_untranslated.csv
accuracy_after_injection_by_combo.csv
accuracy_after_injection_summary_raw.csv
```

## `hypotheses/`

H1/H2 hypothesis summary.

```text
h1_h2_summary.csv
```

## `figures/`

PNG figures for reporting.

```text
baseline_known_accuracy.png
baseline_unknown_accuracy.png
asr_by_position.png
h1_h2_delta.png
accuracy_before_after.png
```

## Not included

The following generated files are not included in the public repository:

```text
results/responses/
results/metrics/
data/raw/
data/prompts/
*.jsonl
```

Full model responses and prompt JSONL files should remain local.
