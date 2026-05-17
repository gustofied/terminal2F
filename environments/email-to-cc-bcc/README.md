# email-to-cc-bcc

### Overview

- **Environment ID**: `email-to-cc-bcc`
- **Short description**: Given a multi-turn email thread with a roster of people, roles, and privacy constraints, assign recipients to To, CC, and BCC fields for each email.
- **Tags**: structured-output, rules, email, train, eval, reasoning

### Datasets

- **Primary dataset(s)**: `gustofied/email-to-cc-bcc` (Hugging Face)
- **Split sizes**: ~5000 rows (train), 91 rows (test)
- **Synthetic data generation**: [`nuggets/email-to-cc-bcc`](../../nuggets/email-to-cc-bcc) — scripts for generating the dataset. v1 used random recipient assignment (labels not derivable from content, models hit a ~0.44 ceiling). v2 rewrote the pipeline with deterministic routing rules and a post-generation validator; ~5000 of 7500 generated rows passed. See [`synthetic_data_generation_v2.py`](../../nuggets/email-to-cc-bcc/synthetic_data_generation_v2.py).

### Task

- **Type**: multi-turn (1–3 turns, configurable)
- **Output format**: JSON with exactly `to`, `cc`, `bcc` keys, each an array of email addresses
- **Rubric overview**: Jaccard overlap per field per turn, averaged. Weighted sum with format scaffolding.

### Quickstart

```bash
prime eval run email-to-cc-bcc -m gpt-4.1-mini -n 20 -r 3
```

Single-turn on test set:
```bash
prime eval run email-to-cc-bcc -m <model> -n 50 -r 3 -a '{"max_turns": 1, "dataset_split": "test"}'
```

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `dataset_name` | str | `gustofied/email-to-cc-bcc` | HuggingFace dataset path |
| `dataset_split` | str | `train` | Dataset split (`train` or `test`) |
| `max_turns` | int | `3` | Number of turns per conversation (1, 2, or 3) |

### Metrics

| Metric | Weight | Meaning |
| ------ | ------ | ------- |
| `reward` | - | Weighted average of all metrics below |
| `to_correct` | 0.40 | Jaccard overlap for To field, averaged across turns |
| `cc_correct` | 0.40 | Jaccard overlap for CC field, averaged across turns |
| `bcc_correct` | 0.10 | Jaccard overlap for BCC field, averaged across turns |
| `format_correct` | 0.05 | Binary: valid JSON with exactly {to, cc, bcc} keys and list values |
| `email_format` | 0.05 | Fractional: proportion of recipients that are email addresses |

### Notes

- BCC ground truth is empty on ~60% of turns. Empty/empty scores 0.2 (small credit), non-empty when GT is empty scores 0.0 (penalizes hallucination).
- `format_correct` and `email_format` are scaffolding rewards. They help small models learn output format before recipient placement.
- The `test` split (91 rows) was generated independently from the training data using the same pipeline and validator.
