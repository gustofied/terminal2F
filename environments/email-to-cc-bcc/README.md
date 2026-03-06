# email-to-cc-bcc

### Overview

- **Environment ID**: `email-to-cc-bcc`
- **Short description**: Given a multi-turn email thread with a roster of people, roles, and privacy constraints, assign recipients to To, CC, and BCC fields for each email.
- **Tags**: structured-output, rules, email, train, eval, reasoning

### Datasets

- **Primary dataset(s)**: `gustofied/email-to-cc-bcc` (Hugging Face)
- **Split sizes**: ~5000 rows (train)

### Task

- **Type**: multi-turn (3 turns)
- **Output format**: JSON with `to`, `cc`, `bcc` arrays of email addresses
- **Rubric overview**: Jaccard overlap per field per turn, averaged. Invalid JSON or missing keys scores 0.

### Quickstart

```bash
prime eval run email-to-cc-bcc
```

```bash
prime eval run email-to-cc-bcc -m gpt-4.1-mini -n 20 -r 3 -t 1024
```

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `dataset_name` | str | `gustofied/email-to-cc-bcc` | HuggingFace dataset path |
| `dataset_split` | str | `train` | Dataset split to use |
| `max_turns` | int | `3` | Number of turns per conversation |

### Metrics

| Metric        | Meaning                                              |
| ------------- | ---------------------------------------------------- |
| `reward`      | Weighted average of to/cc/bcc (equal weights)        |
| `to_correct`  | Jaccard overlap for To field, averaged across turns   |
| `cc_correct`  | Jaccard overlap for CC field, averaged across turns   |
| `bcc_correct` | Jaccard overlap for BCC field, averaged across turns  |
