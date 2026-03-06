# email-to-cc-bcc

### Overview

- **Environment ID**: `email-to-cc-bcc`
- **Short description**: Given an email scenario with people, roles, and privacy constraints, correctly assign recipients to To, CC, and BCC fields.
- **Tags**: structured-output, rules, email, train, eval, reasoning

### Datasets

- **Primary dataset(s)**: Synthetic email scenarios with contact lists, role tags, and routing policies

### Task

- **Type**: multi-turn
- **Output format expectations**: Structured fields — XML tags or JSON with `to`, `cc`, `bcc` arrays of email addresses
- **Rubric overview**: Verifies correct placement of each recipient based on scenario constraints (who must receive, who should be visible, privacy rules)

### Quickstart

Run an evaluation with default settings:

```bash
prime eval run email-to-cc-bcc
```

Configure model and sampling:

```bash
prime eval run email-to-cc-bcc -m gpt-4.1-mini -n 20 -r 3 -t 1024 -T 0.7
```

### Metrics

| Metric        | Meaning                                        |
| ------------- | ---------------------------------------------- |
| `reward`      | Overall correctness of To/CC/BCC assignment    |
| `to_correct`  | All required To recipients present, no extras  |
| `cc_correct`  | All required CC recipients present, no extras  |
| `bcc_correct` | All required BCC recipients present, no extras |
