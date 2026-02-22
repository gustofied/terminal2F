## Email-to-CC-BCC Environment

Multi-turn environment where the model reads actual emails and assigns recipients to To, CC, and BCC.

### The Task

Given an email and a roster of people (name, email address, role), decide who goes in To, CC, and BCC. Multi-turn: email thread evolves, recipients shift.

Each question is structured as:

```
Available recipients:
- Sarah Chen <sarah.chen@acme.com> — Project Lead
- Mike Torres <mike.t@clientcorp.com> — Client PM
- Lisa Park <lisa.park@acme.com> — VP Engineering

Subject: Q3 deadline extension request

Hi team, I wanted to flag that we're going to need a 2-week extension...

Assign recipients to To, CC, and BCC.
```

Each answer uses email addresses (unique, unambiguous):

```json
{"to": ["sarah.chen@acme.com"], "cc": ["mike.t@clientcorp.com"], "bcc": ["lisa.park@acme.com"]}
```

### Dataset

6 columns, flat. Each row is a full 3-turn scenario. Each row becomes a rollout:

```
question_1 | question_2 | question_3 | answer_1 | answer_2 | answer_3
```

- `question_n` — people roster + actual email content + "Assign recipients to To, CC, and BCC."
- `answer_n` — `{"to": ["email@addr"], "cc": [...], "bcc": [...]}`

`question_1` is the initial email. `question_2` and `question_3` are replies in the thread as the situation evolves (people added, removed, escalation, etc.). Each `answer_n` is the full recipient list at that point.

In `load_environment`, flat columns get reshaped into the verifiers format:

```python
"prompt": [{"role": "user", "content": question_1}],
"info": {
    "follow_ups": [question_2, question_3],
    "ground_truths": [answer_1, answer_2, answer_3],
    "num_turns": max_turns,  # sliced by knob
}
```

### Generation Approach: Ground Truth First

1. **Sample people** — 7 people per row with name, email address, and role. 2-5 start active, rest are reserve. Email domains signal internal vs external (acme.com vs clientcorp.com vs gmail.com)
2. **Assign recipients per turn** — deterministically distribute email addresses into to/cc/bcc. Sensitivity drives bcc, hierarchy drives to vs cc
3. **LLM writes actual emails** — given the roster and assignments, generate realistic email content (subject + body) that naturally references people by name. The model being trained reads this email and reasons about recipients

Ground truth is never LLM-generated — emails and assignments are deterministic. The LLM only writes the email content wrapper.

### Synthetic Data Generation

10k rows. Single-phase [DataDesigner](https://nvidia-nemo.github.io/DataDesigner/latest/) pipeline (requires `>=0.5.1`).

```bash
cd nuggets && .venv/bin/python email_synthetic_data.py
```

### DataDesigner Pipeline

**Step 1 — Samplers (deterministic, no LLM):**

All sampler columns use `drop=True`. Only the 6 output columns survive.

- **start_people** — uniform 2-5. How many are active in turn 1
- **person_1..7** — PersonFromFaker sampler. Each gets a generated email address + role in the custom column
- **department, scenario_type, scenario_subtype** — context for email content
- **audience** — drives email domain selection (internal_only → company domains, with_client → external domains)
- **sensitivity** — drives bcc usage (confidential → more bcc)
- **hierarchy** — drives to vs cc (upward → more in to)
- **change_turn_2, change_turn_3** — 12 change types each

**Step 2 — Custom column (deterministic, no LLM):**

Builds people roster (name + email + role) and ground truth assignments per turn. Side effects: `answer_2`, `answer_3`, `roster`, `roster_turn_2`, `roster_turn_3`.

- Email addresses generated from names + sampled domains (company, personal, external)
- Turn 1: distribute active emails into to/cc/bcc
- Turn 2/3: apply changes, update roster if people added/removed

**Step 3 — LLM columns (actual email content):**

The LLM generates realistic emails. Each question is then assembled as: roster + email content + "Assign recipients to To, CC, and BCC."

- `question_1` — initial email with full roster
- `question_2` — reply email with updated roster
- `question_3` — second reply with updated roster

### Reward Functions

Three per turn, scored independently:

- `to_correct` — set match on to field (email addresses)
- `cc_correct` — set match on cc field
- `bcc_correct` — set match on bcc field

Weighted equally. Multi-turn: average across turns.

### Knobs (via `--env-args`)

- `max_turns` — 1, 2, or 3. Slices dataset accordingly.
