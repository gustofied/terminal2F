## Email-to-CC-BCC Environment

Multi-turn environment where the model reads actual emails and assigns recipients to To, CC, and BCC.

### The Task

Given an email and a list of 7 people (name, email address, role), decide who goes in To, CC, and BCC. Multi-turn: email thread evolves, recipients shift.

The environment assembles the full prompt at runtime:

```
Available recipients:
- Sarah Chen <sarah.chen@acme.com> — Project Lead
- Mike Torres <mike.t@clientcorp.com> — Client PM
- Lisa Park <lisa.park@acme.com> — VP Engineering
...

Subject: Q3 deadline extension request

Hi team, I wanted to flag that we're going to need a 2-week extension...

Assign recipients to To, CC, and BCC.
```

Answer format — email addresses, JSON:

```json
{"to": ["sarah.chen@acme.com"], "cc": ["mike.t@clientcorp.com"], "bcc": ["lisa.park@acme.com"]}
```

### Dataset

7 columns, flat. Each row is a full 3-turn scenario:

```
email_list | question_1 | question_2 | question_3 | answer_1 | answer_2 | answer_3
```

- `email_list` — all 7 people, always present (`name <email> — role`)
- `question_n` — email content (subject + body). The environment prepends `email_list` and appends the instruction at runtime
- `answer_n` — `{"to": [...], "cc": [...], "bcc": [...]}`

`question_1` is the initial email. `question_2` and `question_3` are replies as the situation evolves (people added, removed, escalation, etc.).

In `load_environment`, flat columns get reshaped into the verifiers format:

```python
"prompt": [{"role": "user", "content": email_list + question_1 + instruction}],
"info": {
    "follow_ups": [question_2, question_3],
    "ground_truths": [answer_1, answer_2, answer_3],
    "num_turns": max_turns,
}
```

### Generation Approach

Ground truth first — the LLM never decides who goes where.

1. **Sample people** — 7 per row with name, email, role. 2–6 start active, rest are reserve. One company domain per row, one external domain. Roles sampled without replacement
2. **Assign recipients per turn** — deterministic. Sensitivity drives bcc, hierarchy drives to vs cc. Changes between turns (escalation, person added, made confidential, etc.) retry if they'd be a no-op
3. **LLM writes email content** — given the roster and visible recipients (to/cc only, no bcc), generate realistic email thread. The LLM never sees bcc assignments

### Synthetic Data Generation

10k rows. Single-phase [DataDesigner](https://nvidia-nemo.github.io/DataDesigner/latest/) pipeline (`>=0.5.1`).

```bash
cd nuggets
uv venv && uv pip install 'data-designer>=0.5.1' faker
.venv/bin/python email_to_cc_bcc_synthetic_data_generation.py --generate 10000
```

Preview:

```bash
.venv/bin/python email_to_cc_bcc_synthetic_data_generation.py --preview 5
```

### Pipeline Details

**Samplers (deterministic, no LLM):**

- `start_people` — uniform 2–6
- `person_1..7` — PersonFromFaker
- `department`, `scenario_type`, `scenario_subtype` — context for email content
- `audience` — internal_only, with_client, with_vendor, etc. Drives domain selection
- `sensitivity` — public, internal, confidential, restricted. Drives bcc usage
- `hierarchy` — upward, downward, lateral, mixed. Drives to vs cc
- `change_turn_2`, `change_turn_3` — 12 change types (person_added, escalation, made_confidential, delegation, etc.)

**Custom column (deterministic, no LLM):**

Builds `email_list` (all 7 people) and ground truth `answer_1/2/3`. Email addresses generated from faker names + sampled domains. Turn 1 distributes active emails into to/cc/bcc. Turns 2/3 apply changes with no-op retry.

**LLM columns (email content):**

The LLM generates realistic emails given the roster and visible recipients (to/cc only). It never sees bcc. Strict rules: no placeholders, no mention of bcc, spell names exactly.

- `question_1` — initial email
- `question_2` — reply after change
- `question_3` — second reply after another change

### Reward Functions

Verifiers scores the full rollout once at the end — the rubric calls each reward function with the complete state (all turns, all ground truths). Each function internally loops over turns and averages. The rubric combines them via weighted sum into a single reward scalar for the trainer.

Planned reward functions (v1):

- `visible_correct` — set match on To ∪ CC (avoids noisy To/CC split)
- `bcc_correct` — set match on BCC
- `format_correct` — valid JSON with correct keys

Each function averages its score across turns internally. Weighted equally by the rubric.

### Ongoing Discussion

**To vs CC learnability.** The current `distribute_emails()` shuffles active people and assigns To/CC somewhat arbitrarily. The model sees the email text + people list with roles, so it has *some* signal (hierarchy, who's addressed, who's asked to act), but the exact To/CC split has randomness the model can't fully recover. BCC is cleaner — driven directly by sensitivity.

In practice this means: `bcc_correct` should learn well, `to_correct` and `cc_correct` individually will be noisier. The model will likely learn the *visible set* (To + CC combined) better than the exact split.

**v2 fix:** replace the shuffle with deterministic role-based rules — e.g., person being asked to act → To, their manager → CC when hierarchy=upward, external stakeholders → CC when audience=with_client. This makes the split recoverable from the observation (email content + role labels) and gives GRPO a clean signal.

**Other things to watch for:**
- BCC recipients sometimes get greeted in the email text (~rare, but contradictory)
- LLM occasionally duplicates Q1 text verbatim for Q2 (~5% of rows). Answers still differ, but the thread doesn't evolve
- Sender sometimes appears in their own BCC

None of these block a first training run. Plan is: train on v1, check reward curves. If `to_correct`/`cc_correct` plateau while `bcc_correct` climbs, that confirms the To/CC noise and we tighten for v2.

**Reward restructuring for v1.** Since To vs CC is noisy but the visible set (To ∪ CC) is learnable, scoring them separately risks GRPO amplifying noise — the model chases an impossible target and you get policy thrash from within-group ranking on random distinctions. Practical fix for v1 without regenerating data: restructure the reward to score `visible_correct` (set match on To ∪ CC) + `bcc_correct` (set match on BCC) + `format_correct` (valid JSON). Drop the individual To/CC rewards. Re-add them in v2 once `distribute_emails()` is deterministic and the split is recoverable.

**Deterministic role-based routing (v2).** Replace the shuffle with rules driven by scenario type, hierarchy, audience, and roles. E.g., the person being asked to act → To (picked by scenario-to-role priority), managers/stakeholders → CC (shaped by hierarchy), compliance observers from reserve → BCC (when sensitivity is high). Active participants never end up in BCC. This makes the exact To/CC/BCC split derivable from the observation without turning it into greeting-parsing.

**Metadata columns.** The generation pipeline samples `scenario_type`, `audience`, `sensitivity`, `hierarchy` but currently drops them from the final dataset. If we expose them to the model, the task gets easier but more deterministic. If we hide them, the model infers sensitivity/hierarchy from the email prose — harder, more realistic, but noisier. Leaning towards hiding them for now. Revisit based on reward curves.

### v2 Discussion and Upgrades

**mbox format for realistic email state.** Archipelago's mail MCP server uses mbox — the standard Unix mailbox format. All emails in a single file, separated by `From ` lines. Each email is a full RFC 822 message with real headers (From, To, Cc, Bcc, Subject, Date, In-Reply-To, References). Python stdlib has `mailbox.mbox` to read/write it.

Instead of passing email content as plain text strings, v2 could store the thread as an mbox file per rollout. The model interacts with actual email structure — threading via In-Reply-To, proper headers, multipart bodies. This makes the task harder and more realistic: the model must parse real email format, not a simplified prompt.

Archipelago's full mail server exposes 7 tools via FastMCP: `list_mails`, `read_mail`, `search_mail`, `send_mail`, `reply_mail`, `reply_all_mail`, `forward_mail`. All backed by mbox files (~2,100 lines). Their dual-mode pattern is worth noting — individual tools or a single `mail` meta-tool with an action parameter, controlled by env var.

**What this means for us:** v2 could evolve from "read this email, assign recipients" to "interact with a mail server via MCP tools, manage an inbox." The model would `read_mail` to see the thread, then `reply_mail` or `forward_mail` with the right recipients. Scoring shifts from JSON extraction to checking the actual sent email's headers in the mbox file. This aligns with the Zapier/MCP direction — same `env_response` pattern, just routing tool calls to a FastMCP mail server instead of returning follow-up strings.

### Knobs (`--env-args`)

- `max_turns` — 1, 2, or 3. Slices dataset accordingly.
