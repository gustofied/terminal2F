import json
import re
import verifiers as vf
from datasets import Dataset, load_dataset


class EmailEnv(vf.MultiTurnEnv):
    @vf.stop
    async def max_turns_reached(self, state: vf.State) -> bool:
        return len(state["trajectory"]) >= state["info"]["num_turns"]

    async def env_response(self, messages: vf.Messages, state: vf.State, **kwargs) -> vf.Messages:
        turn = len(state["trajectory"]) - 1
        follow_ups = state["info"]["follow_ups"]
        return [{"role": "user", "content": follow_ups[turn]}] if 0 <= turn < len(follow_ups) else []


def extract_json(text: str) -> dict:
    """Find the last balanced {...} JSON object in text."""
    for i in range(len(text) - 1, -1, -1):
        if text[i] == '}':
            depth, start = 0, None
            for j in range(i, -1, -1):
                if text[j] == '}':
                    depth += 1
                elif text[j] == '{':
                    depth -= 1
                if depth == 0:
                    start = j
                    break
            if start is not None:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    continue
    return {}


def set_overlap(predicted: set, expected: set) -> float:
    """Jaccard index: |intersection| / |union|. 1.0 if both empty."""
    union = predicted | expected
    return len(predicted & expected) / len(union) if union else 1.0


def score_turn(response: str, ground_truth: str, field: str) -> float:
    """Score a single turn for a single field (to/cc/bcc)."""
    pred = set(extract_json(response).get(field, []))
    expected = set(extract_json(ground_truth).get(field, []))
    return set_overlap(pred, expected)


def score_field(completion: list[dict], state: dict, field: str) -> float:
    """Average score for a field across all expected turns. Missing turns count as 0."""
    responses = [m["content"] for m in completion if m["role"] == "assistant"]
    ground_truths = state["info"]["ground_truths"]
    n = len(ground_truths)
    if not n:
        return 0.0
    scored = min(len(responses), n)
    return sum(score_turn(responses[i], ground_truths[i], field) for i in range(scored)) / n


def load_environment(
    dataset_name="gustofied/email-to-cc-bcc",
    dataset_split="train",
    max_turns: int = 3,
    **kwargs,
) -> vf.Environment:

    dataset = load_dataset(path=dataset_name, split=dataset_split)
    data = []
    for row in dataset:
        prompt = f"""You are given an email thread. For each email, assign the correct To, CC, and BCC recipients. Recipients may change between emails as the situation evolves.

Roster:
{row["email_list"]}

{row["question_1"]}

Reply with JSON only: {{"to": [...], "cc": [...], "bcc": [...]}}
Use email addresses, not names."""

        follow_ups = [
            f"Next email in the thread:\n\n{row[f'question_{i}']}\n\n"
            f"Reply with JSON only."
            for i in range(2, max_turns + 1)
        ]

        data.append({
            "prompt": [{"role": "user", "content": prompt}],
            "info": {
                "follow_ups": follow_ups,
                "ground_truths": [row[f"answer_{i}"] for i in range(1, max_turns + 1)],
                "num_turns": max_turns,
            },
        })

    dataset = Dataset.from_list(data)

    async def to_correct(completion, state, **kwargs):
        return score_field(completion, state, "to")

    async def cc_correct(completion, state, **kwargs):
        return score_field(completion, state, "cc")

    async def bcc_correct(completion, state, **kwargs):
        return score_field(completion, state, "bcc")

    rubric = vf.Rubric(
        funcs=[to_correct, cc_correct, bcc_correct],
        weights=[1/3, 1/3, 1/3],
    )

    return EmailEnv(
        dataset=dataset,
        rubric=rubric,
        max_turns=max_turns,
        system_prompt="You are an email assistant",
    )
