# Dataset generation requires data-designer>=0.5.1 (separate venv — pyarrow conflict):
#   cd nuggets && uv venv && uv pip install 'data-designer>=0.5.1' faker
#   .venv/bin/python email_to_cc_bcc_synthetic_data_generation.py --generate 10000
# This generates the 7-column dataset (email_list, question_1-3, answer_1-3) via:
#   1. Samplers — people, context, change types (deterministic)
#   2. Custom column — ground truth to/cc/bcc assignments (deterministic)
#   3. LLM columns — scenario writing via OpenRouter (openrouter-text alias)

import verifiers as vf
from datasets import Dataset


dataset_example = Dataset.from_list([
        {
            "prompt": [{"role": "user", "content": "What is capital of Norway?"}],
            "info": {
                "expected_answer": "Oslo",
            },
        },
    ])

class EmailEnv(vf.MultiTurnEnv):
    async def env_response(self, messages: vf.Messages, state: vf.State) -> vf.Messages:
        return []

def load_environment(max_turns: int = 1, **kwargs) -> vf.Environment:
    
    dataset = dataset_example

    async def correct_answer(completion, state, **kwargs) -> float:
        response = completion[-1]["content"].lower()
        expected = state["info"]["expected_answer"].lower()
        return 1.0 if expected in response else 0.0

    async def length_answer(completion, **kwargs) -> float:
        response = completion[-1]["content"]
        return 1.0 if len(response) < 200 else 0.5

    rubric = vf.Rubric(
        funcs=[correct_answer, length_answer],
        weights=[0.8, 0.2],
    )
    
    return EmailEnv(dataset=dataset, rubric=rubric, max_turns=max_turns)

