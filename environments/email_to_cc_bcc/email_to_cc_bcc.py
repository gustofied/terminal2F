import verifiers as vf
from datasets import Dataset


def load_environment(**kwargs) -> vf.Environment:
    dataset = Dataset.from_list([
        {
            "prompt": [{"role": "user", "content": "What is capital of Norway?"}],
            "info": {
                "expected_answer": "Oslo",
            },
        },
    ])

    def correct_answer(completion, state, **kwargs) -> float:
        response = completion[-1]["content"].lower()
        expected = state["info"]["expected_answer"].lower()
        return 1.0 if expected in response else 0.0

    rubric = vf.Rubric(funcs=[correct_answer])

    return vf.SingleTurnEnv(dataset=dataset, rubric=rubric)
