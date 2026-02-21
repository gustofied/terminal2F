import verifiers as vf
from datasets import Dataset

def load_environment(**kwargs) -> vf.Environment:
    dataset = Dataset.from_list([
        {"prompt": [{"role": "user", "content": "What is capital of Norway?"}], "answer": "4"},
        {"prompt": [{"role": "user", "content": "What is capital of Sweden?"}], "answer": "15"},
    ])

    # Your reward function
    async def correct_answer(completion, answer) -> float:
        response = completion[-1]["content"]
        return 1.0 if answer in response else 0.0
    
    rubric = vf.Rubric(funcs=[correct_answer])
    
    return vf.SingleTurnEnv(dataset=dataset, rubric=rubric)
