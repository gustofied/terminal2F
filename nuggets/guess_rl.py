# based on https://ivanleo.com/blog/spinning-up-rl

import numpy as np
from abc import ABC, abstractmethod

rng = np.random.default_rng(seed=1420)

class Guesser:
    def __init__(self, min: int, max: int):
        self.action_space = list(range(min, max + 1))
        # starting with a uniform distribution
        self.policy = np.ones(len(self.action_space)) / len(self.action_space)

    def guess(self, size: int = 1) -> list[int]:
        """Makes guesses based on current policy"""
        return rng.choice(
            a = self.action_space,
            size = size,
            p = self.policy,
        )  # ty:ignore[invalid-return-type]

    def update_policy(self, reward_vector: list[float]):
        """Updates the policy basde on a full reward vector."""
        rewards = np.array(reward_vector)
        self.policy = self.policy + rewards
        self.policy = np.maximum(self.policy, 1e-8)
        self.policy = self.policy / self.policy.sum()



class Rubric(ABC):
    """Base class for reward rubrics"""

    @abstractmethod
    def evaluate_batch(self, guesses: list[int]) -> list[float]:
        """Evaluate a batch of guesses and retunrs a reward vector"""
        pass

class BinaryRubric(Rubric):
    """A rubric that provides reward vectors for entire batches of guesses."""

    def __init__(self, target: int, action_space: list[int]):
        self.target = target
        self.action_space = action_space

    def evaluate_batch(self, guesses: list[int]) -> list[float]:
        """
        Evaluates a batch of guesses and returns a reward vector.
        The reward vector has the same dimension as the action space
        """
        reward_vector = [0.0] * len(self.action_space)

        for guess in guesses:
            reward = 1.0 if guess == self.target else 0.0
            action_idx = self.action_space.index(guess)
            reward_vector[action_idx] += reward

        return [r / len(guesses) for r in reward_vector]


if __name__ == "__main__":
    MIN = 0
    MAX = 10
    TARGET = 6
    BATCH_SIZE = 10
    ITERATIONS = 20

    agent = Guesser(MIN, MAX)
    rubric = BinaryRubric(target=TARGET, action_space=agent.action_space)

    target_idx = agent.action_space.index(TARGET)
    print(f"Target number: {TARGET}, Probability: {agent.policy[target_idx]:.4f}\n")

    for i in range(ITERATIONS):
        guesses = agent.guess(BATCH_SIZE)
        reward_vector = rubric.evaluate_batch(guesses)
        agent.update_policy(reward_vector)
        print(f"Iteration {i + 1}: P(guess={TARGET}) = {agent.policy[target_idx]:.3f}")
