import importlib
from typing import Any


class RunnerAPI:
    """
    run_agent = load("regular")
    run_agent(agent, "...", episode_id=..., step=..., ui=...)

    Also:
      run_agent.reset(agent)
    """

    def __init__(self, mod: Any):
        self._mod = mod

    def __call__(self, *args, **kwargs):
        return self._mod.run_agent(*args, **kwargs)

    def reset(self, agent) -> None:
        self._mod.reset(agent)


def load(name: str) -> RunnerAPI:
    mod = importlib.import_module(f"{__name__}.{name}")

    if not hasattr(mod, "run_agent") or not callable(mod.run_agent):
        raise AttributeError(f"Runner module '{name}' must define run_agent(...)")

    if not hasattr(mod, "reset") or not callable(mod.reset):
        raise AttributeError(f"Runner module '{name}' must define reset(agent)")

    return RunnerAPI(mod)
