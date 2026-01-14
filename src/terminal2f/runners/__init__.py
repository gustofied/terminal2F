import importlib
from typing import Any


class RunnerAPI:
    def __init__(self, mod: Any):
        self._m = mod

    def __call__(self, *args, **kwargs):
        return self._m.run_agent(*args, **kwargs)

    def new_memory(self, agent):
        return self._m.new_memory(agent)

    def reset(self, agent, memory) -> None:
        self._m.reset(agent, memory)


def load(name: str) -> RunnerAPI:
    m = importlib.import_module(f"{__name__}.{name}")
    for fn in ("run_agent", "new_memory", "reset"):
        if not callable(getattr(m, fn, None)):
            raise AttributeError(f"Runner module '{name}' must define {fn}(...)")
    return RunnerAPI(m)
