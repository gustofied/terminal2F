import importlib


def load(name: str):
    mod = importlib.import_module(f"{__name__}.{name}")
    return mod.run_agent