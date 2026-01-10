from .regular import RegularRunner


def make_runner(kind: str = "regular"):
    kind = (kind or "regular").lower().strip()

    if kind == "regular":
        return RegularRunner

    # Not implemented yet.
    # See runners/fsm.py, runners/pda.py, runners/tm.py for what they will be.
    raise NotImplementedError(f"Runner '{kind}' is not implemented yet.")
