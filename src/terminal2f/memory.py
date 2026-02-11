from __future__ import annotations


class Memory:
    """Holds all the data an agent could ever touch.
    The stack is the single source of truth for interaction history.
    What gets used and how it gets read is up to the automaton (FSM/PDA/Loop),
    not memory itself. Memory is just storage."""

    def __init__(self):
        self.messages: list = []           # Raw message dicts. LOOP uses this directly.
        self.stack: list = []              # Interaction stack — typed entries, append-only.
        self.object_store: list = []       # Long-term artifact storage. TM-level memory.

    def push(self, msg) -> None:
        self.messages.append(msg)

    def get_messages(self, k: int | None = None) -> list:
        return list(self.messages if k is None else self.messages[-k:])

    def tape(self) -> list:
        """Everything. Messages, stack, and object store. The full picture."""
        return [self.messages, self.stack, self.object_store]
