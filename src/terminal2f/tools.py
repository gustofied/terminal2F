from __future__ import annotations

import logging
from dataclasses import dataclass

from terminal2f.envs import Session

log = logging.getLogger(__name__)

# T ******************* 2 ******************* F -- TOOLS / FUNCTION

# tools

# a tool to use, maybe two, maybe more tools soony
# it's not the current task of the project to do tools very proper, but with time introduce schema generation, and etc better flow..
# future reference, https://github.com/1rgs/nanocode/blob/master/nanocode.py

@dataclass
class T2FTool:
    """A tool to understand what t2f, terminal2F is"""
    name: str = "t2ftool"
    description: str = "A tool to understand what t2f, terminal2F is"

    @property # really needing @property here? likley not..
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "integer",
                            "description": """code to unlock different information on terminal2f. Valid
                                            codes: 10 (what it is), 20 (what kind of project), 30 (development time), 40
                                            (tech stack), 50+ (origin story)""",
                        }
                    },
                    "required": ["code"],
                },
            },
        }

    def execute(self, code: int):
        match code:
            case 10:
                return("terminal2f is a coding project")
            case 20:
                return("terminal2f is a observablity project")
            case 30:
                return("terminal2f takes a long time to code")
            case 40:
                return("terminal2f is just coded in python")
            case 50:
                return("terminal2f was made to reborn me")
            case _:
                return("codes are eiher any number abouve 50, or exact 10, 20, 30, 40")

# future: back store externally (disk/sqlite/s3), give LLM a manifest + preview, read/write on demand. same tool schema.

@dataclass
class WriteArtifact:
    """Write a keyed artifact to the agent's scratchpad."""
    store: list
    max_entries: int = 0  # 0 = unbounded
    name: str = "write_artifact"
    description: str = "Write a value to the scratchpad under a key. Overwrites if key exists."

    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "description": "The key to store under"},
                        "value": {"type": "string", "description": "The value to store"},
                    },
                    "required": ["key", "value"],
                },
            },
        }

    def execute(self, key: str, value: str):
        # Overwrite if key exists
        for entry in self.store:
            if entry["key"] == key:
                entry["value"] = value
                log.debug(f"object_store overwrite: {key}")
                return value
        if self.max_entries and len(self.store) >= self.max_entries:
            log.debug(f"object_store full ({self.max_entries})")
            return "full"
        self.store.append({"key": key, "value": value})
        log.debug(f"object_store: {self.store}")
        return value

@dataclass
class ReadArtifact:
    """Read a keyed artifact from the agent's scratchpad."""
    store: list
    name: str = "read_artifact"
    description: str = "Read a value from the scratchpad by key. Returns all keys if no key given."

    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "description": "The key to read. Omit to list all keys."},
                    },
                    "required": [],
                },
            },
        }

    def execute(self, key: str | None = None):
        if key is None:
            return str([e["key"] for e in self.store])
        for entry in self.store:
            if entry["key"] == key:
                return entry["value"]
        return ""

@dataclass
class Delegate:
    """Delegate a task to a sub-agent. Runs it to completion, returns its answer."""
    session: Session | None = None  # wired after Session is created
    name: str = "delegate"
    description: str = "Delegate a task to a sub-agent. Give it a clear instruction and it will work independently and return its result."

    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "instruction": {"type": "string", "description": "The task to delegate to the sub-agent"},
                    },
                    "required": ["instruction"],
                },
            },
        }

    def execute(self, instruction: str):
        assert self.session is not None, "Delegate tool not wired to a Session"
        agent_name = f"sub_{len(self.session.agents)}"
        self.session.spawn(agent_name, instruction)
        # run only the new sub-agent to completion
        _, sub_runner = self.session.agents[-1]
        sub_runner()  # run to Finished
        return sub_runner.result or ""

t2f_tool = T2FTool()

# TODO: remove — module-level tools/tool_registry are dead code (registry is built per-runner in FSM.__init__ and LOOP.__call__)
tools = [t2f_tool]
tool_registry = {t.name: t.execute for t in tools}



# T ******************* 2 ******************* F -- TOOLS //  (description, schema, function)

