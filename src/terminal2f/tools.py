from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class T2FTool:
    """A tool to understand what t2f, terminal2F is"""
    name: str = "t2ftool"
    description: str = "A tool to understand what t2f, terminal2F is"

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


t2f_tool = T2FTool()
tools = [t2f_tool]
tool_registry = {t.name: t.execute for t in tools}
