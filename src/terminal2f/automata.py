# Automata / Runners
# Each class is both an automaton (computational model) and a runner (agent execution loop).
# Most see them as runners, but since im following the noted paper, and like the lingo
# Automata is what it is, we got:
# LOOP: basic chat loop, no structured state
# FSM: finite state machine, bounded context window (k=3)
# PDA: pushdown automaton, stack-top driven, full history
# LBA: linear-bounded automaton, PDA + bounded scratchpad
# TM: turing machine, PDA + unbounded scratchpad
# All are callable: runner(agent, input, memory, tools=...)() -> result

from __future__ import annotations

from enum import StrEnum, auto

import json
import rerun as rr

from terminal2f.agent import Agent
from terminal2f.memory import Memory
from terminal2f.states import (
    UserMessage, AssistantMessage, ToolCall, ToolResult,
    Finished,
)

class LOOP:
    def __init__(self, agent: Agent, user_input: str, memory: Memory, *, tools: list | None = None, max_turns=10):
        self.agent = agent
        self.user_input = user_input
        self.memory = memory
        self.tools = tools
        self.max_turns = max_turns

    def __call__(self):
        tools = self.tools if self.tools is not None else self.agent.tools
        registry = {t.name: t.execute for t in tools}
        self.memory.push({"role": "user", "content": self.user_input})
        rr.log("agent/conversation", rr.TextLog(f"user: {self.user_input}"))

        for _ in range(self.max_turns):
            response = self.agent.act(self.memory.get_messages(), tools=tools)
            message = response.choices[0].message

            self.memory.push(message)

            if not message.tool_calls:
                rr.log("agent/conversation", rr.TextLog(f"assistant: {message.content[:200]}"))
                return message.content

            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                function_params = json.loads(tool_call.function.arguments)
                rr.log("agent/tool_calls", rr.TextLog(f"{function_name}({function_params})"))

                function_result = registry[function_name](**function_params)
                rr.log("agent/tool_results", rr.TextLog(f"{function_name} -> {function_result}"))

                self.memory.push({
                    "role": "tool",
                    "name": function_name,
                    "content": str(function_result),
                    "tool_call_id": tool_call.id,
                })

class FSM:
    class State:
        class LLMInteractions(StrEnum):
            UserMessage = auto()
            AssistantMessage = auto()
        class ToolInteractions(StrEnum):
            ToolCall = auto()
            ToolResult = auto()
        class FinalStates(StrEnum):
            Finished = auto()
        class UserInteractions(StrEnum):
            UserInputRequired = auto()
            UserResponse = auto()

    context_k: int | None = 3  # bounded window for FSM; PDA overrides to None

    def __init__(self, agent: Agent, user_input: str, memory: Memory, *, tools: list | None = None, max_turns=10):
        self.agent = agent
        self.memory = memory
        self.user_input = user_input
        self.tools = tools if tools is not None else agent.tools
        self.registry = {t.name: t.execute for t in self.tools}
        self.max_turns = max_turns
        self.last_message = None 
        self.result: str | None = None
        self.state = FSM.State.LLMInteractions.UserMessage

    def __call__(self):
        return self.loop()

    def transition(self):
        match self.state:

            case FSM.State.LLMInteractions.UserMessage:
                self.memory.stack.append(UserMessage(content=self.user_input))
                rr.log("agent/conversation", rr.TextLog(f"user: {self.user_input}"))
                self.state = FSM.State.LLMInteractions.AssistantMessage

            case FSM.State.LLMInteractions.AssistantMessage:
                response = self.agent.act(self.memory.render_context(k=self.context_k), tools=self.tools)
                self.last_message = response.choices[0].message
                self.memory.stack.append(AssistantMessage(
                    content=self.last_message.content,
                    tool_calls=self.last_message.tool_calls,
                ))

                if not self.last_message.tool_calls:
                    rr.log("agent/conversation", rr.TextLog(f"assistant: {self.last_message.content[:200]}"))
                    self.result = self.last_message.content
                    self.memory.stack.append(Finished(result=self.last_message.content))
                    self.state = FSM.State.FinalStates.Finished
                else:
                    self.state = FSM.State.ToolInteractions.ToolCall

            case FSM.State.ToolInteractions.ToolCall:
                for tool_call in self.last_message.tool_calls:  # ty:ignore[unresolved-attribute]
                    function_name = tool_call.function.name
                    function_params = json.loads(tool_call.function.arguments)
                    rr.log("agent/tool_calls", rr.TextLog(f"{function_name}({function_params})"))

                    self.memory.stack.append(ToolCall(name=function_name, args=function_params, tool_call_id=tool_call.id))
                    function_result = self.registry[function_name](**function_params)
                    rr.log("agent/tool_results", rr.TextLog(f"{function_name} -> {function_result}"))
                    self.memory.stack.append(ToolResult(
                        name=function_name,
                        output=str(function_result),
                        tool_call_id=tool_call.id,
                    ))
                self.state = FSM.State.ToolInteractions.ToolResult

            case FSM.State.ToolInteractions.ToolResult:
                self.state = FSM.State.LLMInteractions.AssistantMessage

    def loop(self):
        while not self.memory.stack or not isinstance(self.memory.stack[-1], Finished):
            self.transition()
        return self.result


# Context-Free Agent = Pushdown Automaton
# Stack-top driven transitions. The typed interaction entries ARE the stack alphabet.

class PDA(FSM):
    """Context-Free Agent, stack-top driven. Transitions match on isinstance(stack[-1], ...).
    The interaction stack is the pushdown store. Full history rendered for LLM context."""
    context_k = None

    def __init__(self, agent: Agent, user_input: str, memory: Memory, *, tools: list | None = None, max_turns=10):
        super().__init__(agent, user_input, memory, tools=tools, max_turns=max_turns)
        self.memory.stack.append(UserMessage(content=self.user_input))
        rr.log("agent/conversation", rr.TextLog(f"user: {self.user_input}"))

    def transition(self):
        top = self.memory.stack[-1]

        match top:
            case UserMessage() | ToolResult():
                response = self.agent.act(self.memory.render_context(), tools=self.tools)
                self.last_message = response.choices[0].message
                self.memory.stack.append(AssistantMessage(
                    content=self.last_message.content,
                    tool_calls=self.last_message.tool_calls,
                ))

            case AssistantMessage() if not top.tool_calls:
                rr.log("agent/conversation", rr.TextLog(f"assistant: {top.content[:200]}"))
                self.result = top.content
                self.memory.stack.append(Finished(result=top.content))

            case AssistantMessage():
                for tool_call in top.tool_calls:
                    function_name = tool_call.function.name
                    function_params = json.loads(tool_call.function.arguments)
                    rr.log("agent/tool_calls", rr.TextLog(f"{function_name}({function_params})"))

                    self.memory.stack.append(ToolCall(name=function_name, args=function_params, tool_call_id=tool_call.id))
                    function_result = self.registry[function_name](**function_params)
                    rr.log("agent/tool_results", rr.TextLog(f"{function_name} -> {function_result}"))
                    self.memory.stack.append(ToolResult(
                        name=function_name,
                        output=str(function_result),
                        tool_call_id=tool_call.id,
                    ))

    def loop(self):
        while not isinstance(self.memory.stack[-1], Finished):
            self.transition()
        return self.result


# LBA (Linear-Bounded Automaton)
# this would mean PDA + bounded read/write scratchpad (max 16 slots)
class LBA(PDA):
    MAX_ENTRIES = 16

    def __init__(self, agent, user_input, memory, *, tools=None, max_turns=10):
        pass

# TM (Turing Machine)
# gives us PDA + unbounded read/write scratchpad (no limit)
class TM(PDA):

    def __init__(self, agent, user_input, memory, *, tools=None, max_turns=10):
        pass
