# Automata / Runners
# Each class is both an automaton (computational model) and a runner (agent execution loop).
# LOOP — basic chat loop, no structured state
# FSM  — finite state machine, bounded context window (k=3)
# PDA  — pushdown automaton, stack-top driven, full history
# LBA  — linear-bounded automaton, PDA + bounded scratchpad
# TM   — turing machine, PDA + unbounded scratchpad
# All are callable: runner(agent, input, memory, tools=...)() → result

from __future__ import annotations

from enum import StrEnum, auto

import json
import rerun as rr

from terminal2f.agent import Agent
from terminal2f.memory import Memory
from terminal2f.tools import WriteArtifact, ReadArtifact
from terminal2f.states import (
    UserMessage, AssistantMessage, ToolCall, ToolResult,
    AgentCall, AgentResult, Finished, render_context,
)


class FSM:
    # very much event types like in microservies
    class State:
        class LLMInteractions(StrEnum):
        # LLM interactions - using an LLM to choose the next action
            UserMessage = auto() # UserMessage is the state that tells the state machine to execute an LLM call
            AssistantMessage = auto() # AssistantMessage captures the output of an LLM via a UserMessage.
        class AgentInteractions(StrEnum):
            AgentCall = auto() # This state instructs the state machine to send a message to another agent
            AgentResult = auto() # AgentResult captures the reply from the other agent to the calling agent
        class ToolInteractions(StrEnum):
            ToolCall = auto() # ToolCall is the state that tells the state machine to execute a tool
            ToolResult = auto() #ToolResult is the state that captures the output of a ToolCall
        class FinalStates(StrEnum):
            Finished = auto() # Finished captures the final state of the state machine - this happens when the agent has finished it task
        class UserInteractions(StrEnum):
            UserInputRequired = auto() # UserInputRequired tells the state machine to wait for a user input
            UserResponse = auto() # UserResponse captures user inputs - either as an initial state or as a response

    context_k: int | None = 3  # bounded window for FSM; PDA overrides to None

    def __init__(self, agent: Agent, user_input: str, memory: Memory, *, tools: list | None = None, max_turns=10):
        self.agent = agent
        self.memory = memory
        self.user_input = user_input
        self.tools = tools if tools is not None else agent.tools
        self.registry = {t.name: t.execute for t in self.tools}
        self.max_turns = max_turns
        self.last_message = None  # LLM response, needs to survive between states
        self.pending_agent_call = None  # stashed delegate call for AgentCall state
        self.result = None  # the final answer when we hit Finished
        self.state = FSM.State.LLMInteractions.UserMessage

    def __call__(self):
        return self.loop()

    def transition(self):
        match self.state:

            # User gave input → push it to stack, move to LLM
            case FSM.State.LLMInteractions.UserMessage:
                self.memory.stack.append(UserMessage(content=self.user_input))
                rr.log("agent/conversation", rr.TextLog(f"user: {self.user_input}"))
                self.state = FSM.State.LLMInteractions.AssistantMessage

            case FSM.State.LLMInteractions.AssistantMessage:
                response = self.agent.act(render_context(self.memory.stack, k=self.context_k), tools=self.tools)
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

            # Execute tools — delegate calls get stashed and routed to AgentCall
            case FSM.State.ToolInteractions.ToolCall:
                for tool_call in self.last_message.tool_calls:  # ty:ignore[possibly-missing-attribute]
                    function_name = tool_call.function.name
                    function_params = json.loads(tool_call.function.arguments)
                    rr.log("agent/tool_calls", rr.TextLog(f"{function_name}({function_params})"))

                    if function_name == "delegate":
                        # Stash and route to AgentCall state
                        self.pending_agent_call = function_params
                        self.state = FSM.State.AgentInteractions.AgentCall
                    else:
                        self.memory.stack.append(ToolCall(name=function_name, args=function_params, tool_call_id=tool_call.id))
                        function_result = self.registry[function_name](**function_params)
                        rr.log("agent/tool_results", rr.TextLog(f"{function_name} -> {function_result}"))
                        self.memory.stack.append(ToolResult(
                            name=function_name,
                            output=str(function_result),
                            tool_call_id=tool_call.id,
                        ))
                # If no delegate was found, go to ToolResult
                if self.state == FSM.State.ToolInteractions.ToolCall:
                    self.state = FSM.State.ToolInteractions.ToolResult

            case FSM.State.ToolInteractions.ToolResult:
                self.state = FSM.State.LLMInteractions.AssistantMessage

            # Sub-agent: spawn, run its own FSM, collect result
            case FSM.State.AgentInteractions.AgentCall:
                instruction = self.pending_agent_call.get("instruction", "")  # ty:ignore[possibly-missing-attribute]
                self.memory.stack.append(AgentCall(agent_name="sub", instruction=instruction))
                rr.log("agent/agent_call", rr.TextLog(f"delegate({instruction[:200]})"))
                function_result = self.registry["delegate"](**self.pending_agent_call)  # ty:ignore[invalid-argument-type]
                rr.log("agent/agent_result", rr.TextLog(f"delegate -> {str(function_result)[:200]}"))
                self.memory.stack.append(AgentResult(agent_name="sub", result=str(function_result)))
                self.pending_agent_call = None
                self.state = FSM.State.AgentInteractions.AgentResult

            case FSM.State.AgentInteractions.AgentResult:
                self.state = FSM.State.LLMInteractions.AssistantMessage

    def loop(self):
        while not self.memory.stack or not isinstance(self.memory.stack[-1], Finished):
            self.transition()
        return self.result

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
                function_name = tool_call.function.name # The function name to call
                function_params = json.loads(tool_call.function.arguments) # The function arguments
                rr.log("agent/tool_calls", rr.TextLog(f"{function_name}({function_params})"))

                function_result = registry[function_name](**function_params) # The function result
                rr.log("agent/tool_results", rr.TextLog(f"{function_name} -> {function_result}"))

                self.memory.push({
                    "role": "tool",
                    "name": function_name,
                    "content": str(function_result),
                    "tool_call_id": tool_call.id,
                })

# Context-Free Agent ≃ Pushdown Automaton
# δ: S × Σ × Z → S × Z*
# Stack-top driven transitions. The typed interaction entries ARE the stack alphabet.
# No self.state — the stack top determines the next transition.
# Push to advance, Finished on top to stop.
class PDA(FSM):
    """Context-Free Agent — stack-top driven. Transitions match on isinstance(stack[-1], ...).
    The interaction stack is the pushdown store. Full history rendered for LLM context."""
    context_k = None

    def __init__(self, agent: Agent, user_input: str, memory: Memory, *, tools: list | None = None, max_turns=10):
        super().__init__(agent, user_input, memory, tools=tools, max_turns=max_turns)
        # Seed the stack — stack-top drives everything from here
        self.memory.stack.append(UserMessage(content=self.user_input))
        rr.log("agent/conversation", rr.TextLog(f"user: {self.user_input}"))

    def transition(self):
        top = self.memory.stack[-1]

        match top:
            case UserMessage() | ToolResult() | AgentResult():
                # Any of these on top → call the LLM
                response = self.agent.act(render_context(self.memory.stack), tools=self.tools)
                self.last_message = response.choices[0].message
                self.memory.stack.append(AssistantMessage(
                    content=self.last_message.content,
                    tool_calls=self.last_message.tool_calls,
                ))

            case AssistantMessage() if not top.tool_calls:
                # Assistant with no tool calls → done
                rr.log("agent/conversation", rr.TextLog(f"assistant: {top.content[:200]}"))
                self.result = top.content
                self.memory.stack.append(Finished(result=top.content))

            case AssistantMessage():
                # Assistant with tool calls → execute them, push results
                for tool_call in top.tool_calls:
                    function_name = tool_call.function.name
                    function_params = json.loads(tool_call.function.arguments)
                    rr.log("agent/tool_calls", rr.TextLog(f"{function_name}({function_params})"))

                    if function_name == "delegate":
                        self.memory.stack.append(AgentCall(agent_name="sub", instruction=function_params.get("instruction", "")))
                        function_result = self.registry[function_name](**function_params)
                        rr.log("agent/agent_result", rr.TextLog(f"delegate -> {str(function_result)[:200]}"))
                        self.memory.stack.append(AgentResult(agent_name="sub", result=str(function_result)))
                    else:
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


# LBA — bounded random-access scratchpad on top of PDA
class LBA(PDA):
    MAX_ENTRIES = 16  # the bound — this is what makes it linear-bounded

    def __init__(self, agent: Agent, user_input: str, memory: Memory, *, tools: list | None = None, max_turns=10):
        scratchpad_tools = [
            WriteArtifact(store=memory.object_store, max_entries=self.MAX_ENTRIES),
            ReadArtifact(store=memory.object_store),
        ]
        all_tools = (tools if tools is not None else agent.tools) + scratchpad_tools
        super().__init__(agent, user_input, memory, tools=all_tools, max_turns=max_turns)

# TM — unbounded read/write, no cap on object_store
class TM(PDA):

    def __init__(self, agent: Agent, user_input: str, memory: Memory, *, tools: list | None = None, max_turns=10):
        scratchpad_tools = [
            WriteArtifact(store=memory.object_store),
            ReadArtifact(store=memory.object_store),
        ]
        all_tools = (tools if tools is not None else agent.tools) + scratchpad_tools
        super().__init__(agent, user_input, memory, tools=all_tools, max_turns=max_turns)
