from __future__ import annotations
from typing import Self, Generator
from enum import Enum, StrEnum, auto
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial

# Coding Things Like States

# meh example, match very much if-else esque, then we look at dispatch also, a little cleaner

class Color(Enum):
    RED = auto()
    BLUE = auto()

# match (enum), in article

class Mixer:
    def __init__(self):
        self.state = Color.RED
        self.red = 0
        self.blue = 0

    def add(self, amount: int = 25) -> Self:
        match self.state:
            case Color.RED:  
                self.red  = min(255, self.red + amount)
            case Color.BLUE: 
                self.blue = min(255, self.blue + amount)
        return self # so we can chain

    def switch(self) -> Self:
        self.state = Color.BLUE if self.state == Color.RED else Color.RED # Tenary
        return self

    def reset(self) -> Self:
        self.state = Color.RED
        self.red = 0
        self.blue = 0
        
        return self

    def __repr__(self):
        return f"[{self.state.name}] r={self.red} b={self.blue}"

m = Mixer()
print("recognizers-- match case --recognizers")
print(m.add().add().switch().add()) # this should give us that we are at blue state, and the colors are at values r=50 and blue=25

# match (class), not in article

# now could be debated if its needed at all to use match cases, when often just if else do work, so we use this example how you could "complicate"
# I mean in this case, but make it more powerful for other cases when it is needed
# needs more work but very much interesting
# PyBeach 2025 - Brett Slatkin - Patterns and Anti-Patterns in Python's Structural Pattern Matching

# @dataclass
# class Red:
#     value: int = 0

# @dataclass
# class Blue:
#     value: int = 0

# class Mixer2:
#     def __init__(self):
#         self.state = Red()

#     def add(self, amount: int = 25) -> Self:
#         match self.state:
#             case Red(value=value) if value + amount > 255:
#                 self.state = Red(255)
#             case Red(value=value):
#                 self.state = Red(value+ amount)
#             case Blue(value=value) if value + amount > 255:
#                 self.state = Blue(255)
#             case Blue(value=value):
#                 self.state = Blue(value+ amount)
#         return self

#     def switch(self) -> Self:
#         match self.state:
#             case Red():  self.state = Blue()
#             case Blue(): self.state = Red()
#         return self

#     def reset(self) -> Self:
#         self.state = Red()
#         return self

#     def __repr__(self):
#         match self.state:
#             case Red(value=value):  return f"[RED] r={value}"
#             case Blue(value=value): return f"[BLUE] b={value}"

# m2 = Mixer2()
# print(m2.add().add().switch().add())


# transition table, in article

class Phase(Enum):
    RED = auto()
    GREEN = auto()
    BLUE = auto()

phase_next = {
    Phase.RED:   Phase.GREEN,
    Phase.GREEN: Phase.BLUE,
    Phase.BLUE:  Phase.RED,
}

phase_attr = {
    Phase.RED:   "red",
    Phase.GREEN: "green",
    Phase.BLUE:  "blue",
}

class Mixer3:
    def __init__(self):
        self.phase = Phase.RED
        self.red = self.green = self.blue = 0

    def add(self, amount: int = 25) -> Self:
        attr = phase_attr[self.phase]
        setattr(self, attr, min(255, getattr(self, attr) + amount))
        return self

    def next(self) -> Self:
        self.phase = phase_next[self.phase]
        return self

    def reset(self) -> Self:
        self.phase = Phase.RED
        self.red = self.green = self.blue = 0
        return self

    def __repr__(self):
        return f"[{self.phase.name}] rgb({self.red},{self.green},{self.blue})"

tm = Mixer3()
print("recognizers-- dispatch table --recognizers")
print(tm.add().add().next().add().next().add())


# OO State Design Pattern

class Red:
    def red(self, p, a):   p.r = min(255, p.r + a)
    def green(self, p, a): p.g = min(255, p.g + a); p.state = Green()
    def blue(self, p, a):  raise ValueError("add green first")

class Green:
    def red(self, p, a):   raise ValueError("red phase done")
    def green(self, p, a): p.g = min(255, p.g + a)
    def blue(self, p, a):  p.b = min(255, p.b + a); p.state = Blue()

class Blue:
    def red(self, p, a):   raise ValueError("red phase done")
    def green(self, p, a): raise ValueError("green phase done")
    def blue(self, p, a):  p.b = min(255, p.b + a)

class Palette:
    def __init__(self): self.r = self.g = self.b = 0; self.state = Red()
    def red(self, a=25): self.state.red(self, a); return self
    def green(self, a=25): self.state.green(self, a); return self
    def blue(self, a=25): self.state.blue(self, a); return self

p = Palette().red(50).red(20).green(30).blue(10)
print("recognizers-- state design pattern --recognizers")
print(p.r, p.g, p.b)


# coroutine, in the article

def mix_protocol() -> Generator[str, int, None]:
    red = green = blue = 0

    amount = yield "phase: RED"
    while True:
        if amount == 0: break
        red = min(255, red + amount)
        amount = yield f"[RED] rgb({red},{green},{blue})"

    amount = yield "phase: GREEN"
    while True:
        if amount == 0: break
        green = min(255, green + amount)
        amount = yield f"[GREEN] rgb({red},{green},{blue})"

    amount = yield "phase: BLUE"
    while True:
        if amount == 0: break
        blue = min(255, blue + amount)
        amount = yield f"[BLUE] rgb({red},{green},{blue})"

    yield f"done: rgb({red},{green},{blue})"

protocol = mix_protocol()

print("recognizers-- coroutines --recognizers")
next(protocol)

print(protocol.send(50)) 
print(protocol.send(25))   
print(protocol.send(0))   
print(protocol.send(100))  
print(protocol.send(0))  
print(protocol.send(30))    
print(protocol.send(0)) 

def pour_stream(amounts: list[int]) -> str:
    protocol = mix_protocol()
    next(protocol)
    result = ""
    for amount in amounts:
        result = protocol.send(amount)
    return result

print(pour_stream([50, 25, 0, 100, 0, 30, 0]))
print(pour_stream([10, 20, 30, 0, 50, 0, 80, 0]))


# An FSM Class, not in article

class FSM:
    """This is a Finite State Machine (FSM)"""
    def __init__(self, initial_state, memory=None):
        pass

    def reset(self):
        pass

    def add_transition(self, input_symbol, state, action=None, next_state=None):
        pass

    def add_transition_any(self, state, action=None, next_state=None):
        pass

    def set_default_transition(self, action, next_state):
        pass

    def get_transition(self, input_symbol, state):
        pass

    def process(self, input_symbol):
        pass

    def process_list(self, input_symbols):
        pass


# insporation inspir;)
# Pattern Matching and Recursion
# https://raganwald.com/2018/10/17/recursive-pattern-matching.html

def is_balanced(text: str) -> bool:
    """Check if parentheses are balanced using a simple counter."""
    depth = 0
    for char in text:
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
        if depth < 0:
            return False
    return depth == 0




def literal(target: str, text: str) -> str | bool:
    """Match an exact string at the start of text."""
    return target if text.startswith(target) else False


def sequence(*matchers):
    """Match matchers one after another. All must succeed in order."""
    def match(text: str):
        remaining = text
        matched_parts = []
        for matcher in matchers:
            result = matcher(text=remaining)
            if result is False:
                return False
            matched_parts.append(result)
            remaining = remaining[len(result):]
        return "".join(matched_parts)
    return match


def longest(*matchers):
    """Try all matchers on the same input, return the longest match."""
    def match(text: str):
        matches = []
        for matcher in matchers:
            result = matcher(text=text)
            if result is not False:
                matches.append(result)
        if not matches:
            return False
        return max(matches, key=len)
    return match


print("recognizers--  literal --recognizers")
match_parens = partial(literal, "()")
print(match_parens(text="())"))  # "()"

print("recognizers-- sequence --recognizers")
match_fubarfu = sequence(
    partial(literal, "fu"),
    partial(literal, "bar"),
    partial(literal, "fu"),
)
print(match_fubarfu(text="foobar"))    
print(match_fubarfu(text="fubar'd"))   
print(match_fubarfu(text="fubarfu'd"))  

print("recognizers-- longest --recognizers")
match_bad_news = longest(
    partial(literal, "fubar"),
    partial(literal, "snafu"),
)
print(match_bad_news("snafu'd")) 
print(match_bad_news("fubar'd")) 
print(match_bad_news("hello"))    


def balanced(text: str):
    return longest(
        partial(literal, "()"),
        sequence(partial(literal, "()"), balanced),
        sequence(partial(literal, "("), balanced, partial(literal, ")")),
        sequence(partial(literal, "("), balanced, partial(literal, ")"), balanced),
    )(text)

print("recognizers-- balanced --recognizers")
print(balanced("(())("))   
print(balanced("(()())()"))  
print(balanced("())"))    
print(balanced("xyz"))       

# inspo from
# A brutal look at balanced parntheses, computing machines and pushdown automata 
# https://raganwald.com/2019/02/14/i-love-programming-and-programmers.html
# DFA Deterministic Finite Automaton, in article — the core machine

class DFAStates(StrEnum):
    START = auto()
    END = auto()

class DeterministicFiniteAutomaton:

    def __init__(self, internal=DFAStates.START, halted=False, recognized=False):
        self.internal = internal
        self.halted = halted
        self.recognized = recognized

    def transitionTo(self, internal):
        self.internal = internal
        return self

    def recognize(self):
        self.recognized = True
        return self

    def halt(self):
        self.halted = True
        return self

    def consume(self, token):
        return getattr(self, self.internal)(token)


    @classmethod
    def evaluate(cls, input_string):
        current = cls()
        for char in input_string:
            current = current.consume(char)

            if current is None or current.halted:
                return False

            if current.recognized:
                return True

        current = current.consume(DFAStates.END)
        return current is not None and current.recognized


# Recognizer

class Raginald(DeterministicFiniteAutomaton):
    def start(self, token):
        if token == 'R':
            return self.transitionTo('R')

    def R(self, token):
        if token == 'e':
            return self.transitionTo('Re')

    def Re(self, token):
        if token == 'g':
            return self.transitionTo('Reg')

    def Reg(self, token):
        if token == 'g':
            return self.transitionTo('Regg')
        if token == DFAStates.END:
            return self.recognize()

    def Regg(self, token):
        if token == 'i':
            return self.transitionTo('Reggi')

    def Reggi(self, token):
        if token == 'e':
            return self.transitionTo('Reggie')

    def Reggie(self, token):
        if token == DFAStates.END:
            return self.recognize()


def test(recognizer, examples):
    for example in examples:
        print(f"'{example}' => {recognizer.evaluate(example)}")

print("recognizers-- Raginald DFA --recognizers")
test(Raginald, ["Reg", "Reggie", "Re", "Hello"])

class Binary(DeterministicFiniteAutomaton):
    def start(self, token):
        if token == '0':
            return self.transitionTo('zero')
        elif token == '1':
            return self.transitionTo('oneOrMore')

    def zero(self, token):
        if token == DFAStates.END:
            return self.recognize()

    def oneOrMore(self, token):
        if token == '0':
            return self.transitionTo('oneOrMore')
        elif token == '1':
            return self.transitionTo('oneOrMore')
        elif token == DFAStates.END:
            return self.recognize()


print("recognizers-- Binary DFA --recognizers")
test(Binary, [
    '', '0', '1', '00', '01', '10', '11',
    '000', '001', '010', '011', '100',
    '101', '110', '111',
    '10100011011000001010011100101110111'
])


# Deterministic Pushdown Automaton

class DeterministicPushdownAutomaton:
    def __init__(self, internal='start', external=None):
        self.internal = internal
        self.external = external if external is not None else []
        self.halted = False
        self.recognized = False

    def push(self, token):
        self.external.append(token)
        return self

    def pop(self):
        self.external.pop()
        return self

    def replace(self, token):
        self.external[-1] = token
        return self

    def top(self):
        return self.external[-1] if self.external else None

    def hasEmptyStack(self):
        return len(self.external) == 0

    def transitionTo(self, internal):
        self.internal = internal
        return self

    def recognize(self):
        self.recognized = True
        return self

    def halt(self):
        self.halted = True
        return self

    def consume(self, token):
        return getattr(self, self.internal)(token)

    @classmethod
    def evaluate(cls, input_string):
        current = cls()
        for char in input_string:
            current = current.consume(char)
            if current is None or current.halted:
                return False
            if current.recognized:
                return True
        current = current.consume(DFAStates.END)
        return current is not None and current.recognized


class BalancedParentheses(DeterministicPushdownAutomaton):
    def start(self, token):
        if token == '(':
            return self.push(token)
        elif token == '[':
            return self.push(token)
        elif token == '{':
            return self.push(token)
        elif token == ')' and self.top() == '(':
            return self.pop()
        elif token == ']' and self.top() == '[':
            return self.pop()
        elif token == '}' and self.top() == '{':
            return self.pop()
        elif token == DFAStates.END and self.hasEmptyStack():
            return self.recognize()


print("recognizers-- BalancedParentheses DPDA --recognizers")
test(BalancedParentheses, [
    '', '(', '()', '()()', '{()}',
    '([()()]())', '([()())())',
    '())()', '((())(())'
])
