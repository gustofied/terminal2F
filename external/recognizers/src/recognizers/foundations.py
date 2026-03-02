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

# match (enum) basically if/elif on state, just cleaner syntax

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
print(m.add().add().switch().add()) # this should give us that we are at blue state, and the colors are at values r=50 and blue=25

# match (class)

# now could be debated if its needed at all to use match cases, when often just if else do work, so we use this example how you could "complicate"
# I mean in this case, but make it more powerful for other cases when it is needed
# needs more work but very much interesting
# PyBeach 2025 - Brett Slatkin - Patterns and Anti-Patterns in Python's Structural Pattern Matching

@dataclass
class Red:
    value: int = 0

@dataclass
class Blue:
    value: int = 0

class Mixer2:
    def __init__(self):
        self.state = Red()

    def add(self, amount: int = 25) -> Self:
        match self.state:
            case Red(value=value) if value + amount > 255:
                self.state = Red(255)
            case Red(value=value):
                self.state = Red(value+ amount)
            case Blue(value=value) if value + amount > 255:
                self.state = Blue(255)
            case Blue(value=value):
                self.state = Blue(value+ amount)
        return self

    def switch(self) -> Self:
        match self.state:
            case Red():  self.state = Blue()
            case Blue(): self.state = Red()
        return self

    def reset(self) -> Self:
        self.state = Red()
        return self

    def __repr__(self):
        match self.state:
            case Red(value=value):  return f"[RED] r={value}"
            case Blue(value=value): return f"[BLUE] b={value}"

m2 = Mixer2()
print(m2.add().add().switch().add())

# transition table — fully table driven, no match anywhere
# 3 states cycling. adding a state = adding a row to each dict

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
print(tm.add().add().next().add().next().add())  # [BLUE] rgb(50,25,25)



# OO State Design Pattern

class Off:
    def flip(self, p): p.state = Red()
    def cycle(self, p): pass

class Red:
    def flip(self, p): p.state = Off()
    def cycle(self, p): p.state = Green()

class Green:
    def flip(self, p): p.state = Off()
    def cycle(self, p): p.state = Blue()

class Blue:
    def flip(self, p): p.state = Off()
    def cycle(self, p): p.state = Red()

class Pixel:
    def __init__(self): self.state = Off()
    def flip(self): self.state.flip(self)
    def cycle(self): self.state.cycle(self)

p = Pixel(); p.cycle(); print(type(p.state).__name__)  # Off
p.flip(); p.cycle(); print(type(p.state).__name__)     # Green


# coroutine




# ──────────────────────────────────────────
# From pattern to machine
# ──────────────────────────────────────────
# A machine: states, alphabet, δ (transition function), start, accept
# Key shift: the machine reads an input stream and decides acceptance.
# δ(current_state, input_symbol) → next_state. No one calls switch().

# DFA: accept binary strings ending with "01"

class Q(Enum):
    S = auto()   # start
    Z = auto()   # last symbol was 0
    A = auto()   # accept, last two were 01

def delta(state: Q, bit: str) -> Q:
    match (state, bit):
        case (Q.S, "0"): return Q.Z
        case (Q.S, "1"): return Q.S
        case (Q.Z, "0"): return Q.Z
        case (Q.Z, "1"): return Q.A
        case (Q.A, "0"): return Q.Z
        case (Q.A, "1"): return Q.S
    raise ValueError(f"unexpected: {bit}")

def accepts(s: str) -> bool:
    state = Q.S
    for ch in s:
        state = delta(state, ch)
    return state == Q.A

print(accepts("01"))      # True
print(accepts("101"))     # True
print(accepts("011"))     # False
print(accepts("1101"))    # True
print(accepts("1110"))    # False


# ──────────────────────────────────────────
# Old notes and experiments
# ──────────────────────────────────────────

# An FSM Class

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

# a gennie

def generator():
    yield 1
    yield 2
    yield "Hello World"

gen = generator()

# print((next(gen)))

# connie

def corutine():
      while True:
          value = yield
          print(f"given value {value}")

con = corutine()
next(con)
for x in range(10):
    con.send(x)



# Stacks
# Last In - First Out
browsing_session = []
browsing_session.append(1)
browsing_session.append(2)
browsing_session.append(3)
print(browsing_session)
browsing_session.pop(1)
print(browsing_session)


# class FSM:
#     class State():
#         class Readeing(Enum):
#             FAST = auto()
#             SLOW = auto()
#         class Speaking(Enum):
#             FAST = auto()
#             SLOW = auto()


print("- - - - - - - - - - - - -")

class Examply:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

Adam = Examply("Adam")

print(Adam)

class Examply(Examply):
    def __init__(self, name):
        super().__init__(name)
        self.namen = self.name + " Sioud"

    def __repr__(self):
        return self.namen

AdamSioud = Examply("Adam")

print(AdamSioud)


print("- - - - - - - - - - - - -")


print((lambda x, y:  x + y)(2, 3))

print("- - - - - - - - - - - - -")


dispatch = {
    "add": lambda a,b: a+b,
    "mul": lambda a,b: a*b,
}

result = dispatch["mul"](2, 3)  # 6
print(result)

print("- - - - - - - - - - - - -")



function_dispatched = {
    "timer": lambda c,d : c*d,
    "timy": lambda c,d : c*d,
}

print(function_dispatched["timer"](1,2))


print("- - - - - - - - - - - - -")

# Pattern Matching and Recursion
# https://raganwald.com/2018/10/17/recursive-pattern-matching.html

# -- Counter-based balance check --

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


# -- Matchers --
# A matcher takes text and returns the matched portion, or False if no match.
# This is the building block for parser combinators.

def literal(target: str, text: str) -> str | bool:
    """Match an exact string at the start of text."""
    return target if text.startswith(target) else False


# -- Combinators --
# Combinators compose matchers into larger matchers.

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


# -- Examples --

print("-- literal --")
match_parens = partial(literal, "()")
print(match_parens(text="())"))  # "()"

print("-- sequence --")
match_fubarfu = sequence(
    partial(literal, "fu"),
    partial(literal, "bar"),
    partial(literal, "fu"),
)
print(match_fubarfu(text="foobar"))     # False
print(match_fubarfu(text="fubar'd"))    # False
print(match_fubarfu(text="fubarfu'd"))  # "fubarfu"

print("-- longest --")
match_bad_news = longest(
    partial(literal, "fubar"),
    partial(literal, "snafu"),
)
print(match_bad_news("snafu'd"))  # "snafu"
print(match_bad_news("fubar'd"))  # "fubar"
print(match_bad_news("hello"))    # False


# -- Recursive balanced parentheses --
# Uses combinators to define a recursive grammar:
#   balanced -> "()" | "()" balanced | "(" balanced ")" | "(" balanced ")" balanced

def balanced(text: str):
    return longest(
        partial(literal, "()"),
        sequence(partial(literal, "()"), balanced),
        sequence(partial(literal, "("), balanced, partial(literal, ")")),
        sequence(partial(literal, "("), balanced, partial(literal, ")"), balanced),
    )(text)

print("-- balanced --")
print(balanced("(())("))      # False
print(balanced("(()())()"))   # "(()())()"
print(balanced("())"))        # "()"
print(balanced("xyz"))        # False


# A brutal look at balanced parntheses, computing machines and pushdown automata
# https://raganwald.com/2019/02/14/i-love-programming-and-programmers.html


# DFA Deterministic Finite Automaton

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

print("-- -- ")
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


print("-- -- ")
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


print("-- -- ")
test(BalancedParentheses, [
    '', '(', '()', '()()', '{()}',
    '([()()]())', '([()())())',
    '())()', '((())(())'
])
