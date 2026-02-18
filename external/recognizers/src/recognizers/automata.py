from enum import Enum, StrEnum, auto
from functools import partial

# FSM using Enum and Match
# Enums for fixed states/actions, match for transitions, function returns the next state

class State(Enum):
    LOCKED = 1
    UNLOCKED = 2
    ALARM = 3

class Action(Enum):
    PUSH = 1
    COIN = 2
    RESET = 3

def handle_action(state: State, action: Action) -> State:
    match state:
        case State.LOCKED:
            if action == Action.COIN:
                return State.UNLOCKED
            if action == Action.PUSH:
                print("door is locked, needs coin")

        case State.UNLOCKED:
            if action == Action.PUSH:
                print("door opens, then locks")
                return State.LOCKED
            if action == Action.RESET:
                return State.LOCKED

        case State.ALARM:
            if action == Action.RESET:
                return State.LOCKED
    return state

handle_action(state=State.LOCKED, action=Action.PUSH)

door = State.LOCKED
door = handle_action(door, Action.COIN)
print(door)


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

# coroutine FSM (dispatch table + coroutine)

def fsm_coroutine(transitions, state):
    while True:
        action = yield state
        state = transitions.get((state, action), state)

transitions = {
    (State.LOCKED, Action.COIN): State.UNLOCKED,
    (State.UNLOCKED, Action.PUSH): State.LOCKED,
    (State.ALARM, Action.RESET): State.LOCKED,
}

machine = fsm_coroutine(transitions, State.LOCKED)
next(machine)
print(machine.send(Action.COIN))
print(machine.send(Action.PUSH))


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
