from enum import Enum
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

# Balanced https://raganwald.com/2018/10/17/recursive-pattern-matching.html
# Pattern Matching and Recurssion


def balanced(input: str):
    openParenthesesCount: int = 0
    closeParenthesesCount: int = 0

    for x in range(0, len(input)):
        c = input[x]

        if c == '(':
            openParenthesesCount += 1
        
        elif c == ')':
            closeParenthesesCount += 1

        if closeParenthesesCount > openParenthesesCount:
            return False

    return closeParenthesesCount == openParenthesesCount


def just(target, text):
    return target if text.startswith(target) else False

caseA = partial(just, target="()")
a = caseA(text="s())")
print(a)

print("- - - - ")

def follows(*patterns):
    def combined(text):
        remaining = text
        matched_parts = []

        for pattern in patterns:
            matched = pattern(remaining)
            if matched is False:
                return False

            matched_parts.append(matched)
            remaining = remaining[len(matched):]

        return "".join(matched_parts)
    return combined


p = follows(partial(just, "fu"), partial(just, "bar"), partial(just, "fu"))
print(p("foobar"))   
print(p("fubar'd"))   
print(p("fubarfu'd"))  

def cases(*patterns):
    def combined(text):
        successful_matches = []

        # Try every pattern on the same input text
        for pattern in patterns:
            matched = pattern(text)

            # Our convention: False means "no match"
            if matched is not False:
                successful_matches.append(matched)

        # If nothing matched, the whole cases() fails
        if len(successful_matches) == 0:
            return False

        # Choose the longest match (by number of characters)
        longest_match = successful_matches[0]
        for matched in successful_matches[1:]:
            if len(matched) > len(longest_match):
                longest_match = matched

        return longest_match

    return combined


print("- - -")

badNews = cases(
    partial(just, "fubar"),
    partial(just, "snafu")
)

print(badNews("snafu'd"))  
print(badNews("fubar'd"))   
print(badNews("hello"))  




def balanced(text):
    return cases(
        partial(just, "()"),                      
        follows(partial(just, "()"), balanced),       
        follows(partial(just, "("), balanced, partial(just, ")")),                
        follows(partial(just, "("), balanced, partial(just, ")"), balanced),    
    )(text)

print("- - - ")

print(balanced("(())("))     
print(balanced("(()())()"))  
print(balanced("())"))         
print(balanced("xyz"))      

print("- - - ")

# https://raganwald.com/2019/02/14/i-love-programming-and-programmers.html






