# Automata

## What is a Recognizer?

A recognizer consumes input one token at a time and answers a yes/no question: does this string belong to the language? No output, no transformation - just accept or reject.

Every recognizer has three things:

- **Internal state** - where the machine is right now
- **Transitions** - rules that say "in state X, if you see token Y, go to state Z"
- **Acceptance condition** - which states count as "yes"

## DFA - Deterministic Finite Automaton

The simplest recognizer. A fixed set of states, a fixed set of transitions, no memory beyond which state you're in.

The `DeterministicFiniteAutomaton` base class works by dispatching on the current state name. Each state is a method. The method receives a token and returns `self` after transitioning, or `None` to reject.

```python
class DeterministicFiniteAutomaton:
    def consume(self, token):
        return getattr(self, self.internal)(token)
```

### Raginald - a string recognizer

Recognizes "Reg" and "Reggie". Each character advances to the next state. Two acceptance points: after "Reg" and after "Reggie".

```
start -> R -> Re -> Reg (accept) -> Regg -> Reggi -> Reggie (accept)
```

The key insight: `Reg` accepts on `END` but also transitions to `Regg` on `g`. This makes it recognize both "Reg" and "Reggie" with a single path.

### Binary - a number recognizer

Recognizes valid binary numbers: "0", "1", "10", "101", etc. Two states after start:

- `zero` - only accepts "0" alone (no leading zeros)
- `oneOrMore` - loops on 0 or 1, accepts on END

### What DFAs cannot do

DFAs have no memory. They can't count. They can't match brackets. They can't recognize `a^n b^n` (n a's followed by n b's) because that requires remembering how many a's you saw. For that, you need a stack.

## DPDA - Deterministic Pushdown Automaton

A DFA plus a stack. The stack gives you one level of memory - you can push, pop, peek, and check if it's empty.

```python
class DeterministicPushdownAutomaton:
    def __init__(self):
        self.internal = 'start'
        self.external = []  # the stack
```

### BalancedParentheses

The classic DPDA example. Handles `()`, `[]`, and `{}` - including nesting. The algorithm:

- See an opener? Push it.
- See a closer? Check the top of stack matches. If yes, pop. If no, reject.
- Reach END? Accept only if the stack is empty.

```
Input: ([()()]())
Stack: [ ( [ ( () ] ( () ] () ] ) ] (empty) -> accept
```

Single state, all the logic lives in the stack. This is the power of the pushdown automaton - it can count and match nested structures.

## The Chomsky Hierarchy

These machines map to the Chomsky hierarchy of formal languages:

| Machine | Language class | Example |
|---------|---------------|---------|
| DFA | Regular | "Reggie", binary numbers |
| DPDA | Context-free | balanced brackets, arithmetic |
| LBA | Context-sensitive | `a^n b^n c^n` |
| TM | Recursively enumerable | anything computable |

Each level strictly includes the one below. terminal2f's runner strategies (LOOP, FSM, PDA, LBA, TM) mirror this hierarchy - each gives the agent more computational power.
