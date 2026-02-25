# Parser Combinators

## From Recognizers to Parsers

A recognizer says yes or no. A parser says yes or no **and** returns what it matched. Parser combinators build complex parsers by composing small ones.

The building block is a **matcher**: a function that takes text and returns the matched portion, or `False`.

```python
def literal(target: str, text: str) -> str | bool:
    return target if text.startswith(target) else False
```

## Combinators

Combinators are higher-order functions. They take matchers and return new matchers.

### sequence

Runs matchers one after another. All must succeed. Each picks up where the last left off.

```python
match_fubarfu = sequence(
    partial(literal, "fu"),
    partial(literal, "bar"),
    partial(literal, "fu"),
)
match_fubarfu("fubarfu'd")  # "fubarfu"
match_fubarfu("foobar")     # False
```

### longest

Tries all matchers on the same input, returns the longest match. This is ordered choice with a greedy strategy.

```python
match_bad_news = longest(
    partial(literal, "fubar"),
    partial(literal, "snafu"),
)
match_bad_news("snafu'd")  # "snafu"
```

## Recursive Grammars

Combinators become powerful when you add recursion. A matcher can reference itself, defining recursive grammars.

### Balanced parentheses via combinators

The grammar:

```
balanced -> "()"
         | "()" balanced
         | "(" balanced ")"
         | "(" balanced ")" balanced
```

As a combinator:

```python
def balanced(text: str):
    return longest(
        partial(literal, "()"),
        sequence(partial(literal, "()"), balanced),
        sequence(partial(literal, "("), balanced, partial(literal, ")")),
        sequence(partial(literal, "("), balanced, partial(literal, ")"), balanced),
    )(text)
```

This does the same job as the DPDA `BalancedParentheses` class but through function composition instead of explicit state and stack management. Trade-offs:

- **Combinators**: declarative, mirrors the grammar directly, easy to extend
- **Automata**: explicit control, efficient, better for streaming input

## Connection to JSON Parsing

JSON is a context-free grammar. It can be parsed with combinators or with a recursive descent parser (which is essentially hand-written combinators). The two JSON parser files in the recognizers module explore both:

- `json_fast.py` - scanner + token-based parser (lexer-first approach)
- `json_parser_recursive_descent.py` - character-by-character recursive descent
