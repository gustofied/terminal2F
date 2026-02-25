# JSON Parsing - Overview

## Why Hand-Write a JSON Parser?

Every language has `json.loads()`. Writing one by hand teaches you:

- How scanners (lexers) break text into tokens
- How recursive descent works - the most common parsing technique
- How grammars map to code structure
- The difference between recognizing and parsing

JSON is the perfect target. The grammar is small (6 value types, 9 token types) but exercises every important concept: recursion, string escaping, number formats, whitespace handling.

## The JSON Grammar

```
value   -> object | array | string | number | "true" | "false" | "null"
object  -> "{" "}" | "{" members "}"
members -> pair | pair "," members
pair    -> string ":" value
array   -> "[" "]" | "[" elements "]"
elements-> value | value "," elements
```

Six productions. Each one maps to a function.

## Two Approaches

### 1. Scanner + Parser (json_fast.py)

Split parsing into two passes:

**Pass 1 - Scanning (lexing):** walk the raw string and emit tokens.

```
{"name": "Adam", "age": 25}
  -> LEFT_BRACE STRING("name") COLON STRING("Adam")
     COMMA STRING("age") COLON NUMBER(25) RIGHT_BRACE
```

Token types: `STRING`, `NUMBER`, `BOOLEAN`, `NULL`, `LEFT_BRACE`, `RIGHT_BRACE`, `LEFT_BRACKET`, `RIGHT_BRACKET`, `COMMA`, `COLON`, `EOF`.

**Pass 2 - Parsing:** walk the token list and build the value tree. The parser never sees whitespace or raw characters - just clean tokens.

Advantages:

- Clean separation of concerns
- Parser logic is simple - no whitespace or escape handling
- Easy to add good error messages with token positions

### 2. Recursive Descent (json_parser_recursive_descent.py)

Single pass. Each grammar rule becomes a function. Functions call each other recursively.

```python
def parse_value(s, i):
    i = skip_ws(s, i)
    if s[i] == '"':   return parse_string(s, i)
    if s[i] == '{':   return parse_object(s, i)
    if s[i] == '[':   return parse_array(s, i)
    if s[i] == 't':   return parse_true(s, i)
    if s[i] == 'f':   return parse_false(s, i)
    if s[i] == 'n':   return parse_null(s, i)
    return parse_number(s, i)
```

Every function takes the source string and current position, returns the parsed value and new position. No intermediate token list.

Advantages:

- Minimal code, no intermediate data structures
- Fast - single pass, no allocation for tokens
- Direct mapping from grammar to code

## Type Definitions

Both approaches produce the same output types:

```python
type JsonValue = (
    JsonObject | JsonArray | str
    | int | float | bool | NoneType
)
type JsonObject = dict[str, JsonValue]
type JsonArray = list[JsonValue]
```

## Status

Both parsers are in progress. The scanner in `json_fast.py` has the token types defined and a skeleton `Scanner` class. The recursive descent parser in `json_parser_recursive_descent.py` has a whitespace-skipping utility.

Next steps:

- Complete the scanner: character dispatch, string/number scanning, error reporting
- Complete the recursive descent parser: all six parse functions
- Test both against the same JSON fixtures in `data/`
- Compare: lines of code, clarity, error messages, performance
