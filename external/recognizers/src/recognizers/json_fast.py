from types import NoneType
from enum import StrEnum, auto

type JsonValue = JsonObject | JsonArray | str | int | float | bool | NoneType
type JsonObject = dict[str, JsonValue]
type JsonArray = list[JsonValue]

class TokenType(StrEnum): 
    STRING = auto()
    NUMBER = auto()
    BOOLEAN = auto()
    NULL = auto()

    LEFT_BRACE = auto()
    RIGHT_BRACE = auto()
    LEFT_BRACKET = auto()
    RIGHT_BRACKET = auto()
    COMMA = auto()
    COLON = auto()
    EOF = auto()

class Token:
    def __init__(self, token_type: TokenType, value: object):
        self.token_type = token_type
        self.value = value

class Scanner:
    token: list[Token]

    def __init__(self, source: str):
        self.source = source
        self.start = 0
        self.tokens = []
        self.current = 0
        self.line = 1

    def scan(self) -> list[Token]:
        while not self.is_at_end():
            self.start = self.current
            self.scan_token()

        self.tokens.append(Token((TokenType.EOF None)))
        return self.tokens

def is_at_end(self):
    self.current >= len(self.source)
    

    
def parse(input: str) -> JsonValue:
    pass

