"""Tokenizer (lexer) for arithmetic expressions."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Union


class TokenType(Enum):
    NUMBER = auto()
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    POWER = auto()
    LPAREN = auto()
    RPAREN = auto()
    EOF = auto()


@dataclass
class Token:
    type: TokenType
    value: Union[float, str]
    position: int


class Tokenizer:
    """Lexer for arithmetic expressions."""

    def __init__(self, text: str):
        self.text = text.replace(" ", "")
        self.pos = 0
        self.current_char = self.text[0] if self.text else None

    def advance(self):
        """Move to next character."""
        self.pos += 1
        self.current_char = self.text[self.pos] if self.pos < len(self.text) else None

    def peek(self, offset: int = 1) -> str:
        """Look ahead without consuming."""
        peek_pos = self.pos + offset
        return self.text[peek_pos] if peek_pos < len(self.text) else ""

    def number(self) -> Token:
        """Parse a numeric literal."""
        start_pos = self.pos
        result = ""

        while self.current_char and (
            self.current_char.isdigit() or self.current_char == "."
        ):
            result += self.current_char
            self.advance()

        return Token(TokenType.NUMBER, float(result), start_pos)

    def tokenize(self) -> List[Token]:
        """Convert expression to token list."""
        tokens = []

        while self.current_char:
            if self.current_char.isdigit() or self.current_char == ".":
                tokens.append(self.number())
            elif self.current_char == "+":
                tokens.append(Token(TokenType.PLUS, "+", self.pos))
                self.advance()
            elif self.current_char == "-":
                # Determine if this is unary minus or binary minus
                is_unary = not tokens or tokens[-1].type in (
                    TokenType.LPAREN,
                    TokenType.PLUS,
                    TokenType.MINUS,
                    TokenType.STAR,
                    TokenType.SLASH,
                    TokenType.POWER,
                )
                if is_unary and self.peek() and self.peek().isdigit():
                    # Negative number literal
                    self.advance()  # consume '-'
                    num_token = self.number()
                    num_token.value = -num_token.value
                    tokens.append(num_token)
                else:
                    tokens.append(Token(TokenType.MINUS, "-", self.pos))
                    self.advance()
            elif self.current_char == "*":
                if self.peek() == "*":
                    tokens.append(Token(TokenType.POWER, "**", self.pos))
                    self.advance()
                    self.advance()
                else:
                    tokens.append(Token(TokenType.STAR, "*", self.pos))
                    self.advance()
            elif self.current_char == "/":
                tokens.append(Token(TokenType.SLASH, "/", self.pos))
                self.advance()
            elif self.current_char == "(":
                tokens.append(Token(TokenType.LPAREN, "(", self.pos))
                self.advance()
            elif self.current_char == ")":
                tokens.append(Token(TokenType.RPAREN, ")", self.pos))
                self.advance()
            else:
                raise ValueError(
                    f"Unknown character: {self.current_char} at pos {self.pos}"
                )

        tokens.append(Token(TokenType.EOF, "", self.pos))
        return tokens
