"""
Recursive descent parser for arithmetic expressions.

Grammar (with operator precedence):
    expr   -> term (('+' | '-') term)*
    term   -> power (('*' | '/') power)*
    power  -> unary ('**' power)?          # Right associative
    unary  -> '-' unary | primary
    primary -> NUMBER | '(' expr ')'
"""

from typing import List

from .ast_nodes import ASTNode, BinaryOpNode, NumberNode, UnaryMinusNode
from .tokenizer import Token, Tokenizer, TokenType


class Parser:
    """Recursive descent parser producing an AST."""

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        self.current = tokens[0]

    def advance(self):
        """Move to next token."""
        self.pos += 1
        self.current = self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def expect(self, token_type: TokenType) -> Token:
        """Consume token of expected type or raise error."""
        if self.current.type != token_type:
            raise SyntaxError(f"Expected {token_type}, got {self.current.type}")
        token = self.current
        self.advance()
        return token

    def parse(self) -> ASTNode:
        """Parse the full expression."""
        ast = self.expr()
        if self.current.type != TokenType.EOF:
            raise SyntaxError(f"Unexpected token after expression: {self.current}")
        return ast

    def expr(self) -> ASTNode:
        """expr -> term (('+' | '-') term)*"""
        node = self.term()

        while self.current.type in (TokenType.PLUS, TokenType.MINUS):
            op = "+" if self.current.type == TokenType.PLUS else "-"
            self.advance()
            right = self.term()
            node = BinaryOpNode(op, node, right)

        return node

    def term(self) -> ASTNode:
        """term -> power (('*' | '/') power)*"""
        node = self.power()

        while self.current.type in (TokenType.STAR, TokenType.SLASH):
            op = "*" if self.current.type == TokenType.STAR else "/"
            self.advance()
            right = self.power()
            node = BinaryOpNode(op, node, right)

        return node

    def power(self) -> ASTNode:
        """power -> unary ('**' power)?  (right associative)"""
        node = self.unary()

        if self.current.type == TokenType.POWER:
            self.advance()
            right = self.power()  # Right associative
            node = BinaryOpNode("**", node, right)

        return node

    def unary(self) -> ASTNode:
        """unary -> '-' unary | primary"""
        if self.current.type == TokenType.MINUS:
            self.advance()
            operand = self.unary()
            return UnaryMinusNode(operand)
        return self.primary()

    def primary(self) -> ASTNode:
        """primary -> NUMBER | '(' expr ')'"""
        if self.current.type == TokenType.NUMBER:
            value = self.current.value
            self.advance()
            return NumberNode(value)

        if self.current.type == TokenType.LPAREN:
            self.advance()
            node = self.expr()
            self.expect(TokenType.RPAREN)
            return node

        raise SyntaxError(f"Unexpected token: {self.current}")


def parse_expression(expr: str) -> ASTNode:
    """Parse an expression string into an AST."""
    tokenizer = Tokenizer(expr)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens)
    return parser.parse()
