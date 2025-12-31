"""AST node definitions for arithmetic expressions."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


class ASTNode(ABC):
    """Base class for AST nodes."""

    @abstractmethod
    def __repr__(self) -> str:
        pass


@dataclass
class NumberNode(ASTNode):
    """Leaf node representing a numeric literal."""

    value: float

    def __repr__(self) -> str:
        return f"Num({self.value})"


@dataclass
class BinaryOpNode(ASTNode):
    """Binary operation node."""

    op: Literal["+", "-", "*", "/", "**"]
    left: ASTNode
    right: ASTNode

    def __repr__(self) -> str:
        return f"({self.left} {self.op} {self.right})"


@dataclass
class UnaryMinusNode(ASTNode):
    """Unary negation node."""

    operand: ASTNode

    def __repr__(self) -> str:
        return f"(-{self.operand})"
