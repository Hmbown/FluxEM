"""Evaluate AST using FluxEM operations."""

from fluxem import create_extended_ops, create_unified_model

from .ast_nodes import ASTNode, BinaryOpNode, NumberNode, UnaryMinusNode
from .parser import parse_expression


class FluxEMEvaluator:
    """
    Evaluate arithmetic expressions using FluxEM embeddings.

    Operations are performed in embedding space, with decode/re-encode
    at space crossings (linear <-> log).
    """

    def __init__(
        self,
        dim: int = 256,
        linear_scale: float = 1e10,
        log_scale: float = 35.0,
        seed: int = 42,
    ):
        self.model = create_unified_model(
            dim=dim,
            linear_scale=linear_scale,
            log_scale=log_scale,
            seed=seed,
        )
        self.extended_ops = create_extended_ops(
            dim=dim,
            log_scale=log_scale,
            seed=seed,
        )

    def evaluate(self, ast: ASTNode) -> float:
        """Evaluate an AST using FluxEM operations."""
        return self._eval_node(ast)

    def _eval_node(self, node: ASTNode) -> float:
        """Recursively evaluate an AST node."""
        if isinstance(node, NumberNode):
            return node.value

        elif isinstance(node, UnaryMinusNode):
            operand_val = self._eval_node(node.operand)
            return -operand_val

        elif isinstance(node, BinaryOpNode):
            left_val = self._eval_node(node.left)
            right_val = self._eval_node(node.right)
            return self._apply_op(left_val, node.op, right_val)

        else:
            raise ValueError(f"Unknown node type: {type(node)}")

    def _apply_op(self, a: float, op: str, b: float) -> float:
        """Apply a binary operation using FluxEM embeddings."""
        if op == "+":
            emb_a = self.model.linear_encoder.encode_number(a)
            emb_b = self.model.linear_encoder.encode_number(b)
            result_emb = emb_a + emb_b
            return float(self.model.linear_encoder.decode(result_emb))

        elif op == "-":
            emb_a = self.model.linear_encoder.encode_number(a)
            emb_b = self.model.linear_encoder.encode_number(b)
            result_emb = emb_a - emb_b
            return float(self.model.linear_encoder.decode(result_emb))

        elif op == "*":
            emb_a = self.model.log_encoder.encode_number(a)
            emb_b = self.model.log_encoder.encode_number(b)
            result_emb = self.model.log_encoder.multiply(emb_a, emb_b)
            return float(self.model.log_encoder.decode(result_emb))

        elif op == "/":
            if b == 0:
                return float("inf") if a >= 0 else float("-inf")
            emb_a = self.model.log_encoder.encode_number(a)
            emb_b = self.model.log_encoder.encode_number(b)
            result_emb = self.model.log_encoder.divide(emb_a, emb_b)
            return float(self.model.log_encoder.decode(result_emb))

        elif op == "**":
            return float(self.extended_ops.power(a, b))

        else:
            raise ValueError(f"Unknown operator: {op}")


def evaluate_expression_fluxem(expr: str) -> float:
    """Evaluate an expression string with FluxEM."""
    ast = parse_expression(expr)
    evaluator = FluxEMEvaluator()
    return evaluator.evaluate(ast)
