"""
Unified arithmetic model.

This module uses:
- Linear embeddings for addition and subtraction
- Logarithmic embeddings for multiplication and division

Algebraic identities (exact over the reals; approximate under IEEE-754):
    Addition:       embed(a) + embed(b) = embed(a + b)
    Subtraction:    embed(a) - embed(b) = embed(a - b)
    Multiplication: log_mag(a) + log_mag(b) = log_mag(a * b)
    Division:       log_mag(a) - log_mag(b) = log_mag(a / b)

Reference: Flux Mathematics textbook, Chapter 8
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
from typing import Tuple, Optional, Union

from .linear_encoder import NumberEncoder
from .log_encoder import LogarithmicNumberEncoder


class UnifiedArithmeticModel(eqx.Module):
    """
    Unified arithmetic model with deterministic operations.

    Uses:
    - NumberEncoder for +, - (exact over the reals)
    - LogarithmicNumberEncoder for *, / (log-magnitude is additive)

    The arithmetic properties are built into the representation, not learned.
    """

    linear_encoder: NumberEncoder
    log_encoder: LogarithmicNumberEncoder
    dim: int
    linear_scale: float
    log_scale: float

    def __init__(
        self,
        dim: int = 256,
        linear_scale: float = 1e7,
        log_scale: float = 25.0,
        seed: int = 42,
    ):
        """
        Initialize UnifiedArithmeticModel.

        Parameters
        ----------
        dim : int
            Embedding dimension.
        linear_scale : float
            Scale for linear embeddings (max number for add/sub).
        log_scale : float
            Scale for log embeddings (log of max number for mul/div).
        seed : int
            Random seed.
        """
        self.linear_encoder = NumberEncoder(dim=dim, scale=linear_scale, seed=seed)
        self.log_encoder = LogarithmicNumberEncoder(dim=dim, log_scale=log_scale, seed=seed + 1)
        self.dim = dim
        self.linear_scale = linear_scale
        self.log_scale = log_scale

    def __call__(self, input_bytes: jax.Array) -> jax.Array:
        """
        Process an arithmetic expression and return the result embedding.

        The embedding is in LINEAR space (can be decoded with linear_encoder.decode).

        Parameters
        ----------
        input_bytes : jax.Array
            Byte sequence representing an expression like "42+58=".

        Returns
        -------
        jax.Array
            Result embedding, shape [dim].
        """
        op1_value, operator, op2_value = self._parse_expression(input_bytes)
        result = self._compute(op1_value, operator, op2_value)
        return result

    def compute(self, expr: str) -> float:
        """
        Compute an arithmetic expression from string.

        Parameters
        ----------
        expr : str
            Expression like "42+58=" or "123*456".

        Returns
        -------
        float
            Numeric result.
        """
        if '=' not in expr:
            expr = expr + '='

        byte_list = list(expr.encode('utf-8'))
        byte_list = byte_list + [0] * (64 - len(byte_list))
        input_bytes = jnp.array(byte_list, dtype=jnp.uint8)

        op1, operator, op2 = self._parse_expression(input_bytes)

        return self._compute_value(float(op1), int(operator), float(op2))

    def _compute_value(self, op1: float, operator: int, op2: float) -> float:
        """
        Compute the result using appropriate encoder.

        For + and -: uses linear embeddings (exact over the reals)
        For * and /: uses logarithmic embeddings (log-magnitude arithmetic)
        """
        PLUS = ord('+')
        MINUS = ord('-')
        STAR = ord('*')
        SLASH = ord('/')

        if operator == PLUS:
            emb1 = self.linear_encoder.encode_number(op1)
            emb2 = self.linear_encoder.encode_number(op2)
            result_emb = emb1 + emb2
            return self.linear_encoder.decode(result_emb)

        elif operator == MINUS:
            emb1 = self.linear_encoder.encode_number(op1)
            emb2 = self.linear_encoder.encode_number(op2)
            result_emb = emb1 - emb2
            return self.linear_encoder.decode(result_emb)

        elif operator == STAR:
            emb1 = self.log_encoder.encode_number(op1)
            emb2 = self.log_encoder.encode_number(op2)
            result_emb = self.log_encoder.multiply(emb1, emb2)
            return self.log_encoder.decode(result_emb)

        elif operator == SLASH:
            if op2 == 0:
                return float('-inf') if op1 < 0 else float('inf')
            emb1 = self.log_encoder.encode_number(op1)
            emb2 = self.log_encoder.encode_number(op2)
            result_emb = self.log_encoder.divide(emb1, emb2)
            return self.log_encoder.decode(result_emb)

        else:
            return 0.0

    def _compute(self, op1: jax.Array, operator: jax.Array, op2: jax.Array) -> jax.Array:
        """
        Compute result embedding using appropriate encoder.

        Returns a LINEAR embedding for consistency.
        """
        PLUS = ord('+')
        MINUS = ord('-')
        STAR = ord('*')
        SLASH = ord('/')

        lin_emb1 = self.linear_encoder.encode_number(op1)
        lin_emb2 = self.linear_encoder.encode_number(op2)
        add_result = lin_emb1 + lin_emb2
        sub_result = lin_emb1 - lin_emb2

        log_emb1 = self.log_encoder.encode_number(op1)
        log_emb2 = self.log_encoder.encode_number(op2)

        mul_log_result = self.log_encoder.multiply(log_emb1, log_emb2)
        mul_value = self._decode_log_to_value(mul_log_result)
        mul_result = self.linear_encoder.encode_number(mul_value)

        div_log_result = self.log_encoder.divide(log_emb1, log_emb2)
        div_value = jnp.where(
            op2 != 0,
            self._decode_log_to_value(div_log_result),
            jnp.where(op1 < 0, -jnp.inf, jnp.inf),
        )
        div_result = self.linear_encoder.encode_number(div_value)

        result = jnp.where(operator == PLUS, add_result,
                 jnp.where(operator == MINUS, sub_result,
                 jnp.where(operator == STAR, mul_result,
                 jnp.where(operator == SLASH, div_result,
                 jnp.zeros_like(add_result)))))

        return result

    def _decode_log_to_value(self, emb: jax.Array) -> jax.Array:
        """Decode a log embedding to a numeric value."""
        is_zero = self.log_encoder._is_zero_embedding(emb)
        finite_emb = jnp.where(jnp.isfinite(emb), emb, 0.0)

        log_normalized = jnp.dot(finite_emb, self.log_encoder.direction)
        log_value = log_normalized * self.log_encoder.log_scale
        magnitude = jnp.exp(log_value)

        has_inf = jnp.any(jnp.isinf(emb))
        magnitude = jnp.where(has_inf, jnp.inf, magnitude)

        sign = self.log_encoder._extract_sign(finite_emb)
        value = sign * magnitude

        return jnp.where(is_zero, 0.0, value)

    def _parse_expression(self, input_bytes: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Parse an expression into operands and operator."""
        PLUS = ord('+')
        MINUS = ord('-')
        STAR = ord('*')
        SLASH = ord('/')
        EQUALS = ord('=')
        ZERO = ord('0')
        NINE = ord('9')

        max_len = input_bytes.shape[0]
        positions = jnp.arange(max_len)

        is_digit = (input_bytes >= ZERO) & (input_bytes <= NINE)
        prev_is_digit = jnp.concatenate([jnp.array([False]), is_digit[:-1]])

        is_plus_op = (input_bytes == PLUS) & prev_is_digit
        is_minus_op = (input_bytes == MINUS) & prev_is_digit
        is_star_op = (input_bytes == STAR) & prev_is_digit
        is_slash_op = (input_bytes == SLASH) & prev_is_digit

        is_binary_op = is_plus_op | is_minus_op | is_star_op | is_slash_op

        op_positions = jnp.where(is_binary_op, positions, max_len)
        op_pos = jnp.min(op_positions)

        operator = input_bytes[jnp.minimum(op_pos, max_len - 1)]

        eq_positions = jnp.where(input_bytes == EQUALS, positions, max_len)
        eq_pos = jnp.min(eq_positions)

        op1_value = self._parse_number_segment(input_bytes, 0, op_pos)
        op2_start = op_pos + 1
        op2_end = jnp.where(eq_pos < max_len, eq_pos, max_len)
        op2_value = self._parse_number_segment(input_bytes, op2_start, op2_end)

        return op1_value, operator, op2_value

    def _parse_number_segment(
        self,
        input_bytes: jax.Array,
        start: jax.Array,
        end: jax.Array,
    ) -> jax.Array:
        """Parse a number from a segment of the byte array."""
        ZERO = ord('0')
        NINE = ord('9')
        MINUS = ord('-')

        max_len = input_bytes.shape[0]
        positions = jnp.arange(max_len)

        in_segment = (positions >= start) & (positions < end)
        segment_bytes = jnp.where(in_segment, input_bytes, 0)

        first_byte = input_bytes[jnp.minimum(start, max_len - 1)]
        is_negative = first_byte == MINUS

        is_digit = (segment_bytes >= ZERO) & (segment_bytes <= NINE)
        digit_values = jnp.where(is_digit, segment_bytes - ZERO, 0)

        n_digits = jnp.sum(is_digit)

        cumsum = jnp.cumsum(is_digit)
        position_in_number = jnp.where(is_digit, cumsum - 1, 0)

        place_values = jnp.where(
            is_digit,
            10.0 ** (n_digits - 1 - position_in_number),
            0.0
        )

        value = jnp.sum(digit_values * place_values)
        value = jnp.where(is_negative, -value, value)

        return value

    def decode_result(self, embedding: jax.Array) -> float:
        """Decode a linear embedding back to a number."""
        return self.linear_encoder.decode(embedding)


def create_unified_model(
    dim: int = 256,
    linear_scale: float = 1e7,
    log_scale: float = 25.0,
    seed: int = 42,
) -> UnifiedArithmeticModel:
    """Create a UnifiedArithmeticModel instance."""
    return UnifiedArithmeticModel(
        dim=dim,
        linear_scale=linear_scale,
        log_scale=log_scale,
        seed=seed,
    )


def evaluate_all_operations_ood(
    model: UnifiedArithmeticModel,
    n_samples: int = 200,
    min_num: int = -100000,
    max_num: int = 100000,
    seed: int = 42,
) -> dict:
    """
    Evaluate tolerance-based OOD accuracy on all four operations.

    Parameters
    ----------
    model : UnifiedArithmeticModel
        The model.
    n_samples : int
        Number of test samples per operation.
    min_num, max_num : int
        Range for random numbers (addition/subtraction).
    seed : int
        Random seed.

    Returns
    -------
    dict
        Dictionary with accuracy per operation and overall.
    """
    key = random.PRNGKey(seed)

    operations = {
        '+': lambda a, b: a + b,
        '-': lambda a, b: a - b,
        '*': lambda a, b: a * b,
        '/': lambda a, b: a / b if b != 0 else 0,
    }

    results = {}

    for op_char, op_fn in operations.items():
        k1, k2, key = random.split(key, 3)

        if op_char == '*':
            a_vals = random.randint(k1, (n_samples,), 10, 1000)
            b_vals = random.randint(k2, (n_samples,), 10, 1000)
        elif op_char == '/':
            a_vals = random.randint(k1, (n_samples,), 100, 10000)
            b_vals = random.randint(k2, (n_samples,), 10, 100)
        else:
            a_vals = random.randint(k1, (n_samples,), min_num, max_num)
            b_vals = random.randint(k2, (n_samples,), min_num, max_num)

        correct = 0

        for i in range(n_samples):
            a = int(a_vals[i])
            b = int(b_vals[i])

            if op_char == '/' and b == 0:
                b = 1

            expected = op_fn(a, b)
            expr = f"{a}{op_char}{b}="
            predicted = model.compute(expr)

            if abs(expected) > 1:
                rel_error = abs(predicted - expected) / abs(expected)
                is_correct = rel_error < 0.01
            else:
                is_correct = abs(predicted - expected) < 0.5

            if is_correct:
                correct += 1

        accuracy = correct / n_samples
        results[op_char] = accuracy

    results['overall'] = sum(results.values()) / len(operations)

    return results


if __name__ == "__main__":
    print("UnifiedArithmeticModel demo")

    model = create_unified_model(dim=256, linear_scale=1e7, log_scale=25.0, seed=42)

    print("\nIn-distribution tests:")

    test_cases = [
        ("42+58=", 100),
        ("100-37=", 63),
        ("6*7=", 42),
        ("100/4=", 25),
        ("123+456=", 579),
        ("1000-500=", 500),
        ("12*12=", 144),
        ("144/12=", 12),
    ]

    for expr, expected in test_cases:
        result = model.compute(expr)
        error = abs(result - expected)
        rel_error = error / abs(expected) if expected != 0 else error
        status = "PASS" if rel_error < 0.01 else "FAIL"
        print(f"  {expr:12} = {result:>12.2f} (expected: {expected:>8}, rel_err: {rel_error:.2e}) [{status}]")

    print("\nOut-of-distribution tests:")

    ood_cases = [
        ("12345+54321=", 66666),
        ("99999-88888=", 11111),
        ("456*789=", 359784),
        ("56088/123=", 456),
        ("50000+50000=", 100000),
        ("1000*1000=", 1000000),
    ]

    for expr, expected in ood_cases:
        result = model.compute(expr)
        error = abs(result - expected)
        rel_error = error / abs(expected) if expected != 0 else error
        status = "PASS" if rel_error < 0.01 else "FAIL"
        print(f"  {expr:16} = {result:>12.2f} (expected: {expected:>8}, rel_err: {rel_error:.2e}) [{status}]")

    print("\nStatistical OOD evaluation:")

    results = evaluate_all_operations_ood(model, n_samples=100)

    for op, acc in results.items():
        print(f"  {op:>8}: {acc:.1%}")
