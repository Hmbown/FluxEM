"""
Matrix Encoder.

Embeds matrices with log-magnitude representation and structural properties.
Supports matrices up to 4x4 (16 elements).

Matrix addition is component-wise (requires decode-operate-encode due to log).
Matrix multiplication requires decode-operate-encode.
Scalar multiplication is computed in log-space.
"""

import math
from typing import Any, List, Tuple, Union, Optional
from ...backend import get_backend

from ...core.base import (
    DOMAIN_TAGS,
    EMBEDDING_DIM,
    EPSILON,
    create_embedding,
    log_encode_value,
    log_decode_value,
)


# Maximum matrix dimensions (4x4 = 16 elements)
MAX_ROWS = 4
MAX_COLS = 4
MAX_ELEMENTS = MAX_ROWS * MAX_COLS

# Embedding layout within domain-specific region (dims 8-71):
# dims 0-1:   Rows (1.0, rows/4)
# dims 2-3:   Cols (1.0, cols/4)
# dims 4-5:   Determinant (sign, log|det|) - for square matrices
# dims 6-7:   Trace (sign, log|trace|) - for square matrices
# dims 8-9:   Frobenius norm (1.0, log|F|)
# dims 10:    is_square flag
# dims 11:    is_symmetric flag
# dims 12:    is_diagonal flag
# dims 13:    is_identity flag
# dims 14-45: Elements in row-major order (16 pairs of sign, log|element|)
# dims 46-63: Reserved

ROWS_OFFSET = 0
COLS_OFFSET = 2
DET_OFFSET = 4
TRACE_OFFSET = 6
FROB_OFFSET = 8
FLAG_IS_SQUARE = 10
FLAG_IS_SYMMETRIC = 11
FLAG_IS_DIAGONAL = 12
FLAG_IS_IDENTITY = 13
ELEMENTS_OFFSET = 14


class MatrixEncoder:
    """
    Encoder for real matrices.

    Uses log-magnitude representation for each element.
    Supports matrices up to 4x4.

    Stores structural properties: determinant, trace, Frobenius norm.
    """

    domain_tag = DOMAIN_TAGS["math_matrix"]
    domain_name = "math_matrix"

    def encode(self, m: Union[List[List[float]], Any]) -> Any:
        """
        Encode a matrix.

        Args:
            m: Matrix as list of lists or 2D array (max 4x4)

        Returns:
            128-dim embedding
        """
        backend = get_backend()
        if hasattr(m, "tolist"):
            m = m.tolist()

        rows = len(m)
        if rows == 0:
            raise ValueError("Cannot encode empty matrix")
        cols = len(m[0])

        if rows > MAX_ROWS or cols > MAX_COLS:
            raise ValueError(f"Matrix {rows}x{cols} exceeds maximum {MAX_ROWS}x{MAX_COLS}")

        # Ensure all rows have same length
        for row in m:
            if len(row) != cols:
                raise ValueError("All rows must have same length")

        emb = create_embedding()

        # Domain tag
        emb = backend.at_add(emb, slice(0, 8), self.domain_tag)

        # Dimensions
        emb = backend.at_add(emb, 8 + ROWS_OFFSET, 1.0)
        emb = backend.at_add(emb, 8 + ROWS_OFFSET + 1, rows / MAX_ROWS)
        emb = backend.at_add(emb, 8 + COLS_OFFSET, 1.0)
        emb = backend.at_add(emb, 8 + COLS_OFFSET + 1, cols / MAX_COLS)

        # Compute properties
        is_square = rows == cols

        # Encode elements in row-major order
        frob_sq = 0.0
        for i in range(rows):
            for j in range(cols):
                val = m[i][j]
                idx = i * MAX_COLS + j
                sign, log_mag = log_encode_value(val)
                emb = backend.at_add(emb, 8 + ELEMENTS_OFFSET + 2*idx, sign)
                emb = backend.at_add(emb, 8 + ELEMENTS_OFFSET + 2*idx + 1, log_mag)
                frob_sq += val * val

        # Frobenius norm
        frob = math.sqrt(frob_sq) if frob_sq > 0 else 0.0
        if frob < EPSILON:
            emb = backend.at_add(emb, 8 + FROB_OFFSET, 0.0)
            emb = backend.at_add(emb, 8 + FROB_OFFSET + 1, -100.0)
        else:
            emb = backend.at_add(emb, 8 + FROB_OFFSET, 1.0)
            emb = backend.at_add(emb, 8 + FROB_OFFSET + 1, math.log(frob))

        # Flags
        emb = backend.at_add(emb, 8 + FLAG_IS_SQUARE, 1.0 if is_square else 0.0)

        if is_square:
            # Compute determinant and trace
            det = self._compute_determinant(m)
            trace = sum(m[i][i] for i in range(rows))

            det_sign, det_log = log_encode_value(det)
            emb = backend.at_add(emb, 8 + DET_OFFSET, det_sign)
            emb = backend.at_add(emb, 8 + DET_OFFSET + 1, det_log)

            trace_sign, trace_log = log_encode_value(trace)
            emb = backend.at_add(emb, 8 + TRACE_OFFSET, trace_sign)
            emb = backend.at_add(emb, 8 + TRACE_OFFSET + 1, trace_log)

            # Check symmetry
            is_symmetric = all(
                abs(m[i][j] - m[j][i]) < EPSILON
                for i in range(rows) for j in range(i+1, cols)
            )
            emb = backend.at_add(emb, 8 + FLAG_IS_SYMMETRIC, 1.0 if is_symmetric else 0.0)

            # Check diagonal
            is_diagonal = all(
                abs(m[i][j]) < EPSILON
                for i in range(rows) for j in range(cols) if i != j
            )
            emb = backend.at_add(emb, 8 + FLAG_IS_DIAGONAL, 1.0 if is_diagonal else 0.0)

            # Check identity
            is_identity = is_diagonal and all(
                abs(m[i][i] - 1.0) < EPSILON for i in range(rows)
            )
            emb = backend.at_add(emb, 8 + FLAG_IS_IDENTITY, 1.0 if is_identity else 0.0)

        return emb

    def _compute_determinant(self, m: List[List[float]]) -> float:
        """Compute determinant for small matrices."""
        n = len(m)
        if n == 1:
            return m[0][0]
        elif n == 2:
            return m[0][0] * m[1][1] - m[0][1] * m[1][0]
        elif n == 3:
            return (
                m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
            )
        elif n == 4:
            # Laplace expansion along first row
            det = 0.0
            for j in range(4):
                minor = [
                    [m[i][k] for k in range(4) if k != j]
                    for i in range(1, 4)
                ]
                cofactor = ((-1) ** j) * self._compute_determinant(minor)
                det += m[0][j] * cofactor
            return det
        else:
            return 0.0

    def decode(self, emb: Any) -> List[List[float]]:
        """
        Decode embedding to matrix.

        Returns:
            List of lists representing the matrix
        """
        rows = int(round(emb[8 + ROWS_OFFSET + 1].item() * MAX_ROWS))
        cols = int(round(emb[8 + COLS_OFFSET + 1].item() * MAX_COLS))

        rows = max(1, min(rows, MAX_ROWS))
        cols = max(1, min(cols, MAX_COLS))

        result = []
        for i in range(rows):
            row = []
            for j in range(cols):
                idx = i * MAX_COLS + j
                sign = emb[8 + ELEMENTS_OFFSET + 2*idx].item()
                log_mag = emb[8 + ELEMENTS_OFFSET + 2*idx + 1].item()
                row.append(log_decode_value(sign, log_mag))
            result.append(row)

        return result

    def is_valid(self, emb: Any) -> bool:
        """Check if embedding is a valid matrix."""
        backend = get_backend()
        tag = emb[0:8]
        return backend.allclose(tag, self.domain_tag, atol=0.1).item()

    def get_shape(self, emb: Any) -> Tuple[int, int]:
        """Get the shape (rows, cols) of the encoded matrix."""
        rows = int(round(emb[8 + ROWS_OFFSET + 1].item() * MAX_ROWS))
        cols = int(round(emb[8 + COLS_OFFSET + 1].item() * MAX_COLS))
        return (max(1, min(rows, MAX_ROWS)), max(1, min(cols, MAX_COLS)))

    def get_determinant(self, emb: Any) -> Optional[float]:
        """Get the determinant (only for square matrices)."""
        if not self.is_square(emb):
            return None
        sign = emb[8 + DET_OFFSET].item()
        log_mag = emb[8 + DET_OFFSET + 1].item()
        return log_decode_value(sign, log_mag)

    def get_trace(self, emb: Any) -> Optional[float]:
        """Get the trace (only for square matrices)."""
        if not self.is_square(emb):
            return None
        sign = emb[8 + TRACE_OFFSET].item()
        log_mag = emb[8 + TRACE_OFFSET + 1].item()
        return log_decode_value(sign, log_mag)

    def get_frobenius_norm(self, emb: Any) -> float:
        """Get the Frobenius norm."""
        sign = emb[8 + FROB_OFFSET].item()
        log_mag = emb[8 + FROB_OFFSET + 1].item()
        return log_decode_value(sign, log_mag)

    def is_square(self, emb: Any) -> bool:
        """Check if the matrix is square."""
        return emb[8 + FLAG_IS_SQUARE].item() > 0.5

    def is_symmetric(self, emb: Any) -> bool:
        """Check if the matrix is symmetric."""
        return emb[8 + FLAG_IS_SYMMETRIC].item() > 0.5

    def is_diagonal(self, emb: Any) -> bool:
        """Check if the matrix is diagonal."""
        return emb[8 + FLAG_IS_DIAGONAL].item() > 0.5

    def is_identity(self, emb: Any) -> bool:
        """Check if the matrix is the identity."""
        return emb[8 + FLAG_IS_IDENTITY].item() > 0.5

    # =========================================================================
    # Operations
    # =========================================================================

    def add(self, emb1: Any, emb2: Any) -> Any:
        """Add two matrices."""
        m1 = self.decode(emb1)
        m2 = self.decode(emb2)

        shape1 = (len(m1), len(m1[0]))
        shape2 = (len(m2), len(m2[0]))

        if shape1 != shape2:
            raise ValueError(f"Cannot add matrices of shapes {shape1} and {shape2}")

        result = [
            [m1[i][j] + m2[i][j] for j in range(shape1[1])]
            for i in range(shape1[0])
        ]
        return self.encode(result)

    def subtract(self, emb1: Any, emb2: Any) -> Any:
        """Subtract two matrices."""
        m1 = self.decode(emb1)
        m2 = self.decode(emb2)

        shape1 = (len(m1), len(m1[0]))
        shape2 = (len(m2), len(m2[0]))

        if shape1 != shape2:
            raise ValueError(f"Cannot subtract matrices of shapes {shape1} and {shape2}")

        result = [
            [m1[i][j] - m2[i][j] for j in range(shape1[1])]
            for i in range(shape1[0])
        ]
        return self.encode(result)

    def scale(self, emb: Any, scalar: float) -> Any:
        """
        Multiply matrix by scalar.

        Computed in log-space: add log(|scalar|) to each log-magnitude.
        """
        backend = get_backend()
        if abs(scalar) < EPSILON:
            rows, cols = self.get_shape(emb)
            return self.encode([[0.0] * cols for _ in range(rows)])

        result = create_embedding()
        result = backend.at_add(result, slice(0, 8), self.domain_tag)

        # Copy dimensions
        result = backend.at_add(result, 8 + ROWS_OFFSET, emb[8 + ROWS_OFFSET])
        result = backend.at_add(result, 8 + ROWS_OFFSET + 1, emb[8 + ROWS_OFFSET + 1])
        result = backend.at_add(result, 8 + COLS_OFFSET, emb[8 + COLS_OFFSET])
        result = backend.at_add(result, 8 + COLS_OFFSET + 1, emb[8 + COLS_OFFSET + 1])

        rows, cols = self.get_shape(emb)
        scalar_sign = 1.0 if scalar > 0 else -1.0
        log_scalar = math.log(abs(scalar))

        # Scale each element
        for i in range(rows):
            for j in range(cols):
                idx = i * MAX_COLS + j
                sign = emb[8 + ELEMENTS_OFFSET + 2*idx].item()
                log_mag = emb[8 + ELEMENTS_OFFSET + 2*idx + 1].item()

                if abs(sign) < 0.5:
                    result = backend.at_add(result, 8 + ELEMENTS_OFFSET + 2*idx, 0.0)
                    result = backend.at_add(result, 8 + ELEMENTS_OFFSET + 2*idx + 1, -100.0)
                else:
                    new_sign = sign * scalar_sign
                    new_log_mag = log_mag + log_scalar
                    result = backend.at_add(result, 8 + ELEMENTS_OFFSET + 2*idx, new_sign)
                    result = backend.at_add(result, 8 + ELEMENTS_OFFSET + 2*idx + 1, new_log_mag)

        # Update Frobenius norm
        old_frob_sign = emb[8 + FROB_OFFSET].item()
        old_log_frob = emb[8 + FROB_OFFSET + 1].item()
        if abs(old_frob_sign) < 0.5:
            result = backend.at_add(result, 8 + FROB_OFFSET, 0.0)
            result = backend.at_add(result, 8 + FROB_OFFSET + 1, -100.0)
        else:
            result = backend.at_add(result, 8 + FROB_OFFSET, 1.0)
            result = backend.at_add(result, 8 + FROB_OFFSET + 1, old_log_frob + log_scalar)

        # Update flags - square, symmetric, diagonal preserved under scaling
        result = backend.at_add(result, 8 + FLAG_IS_SQUARE, 
            emb[8 + FLAG_IS_SQUARE]
        )
        result = backend.at_add(result, 8 + FLAG_IS_SYMMETRIC, 
            emb[8 + FLAG_IS_SYMMETRIC]
        )
        result = backend.at_add(result, 8 + FLAG_IS_DIAGONAL, 
            emb[8 + FLAG_IS_DIAGONAL]
        )
        # Identity is only preserved if scalar = 1
        if abs(scalar - 1.0) < EPSILON:
            result = backend.at_add(result, 8 + FLAG_IS_IDENTITY, 
                emb[8 + FLAG_IS_IDENTITY]
            )

        # Update determinant: det(cA) = c^n * det(A)
        if self.is_square(emb):
            n = rows
            old_det_sign = emb[8 + DET_OFFSET].item()
            old_det_log = emb[8 + DET_OFFSET + 1].item()

            if abs(old_det_sign) < 0.5:
                result = backend.at_add(result, 8 + DET_OFFSET, 0.0)
                result = backend.at_add(result, 8 + DET_OFFSET + 1, -100.0)
            else:
                new_det_sign = old_det_sign * (scalar_sign ** n)
                new_det_log = old_det_log + n * log_scalar
                result = backend.at_add(result, 8 + DET_OFFSET, new_det_sign)
                result = backend.at_add(result, 8 + DET_OFFSET + 1, new_det_log)

            # Update trace: trace(cA) = c * trace(A)
            old_trace_sign = emb[8 + TRACE_OFFSET].item()
            old_trace_log = emb[8 + TRACE_OFFSET + 1].item()

            if abs(old_trace_sign) < 0.5:
                result = backend.at_add(result, 8 + TRACE_OFFSET, 0.0)
                result = backend.at_add(result, 8 + TRACE_OFFSET + 1, -100.0)
            else:
                new_trace_sign = old_trace_sign * scalar_sign
                new_trace_log = old_trace_log + log_scalar
                result = backend.at_add(result, 8 + TRACE_OFFSET, new_trace_sign)
                result = backend.at_add(result, 8 + TRACE_OFFSET + 1, new_trace_log)

        return result

    def multiply(self, emb1: Any, emb2: Any) -> Any:
        """
        Matrix multiplication.

        Requires decode-operate-encode.
        """
        m1 = self.decode(emb1)
        m2 = self.decode(emb2)

        rows1, cols1 = len(m1), len(m1[0])
        rows2, cols2 = len(m2), len(m2[0])

        if cols1 != rows2:
            raise ValueError(
                f"Cannot multiply matrices: {rows1}x{cols1} and {rows2}x{cols2}"
            )

        result = [
            [
                sum(m1[i][k] * m2[k][j] for k in range(cols1))
                for j in range(cols2)
            ]
            for i in range(rows1)
        ]
        return self.encode(result)

    def transpose(self, emb: Any) -> Any:
        """Transpose the matrix."""
        m = self.decode(emb)
        rows, cols = len(m), len(m[0])
        result = [[m[j][i] for j in range(rows)] for i in range(cols)]
        return self.encode(result)

    def negate(self, emb: Any) -> Any:
        """Negate the matrix."""
        return self.scale(emb, -1.0)

    def element(self, emb: Any, i: int, j: int) -> float:
        """Get element at position (i, j)."""
        rows, cols = self.get_shape(emb)
        if i < 0 or i >= rows or j < 0 or j >= cols:
            raise IndexError(f"Index ({i}, {j}) out of range for {rows}x{cols} matrix")

        idx = i * MAX_COLS + j
        sign = emb[8 + ELEMENTS_OFFSET + 2*idx].item()
        log_mag = emb[8 + ELEMENTS_OFFSET + 2*idx + 1].item()
        return log_decode_value(sign, log_mag)

    def row(self, emb: Any, i: int) -> List[float]:
        """Get row i of the matrix."""
        rows, cols = self.get_shape(emb)
        if i < 0 or i >= rows:
            raise IndexError(f"Row {i} out of range for {rows}x{cols} matrix")

        return [self.element(emb, i, j) for j in range(cols)]

    def column(self, emb: Any, j: int) -> List[float]:
        """Get column j of the matrix."""
        rows, cols = self.get_shape(emb)
        if j < 0 or j >= cols:
            raise IndexError(f"Column {j} out of range for {rows}x{cols} matrix")

        return [self.element(emb, i, j) for i in range(rows)]

    # =========================================================================
    # Special Matrices
    # =========================================================================

    @staticmethod
    def identity(n: int) -> Any:
        """Create an n x n identity matrix embedding."""
        if n < 1 or n > MAX_ROWS:
            raise ValueError(f"Identity size {n} must be between 1 and {MAX_ROWS}")

        m = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        return MatrixEncoder().encode(m)

    @staticmethod
    def zeros(rows: int, cols: int) -> Any:
        """Create a zero matrix embedding."""
        if rows < 1 or rows > MAX_ROWS or cols < 1 or cols > MAX_COLS:
            raise ValueError(f"Matrix size {rows}x{cols} out of range")

        m = [[0.0 for _ in range(cols)] for _ in range(rows)]
        return MatrixEncoder().encode(m)

    @staticmethod
    def diagonal(values: List[float]) -> Any:
        """Create a diagonal matrix from values."""
        n = len(values)
        if n < 1 or n > MAX_ROWS:
            raise ValueError(f"Diagonal size {n} must be between 1 and {MAX_ROWS}")

        m = [[values[i] if i == j else 0.0 for j in range(n)] for i in range(n)]
        return MatrixEncoder().encode(m)
