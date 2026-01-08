"""Control systems domain tools and encoder."""

from __future__ import annotations

import cmath
from dataclasses import dataclass
from typing import Any, List, Sequence

from ...backend import get_backend
from ...core.base import (
    DOMAIN_TAGS,
    create_embedding,
    log_encode_value,
    log_decode_value,
)


def _mat_vec_mul(matrix: Sequence[Sequence[float]], vec: Sequence[float]) -> List[float]:
    return [sum(m_ij * v_j for m_ij, v_j in zip(row, vec)) for row in matrix]


def state_update(
    a: Sequence[Sequence[float]],
    b: Sequence[Sequence[float]],
    x: Sequence[float],
    u: Sequence[float],
) -> List[float]:
    """Discrete state update x_{t+1} = A x_t + B u_t."""
    if len(a) == 0 or len(b) == 0:
        raise ValueError("A and B must be non-empty")
    if len(a) != len(x):
        raise ValueError("A rows must match x length")
    if any(len(row) != len(x) for row in a):
        raise ValueError("A must be square with size len(x)")
    if any(len(row) != len(u) for row in b):
        raise ValueError("B columns must match u length")

    ax = _mat_vec_mul(a, x)
    bu = _mat_vec_mul(b, u)
    return [a_val + b_val for a_val, b_val in zip(ax, bu)]


def is_stable_2x2(a: Sequence[Sequence[float]]) -> bool:
    """Check discrete-time stability for 2x2 A (eigenvalues inside unit circle)."""
    if len(a) != 2 or any(len(row) != 2 for row in a):
        raise ValueError("A must be 2x2")
    a11, a12 = a[0]
    a21, a22 = a[1]
    trace = a11 + a22
    det = a11 * a22 - a12 * a21
    discriminant = trace * trace - 4 * det
    sqrt_disc = cmath.sqrt(discriminant)
    lam1 = (trace + sqrt_disc) / 2
    lam2 = (trace - sqrt_disc) / 2
    return abs(lam1) < 1.0 and abs(lam2) < 1.0


A_OFFSET = 0
B_OFFSET = 8
X_OFFSET = 16
U_OFFSET = 20
TRACE_OFFSET = 24
DET_OFFSET = 26
SPECTRAL_OFFSET = 28
STABLE_FLAG = 30


@dataclass(frozen=True)
class ControlState:
    """Linear system state representation for encoding."""

    a: List[List[float]]
    b: List[List[float]]
    x: List[float]
    u: List[float]
    trace: float
    det: float
    spectral_radius: float
    is_stable: bool


def _check_2x2(matrix: Sequence[Sequence[float]], name: str) -> None:
    if len(matrix) != 2 or any(len(row) != 2 for row in matrix):
        raise ValueError(f"{name} must be 2x2")


def _scalar(value: Any) -> float:
    return value.item() if hasattr(value, "item") else float(value)


class ControlSystemEncoder:
    """Encoder for 2x2 linear control system state."""

    domain_tag = DOMAIN_TAGS["control_state"]
    domain_name = "control_state"

    def encode(
        self,
        a: Sequence[Sequence[float]],
        b: Sequence[Sequence[float]],
        x: Sequence[float],
        u: Sequence[float],
    ) -> Any:
        """Encode system matrices and state into an embedding."""
        _check_2x2(a, "A")
        _check_2x2(b, "B")
        if len(x) != 2 or len(u) != 2:
            raise ValueError("x and u must have length 2")

        a11, a12 = a[0]
        a21, a22 = a[1]
        trace = a11 + a22
        det = a11 * a22 - a12 * a21
        discriminant = trace * trace - 4 * det
        sqrt_disc = cmath.sqrt(discriminant)
        lam1 = (trace + sqrt_disc) / 2
        lam2 = (trace - sqrt_disc) / 2
        spectral_radius = max(abs(lam1), abs(lam2))
        is_stable = spectral_radius < 1.0

        backend = get_backend()
        emb = create_embedding()
        emb = backend.at_add(emb, slice(0, 16), self.domain_tag)

        a_values = [a11, a12, a21, a22]
        for i, val in enumerate(a_values):
            sign, log_mag = log_encode_value(float(val))
            offset = 16 + A_OFFSET + 2 * i
            emb = backend.at_add(emb, offset, sign)
            emb = backend.at_add(emb, offset + 1, log_mag)

        b_values = [b[0][0], b[0][1], b[1][0], b[1][1]]
        for i, val in enumerate(b_values):
            sign, log_mag = log_encode_value(float(val))
            offset = 16 + B_OFFSET + 2 * i
            emb = backend.at_add(emb, offset, sign)
            emb = backend.at_add(emb, offset + 1, log_mag)

        for i, val in enumerate(list(x)):
            sign, log_mag = log_encode_value(float(val))
            offset = 16 + X_OFFSET + 2 * i
            emb = backend.at_add(emb, offset, sign)
            emb = backend.at_add(emb, offset + 1, log_mag)

        for i, val in enumerate(list(u)):
            sign, log_mag = log_encode_value(float(val))
            offset = 16 + U_OFFSET + 2 * i
            emb = backend.at_add(emb, offset, sign)
            emb = backend.at_add(emb, offset + 1, log_mag)

        trace_sign, trace_log = log_encode_value(float(trace))
        emb = backend.at_add(emb, 16 + TRACE_OFFSET, trace_sign)
        emb = backend.at_add(emb, 16 + TRACE_OFFSET + 1, trace_log)

        det_sign, det_log = log_encode_value(float(det))
        emb = backend.at_add(emb, 16 + DET_OFFSET, det_sign)
        emb = backend.at_add(emb, 16 + DET_OFFSET + 1, det_log)

        spec_sign, spec_log = log_encode_value(float(spectral_radius))
        emb = backend.at_add(emb, 16 + SPECTRAL_OFFSET, spec_sign)
        emb = backend.at_add(emb, 16 + SPECTRAL_OFFSET + 1, spec_log)

        emb = backend.at_add(emb, 16 + STABLE_FLAG, 1.0 if is_stable else 0.0)

        return emb

    def decode(self, emb: Any) -> ControlState:
        """Decode an embedding back to a control system summary."""
        a_values = []
        for i in range(4):
            offset = 16 + A_OFFSET + 2 * i
            a_values.append(log_decode_value(_scalar(emb[offset]), _scalar(emb[offset + 1])))
        a = [[a_values[0], a_values[1]], [a_values[2], a_values[3]]]

        b_values = []
        for i in range(4):
            offset = 16 + B_OFFSET + 2 * i
            b_values.append(log_decode_value(_scalar(emb[offset]), _scalar(emb[offset + 1])))
        b = [[b_values[0], b_values[1]], [b_values[2], b_values[3]]]

        x_vals = []
        for i in range(2):
            offset = 16 + X_OFFSET + 2 * i
            x_vals.append(log_decode_value(_scalar(emb[offset]), _scalar(emb[offset + 1])))

        u_vals = []
        for i in range(2):
            offset = 16 + U_OFFSET + 2 * i
            u_vals.append(log_decode_value(_scalar(emb[offset]), _scalar(emb[offset + 1])))

        trace = log_decode_value(_scalar(emb[16 + TRACE_OFFSET]), _scalar(emb[16 + TRACE_OFFSET + 1]))
        det = log_decode_value(_scalar(emb[16 + DET_OFFSET]), _scalar(emb[16 + DET_OFFSET + 1]))
        spectral = log_decode_value(_scalar(emb[16 + SPECTRAL_OFFSET]), _scalar(emb[16 + SPECTRAL_OFFSET + 1]))
        is_stable = _scalar(emb[16 + STABLE_FLAG]) > 0.5

        return ControlState(
            a=a,
            b=b,
            x=x_vals,
            u=u_vals,
            trace=trace,
            det=det,
            spectral_radius=spectral,
            is_stable=is_stable,
        )

    def is_valid(self, emb: Any) -> bool:
        """Check if embedding is a valid control system state."""
        backend = get_backend()
        tag = emb[0:16]
        return backend.allclose(tag, self.domain_tag, atol=0.1).item()


__all__ = [
    "ControlState",
    "ControlSystemEncoder",
    "state_update",
    "is_stable_2x2",
]
