"""Signal processing domain tools and encoder."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from ...backend import get_backend
from ...core.base import (
    DOMAIN_TAGS,
    create_embedding,
    log_encode_value,
    log_decode_value,
)


def convolution(signal: Sequence[float], kernel: Sequence[float]) -> List[float]:
    """Discrete convolution of two sequences."""
    if not signal or not kernel:
        raise ValueError("signal and kernel must be non-empty")
    out_len = len(signal) + len(kernel) - 1
    output = [0.0] * out_len
    for i, s_val in enumerate(signal):
        for j, k_val in enumerate(kernel):
            output[i + j] += s_val * k_val
    return output


def moving_average(signal: Sequence[float], window: int) -> List[float]:
    """Simple moving average with fixed window size."""
    if window <= 0:
        raise ValueError("window must be > 0")
    if len(signal) < window:
        raise ValueError("window must be <= len(signal)")
    result = []
    window_sum = sum(signal[:window])
    result.append(window_sum / window)
    for i in range(window, len(signal)):
        window_sum += signal[i] - signal[i - window]
        result.append(window_sum / window)
    return result


def dft_magnitude(signal: Sequence[float]) -> List[float]:
    """Compute DFT magnitudes for a real-valued signal."""
    if not signal:
        raise ValueError("signal must be non-empty")
    n = len(signal)
    magnitudes = []
    for k in range(n):
        real = 0.0
        imag = 0.0
        for t, x_t in enumerate(signal):
            angle = -2.0 * math.pi * k * t / n
            real += x_t * math.cos(angle)
            imag += x_t * math.sin(angle)
        magnitudes.append((real ** 2 + imag ** 2) ** 0.5)
    return magnitudes


def _scalar(value: Any) -> float:
    return value.item() if hasattr(value, "item") else float(value)


MAX_SIGNAL_VALUES = 16
MAX_SIGNAL_LEN = 64
LENGTH_OFFSET = 0
MEAN_OFFSET = 1
RMS_OFFSET = 3
ENERGY_OFFSET = 5
MIN_OFFSET = 7
MAX_OFFSET = 9
SAMPLE_RATE_OFFSET = 11
HAS_SAMPLE_RATE_FLAG = 13
VALUES_OFFSET = 14


@dataclass(frozen=True)
class SignalSummary:
    """Summary of a signal for embedding."""

    samples: List[float]
    length: int
    sample_rate: Optional[float]
    mean: float
    rms: float
    energy: float
    min_value: float
    max_value: float


class SignalEncoder:
    """Encoder for signal samples and summary statistics."""

    domain_tag = DOMAIN_TAGS["signal_sequence"]
    domain_name = "signal_sequence"

    def encode(self, value: SignalSummary | Sequence[float], sample_rate: Optional[float] = None) -> Any:
        """Encode a signal into an embedding."""
        if isinstance(value, SignalSummary):
            summary = value
        else:
            samples = list(value)
            if not samples:
                raise ValueError("signal must be non-empty")
            length = len(samples)
            mean_val = sum(samples) / length
            energy_val = sum(x * x for x in samples)
            rms_val = math.sqrt(energy_val / length)
            min_val = min(samples)
            max_val = max(samples)
            summary = SignalSummary(
                samples=samples,
                length=length,
                sample_rate=sample_rate,
                mean=mean_val,
                rms=rms_val,
                energy=energy_val,
                min_value=min_val,
                max_value=max_val,
            )

        backend = get_backend()
        emb = create_embedding()
        emb = backend.at_add(emb, slice(0, 16), self.domain_tag)

        length_norm = min(summary.length, MAX_SIGNAL_LEN) / MAX_SIGNAL_LEN
        emb = backend.at_add(emb, 16 + LENGTH_OFFSET, length_norm)

        mean_sign, mean_log = log_encode_value(summary.mean)
        emb = backend.at_add(emb, 16 + MEAN_OFFSET, mean_sign)
        emb = backend.at_add(emb, 16 + MEAN_OFFSET + 1, mean_log)

        rms_sign, rms_log = log_encode_value(summary.rms)
        emb = backend.at_add(emb, 16 + RMS_OFFSET, rms_sign)
        emb = backend.at_add(emb, 16 + RMS_OFFSET + 1, rms_log)

        energy_sign, energy_log = log_encode_value(summary.energy)
        emb = backend.at_add(emb, 16 + ENERGY_OFFSET, energy_sign)
        emb = backend.at_add(emb, 16 + ENERGY_OFFSET + 1, energy_log)

        min_sign, min_log = log_encode_value(summary.min_value)
        emb = backend.at_add(emb, 16 + MIN_OFFSET, min_sign)
        emb = backend.at_add(emb, 16 + MIN_OFFSET + 1, min_log)

        max_sign, max_log = log_encode_value(summary.max_value)
        emb = backend.at_add(emb, 16 + MAX_OFFSET, max_sign)
        emb = backend.at_add(emb, 16 + MAX_OFFSET + 1, max_log)

        if summary.sample_rate is not None:
            sr_sign, sr_log = log_encode_value(summary.sample_rate)
            emb = backend.at_add(emb, 16 + SAMPLE_RATE_OFFSET, sr_sign)
            emb = backend.at_add(emb, 16 + SAMPLE_RATE_OFFSET + 1, sr_log)
            emb = backend.at_add(emb, 16 + HAS_SAMPLE_RATE_FLAG, 1.0)

        for i, sample in enumerate(summary.samples[:MAX_SIGNAL_VALUES]):
            sign, log_mag = log_encode_value(float(sample))
            offset = 16 + VALUES_OFFSET + 2 * i
            emb = backend.at_add(emb, offset, sign)
            emb = backend.at_add(emb, offset + 1, log_mag)

        return emb

    def decode(self, emb: Any) -> SignalSummary:
        """Decode an embedding back to a signal summary."""
        length = int(round(_scalar(emb[16 + LENGTH_OFFSET]) * MAX_SIGNAL_LEN))
        mean_val = log_decode_value(_scalar(emb[16 + MEAN_OFFSET]), _scalar(emb[16 + MEAN_OFFSET + 1]))
        rms_val = log_decode_value(_scalar(emb[16 + RMS_OFFSET]), _scalar(emb[16 + RMS_OFFSET + 1]))
        energy_val = log_decode_value(_scalar(emb[16 + ENERGY_OFFSET]), _scalar(emb[16 + ENERGY_OFFSET + 1]))
        min_val = log_decode_value(_scalar(emb[16 + MIN_OFFSET]), _scalar(emb[16 + MIN_OFFSET + 1]))
        max_val = log_decode_value(_scalar(emb[16 + MAX_OFFSET]), _scalar(emb[16 + MAX_OFFSET + 1]))

        has_sr = _scalar(emb[16 + HAS_SAMPLE_RATE_FLAG]) > 0.5
        sample_rate = None
        if has_sr:
            sample_rate = log_decode_value(
                _scalar(emb[16 + SAMPLE_RATE_OFFSET]),
                _scalar(emb[16 + SAMPLE_RATE_OFFSET + 1]),
            )

        samples: List[float] = []
        count = min(length, MAX_SIGNAL_VALUES)
        for i in range(count):
            offset = 16 + VALUES_OFFSET + 2 * i
            sample = log_decode_value(_scalar(emb[offset]), _scalar(emb[offset + 1]))
            samples.append(sample)

        return SignalSummary(
            samples=samples,
            length=length,
            sample_rate=sample_rate,
            mean=mean_val,
            rms=rms_val,
            energy=energy_val,
            min_value=min_val,
            max_value=max_val,
        )

    def is_valid(self, emb: Any) -> bool:
        """Check if embedding is a valid signal summary."""
        backend = get_backend()
        tag = emb[0:16]
        return backend.allclose(tag, self.domain_tag, atol=0.1).item()


__all__ = [
    "SignalSummary",
    "SignalEncoder",
    "convolution",
    "moving_average",
    "dft_magnitude",
]
