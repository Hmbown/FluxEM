"""
Array Encoder for Data Domain.

Encodes typed arrays with statistical summaries and structure information.
Supports numeric, boolean, and categorical arrays up to 64 elements.

Array concatenation and slicing are supported operations.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
from ...backend import get_backend

from ...core.base import (
    DOMAIN_TAGS,
    EMBEDDING_DIM,
    EPSILON,
    create_embedding,
    log_encode_value,
    log_decode_value,
)

# Get backend at module level
backend = get_backend()


class ArrayDType(Enum):
    """Data types for arrays."""
    FLOAT = 0
    INT = 1
    BOOL = 2
    CATEGORICAL = 3
    STRING = 4  # Encoded as categorical with hash


# Maximum array length we can encode with full fidelity
MAX_ARRAY_LEN = 32

# Embedding layout within domain-specific region (dims 8-71):
# dims 0:     DType (normalized 0-4)
# dims 1:     Length (normalized, len/MAX)
# dims 2-3:   Min value (sign, log|min|)
# dims 4-5:   Max value (sign, log|max|)
# dims 6-7:   Mean (sign, log|mean|)
# dims 8-9:   Std dev (sign, log|std|)
# dims 10-11: Sum (sign, log|sum|)
# dims 12:    Has nulls flag
# dims 13:    Is sorted flag
# dims 14:    Is unique flag
# dims 15:    Is constant flag
# dims 16-47: First 16 values (pairs of sign, log|val|)
# dims 48-63: Histogram/distribution (8 bins with 2 dims each)

DTYPE_OFFSET = 0
LENGTH_OFFSET = 1
MIN_OFFSET = 2
MAX_OFFSET = 4
MEAN_OFFSET = 6
STD_OFFSET = 8
SUM_OFFSET = 10
HAS_NULLS_FLAG = 12
IS_SORTED_FLAG = 13
IS_UNIQUE_FLAG = 14
IS_CONSTANT_FLAG = 15
VALUES_OFFSET = 16
HISTOGRAM_OFFSET = 48

# Domain tag for data arrays
DATA_ARRAY_TAG = backend.array([0, 0, 0, 0, 1, 0, 0, 0])


class ArrayEncoder:
    """
    Encoder for typed arrays.

    Encodes arrays with statistical summaries (mean, std, min, max)
    and stores first N values for reconstruction of small arrays.
    """

    domain_tag = DATA_ARRAY_TAG
    domain_name = "data_array"

    def encode(
        self,
        arr: Union[List, Tuple, backend.array],
        dtype: Optional[ArrayDType] = None
    ) -> Any:
        """
        Encode an array.

        Args:
            arr: Array as list, tuple, or mlx array
            dtype: Optional dtype hint (auto-detected if None)

        Returns:
            128-dim embedding
        """
        # Convert array-like objects to list
        if hasattr(arr, 'tolist'):
            arr = arr.tolist()
        arr = list(arr)

        n = len(arr)
        if n == 0:
            raise ValueError("Cannot encode empty array")

        # Auto-detect dtype if not provided
        if dtype is None:
            dtype = self._detect_dtype(arr)

        emb = create_embedding()

        # Domain tag
        emb = backend.at_add(emb, slice(0, 8), self.domain_tag)

        # DType
        emb = backend.at_add(emb, 8 + DTYPE_OFFSET, dtype.value / 4.0)

        # Length
        emb = backend.at_add(emb, 8 + LENGTH_OFFSET, min(n, MAX_ARRAY_LEN * 2) / (MAX_ARRAY_LEN * 2))

        # For numeric arrays, compute statistics
        if dtype in (ArrayDType.FLOAT, ArrayDType.INT):
            numeric = [float(x) for x in arr if x is not None]
            if numeric:
                emb = self._encode_numeric_stats(emb, numeric)
        elif dtype == ArrayDType.BOOL:
            # Convert bools to 0/1 for statistics
            numeric = [1.0 if x else 0.0 for x in arr if x is not None]
            if numeric:
                emb = self._encode_numeric_stats(emb, numeric)

        # Encode first N values
        for i, val in enumerate(arr[:16]):
            if val is None:
                emb = backend.at_add(emb, 8 + VALUES_OFFSET + 2*i, 0.0)
                emb = backend.at_add(emb, 8 + VALUES_OFFSET + 2*i + 1, -100.0)
            elif dtype in (ArrayDType.FLOAT, ArrayDType.INT):
                sign, log_mag = log_encode_value(float(val))
                emb = backend.at_add(emb, 8 + VALUES_OFFSET + 2*i, sign)
                emb = backend.at_add(emb, 8 + VALUES_OFFSET + 2*i + 1, log_mag)
            elif dtype == ArrayDType.BOOL:
                emb = backend.at_add(emb, 8 + VALUES_OFFSET + 2*i, 1.0 if val else -1.0)
                emb = backend.at_add(emb, 8 + VALUES_OFFSET + 2*i + 1, 0.0)
            else:
                # Categorical/string - use hash
                hash_val = hash(str(val)) % 10000 / 10000.0
                emb = backend.at_add(emb, 8 + VALUES_OFFSET + 2*i, 1.0)
                emb = backend.at_add(emb, 8 + VALUES_OFFSET + 2*i + 1, hash_val)

        # Flags
        has_nulls = any(x is None for x in arr)
        emb = backend.at_add(emb, 8 + HAS_NULLS_FLAG, 1.0 if has_nulls else 0.0)

        # Check if sorted (for numeric)
        if dtype in (ArrayDType.FLOAT, ArrayDType.INT):
            non_null = [x for x in arr if x is not None]
            is_sorted = non_null == sorted(non_null)
            emb = backend.at_add(emb, 8 + IS_SORTED_FLAG, 1.0 if is_sorted else 0.0)

            is_unique = len(non_null) == len(set(non_null))
            emb = backend.at_add(emb, 8 + IS_UNIQUE_FLAG, 1.0 if is_unique else 0.0)

            is_constant = len(set(non_null)) <= 1
            emb = backend.at_add(emb, 8 + IS_CONSTANT_FLAG, 1.0 if is_constant else 0.0)

        return emb

    def _detect_dtype(self, arr: List) -> ArrayDType:
        """Auto-detect array dtype from values."""
        non_null = [x for x in arr if x is not None]
        if not non_null:
            return ArrayDType.FLOAT

        sample = non_null[0]
        if isinstance(sample, bool):
            return ArrayDType.BOOL
        elif isinstance(sample, int):
            return ArrayDType.INT
        elif isinstance(sample, float):
            return ArrayDType.FLOAT
        elif isinstance(sample, str):
            return ArrayDType.STRING
        else:
            return ArrayDType.CATEGORICAL

    def _encode_numeric_stats(self, emb: Any, values: List[float]) -> Any:
        """Encode statistical summary of numeric values."""
        backend = get_backend()
        n = len(values)
        min_val = min(values)
        max_val = max(values)
        mean_val = sum(values) / n
        sum_val = sum(values)

        # Variance and std
        if n > 1:
            variance = sum((x - mean_val) ** 2 for x in values) / (n - 1)
            std_val = math.sqrt(variance) if variance > 0 else 0.0
        else:
            std_val = 0.0

        # Encode min
        sign, log_mag = log_encode_value(min_val)
        emb = backend.at_add(emb, 8 + MIN_OFFSET, sign)
        emb = backend.at_add(emb, 8 + MIN_OFFSET + 1, log_mag)

        # Encode max
        sign, log_mag = log_encode_value(max_val)
        emb = backend.at_add(emb, 8 + MAX_OFFSET, sign)
        emb = backend.at_add(emb, 8 + MAX_OFFSET + 1, log_mag)

        # Encode mean
        sign, log_mag = log_encode_value(mean_val)
        emb = backend.at_add(emb, 8 + MEAN_OFFSET, sign)
        emb = backend.at_add(emb, 8 + MEAN_OFFSET + 1, log_mag)

        # Encode std
        sign, log_mag = log_encode_value(std_val)
        emb = backend.at_add(emb, 8 + STD_OFFSET, sign)
        emb = backend.at_add(emb, 8 + STD_OFFSET + 1, log_mag)

        # Encode sum
        sign, log_mag = log_encode_value(sum_val)
        emb = backend.at_add(emb, 8 + SUM_OFFSET, sign)
        emb = backend.at_add(emb, 8 + SUM_OFFSET + 1, log_mag)

        # Histogram (8 bins)
        if max_val > min_val:
            bin_width = (max_val - min_val) / 8.0
            bins = [0] * 8
            for v in values:
                bin_idx = min(7, int((v - min_val) / bin_width))
                bins[bin_idx] += 1
            for i, count in enumerate(bins):
                emb = backend.at_add(emb, 8 + HISTOGRAM_OFFSET + 2*i, 1.0)
                emb = backend.at_add(emb, 8 + HISTOGRAM_OFFSET + 2*i + 1, count / n)

        return emb

    def decode(self, emb: Any) -> Tuple[List, ArrayDType]:
        """
        Decode embedding to array and dtype.

        Note: Only first 16 values can be exactly reconstructed.

        Returns:
            Tuple of (values list, dtype)
        """
        dtype_val = int(round(emb[8 + DTYPE_OFFSET].item() * 4.0))
        dtype_val = max(0, min(4, dtype_val))
        dtype = ArrayDType(dtype_val)

        length = int(round(emb[8 + LENGTH_OFFSET].item() * MAX_ARRAY_LEN * 2))
        length = max(1, min(length, 16))  # Can only decode first 16

        values = []
        for i in range(length):
            sign = emb[8 + VALUES_OFFSET + 2*i].item()
            log_mag = emb[8 + VALUES_OFFSET + 2*i + 1].item()

            if abs(sign) < 0.5:
                values.append(None)
            elif dtype == ArrayDType.BOOL:
                values.append(sign > 0)
            elif dtype == ArrayDType.INT:
                values.append(int(round(log_decode_value(sign, log_mag))))
            else:
                values.append(log_decode_value(sign, log_mag))

        return (values, dtype)

    def is_valid(self, emb: Any) -> bool:
        """Check if embedding is a valid array."""
        backend = get_backend()
        tag = emb[0:8]
        return backend.allclose(tag, self.domain_tag, atol=0.1).item()

    # =========================================================================
    # Statistical Queries
    # =========================================================================

    def get_length(self, emb: Any) -> int:
        """Get array length."""
        return int(round(emb[8 + LENGTH_OFFSET].item() * MAX_ARRAY_LEN * 2))

    def get_dtype(self, emb: Any) -> ArrayDType:
        """Get array dtype."""
        dtype_val = int(round(emb[8 + DTYPE_OFFSET].item() * 4.0))
        return ArrayDType(max(0, min(4, dtype_val)))

    def get_min(self, emb: Any) -> Optional[float]:
        """Get minimum value."""
        sign = emb[8 + MIN_OFFSET].item()
        if abs(sign) < 0.5:
            return None
        log_mag = emb[8 + MIN_OFFSET + 1].item()
        return log_decode_value(sign, log_mag)

    def get_max(self, emb: Any) -> Optional[float]:
        """Get maximum value."""
        sign = emb[8 + MAX_OFFSET].item()
        if abs(sign) < 0.5:
            return None
        log_mag = emb[8 + MAX_OFFSET + 1].item()
        return log_decode_value(sign, log_mag)

    def get_mean(self, emb: Any) -> Optional[float]:
        """Get mean value."""
        sign = emb[8 + MEAN_OFFSET].item()
        if abs(sign) < 0.5:
            return None
        log_mag = emb[8 + MEAN_OFFSET + 1].item()
        return log_decode_value(sign, log_mag)

    def get_std(self, emb: Any) -> Optional[float]:
        """Get standard deviation."""
        sign = emb[8 + STD_OFFSET].item()
        log_mag = emb[8 + STD_OFFSET + 1].item()
        return log_decode_value(sign, log_mag)

    def get_sum(self, emb: Any) -> Optional[float]:
        """Get sum."""
        sign = emb[8 + SUM_OFFSET].item()
        if abs(sign) < 0.5:
            return None
        log_mag = emb[8 + SUM_OFFSET + 1].item()
        return log_decode_value(sign, log_mag)

    def has_nulls(self, emb: Any) -> bool:
        """Check if array has null values."""
        return emb[8 + HAS_NULLS_FLAG].item() > 0.5

    def is_sorted(self, emb: Any) -> bool:
        """Check if array is sorted."""
        return emb[8 + IS_SORTED_FLAG].item() > 0.5

    def is_unique(self, emb: Any) -> bool:
        """Check if all values are unique."""
        return emb[8 + IS_UNIQUE_FLAG].item() > 0.5

    def is_constant(self, emb: Any) -> bool:
        """Check if all values are the same."""
        return emb[8 + IS_CONSTANT_FLAG].item() > 0.5

    # =========================================================================
    # Operations
    # =========================================================================

    def concatenate(self, emb1: Any, emb2: Any) -> Any:
        """Concatenate two arrays."""
        arr1, dtype1 = self.decode(emb1)
        arr2, dtype2 = self.decode(emb2)

        if dtype1 != dtype2:
            # Promote to more general type
            dtype = ArrayDType.FLOAT
        else:
            dtype = dtype1

        return self.encode(arr1 + arr2, dtype)

    def slice(self, emb: Any, start: int, end: int) -> Any:
        """Slice array from start to end."""
        arr, dtype = self.decode(emb)
        return self.encode(arr[start:end], dtype)

    def filter_nulls(self, emb: Any) -> Any:
        """Remove null values from array."""
        arr, dtype = self.decode(emb)
        filtered = [x for x in arr if x is not None]
        return self.encode(filtered, dtype)
