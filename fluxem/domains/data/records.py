"""
Record Encoder for Data Domain.

Encodes structured records (like dicts/structs) with typed fields.
Supports up to 8 fields with various value types.

Record operations like merge and project are supported.
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


class FieldType(Enum):
    """Types for record fields."""
    NULL = 0
    BOOL = 1
    INT = 2
    FLOAT = 3
    STRING = 4


# Maximum number of fields we can encode
MAX_FIELDS = 8

# Embedding layout within domain-specific region (dims 8-71):
# dims 0:     Number of fields (normalized)
# dims 1:     Schema hash (for quick equality check)
# dims 2-9:   Field name hashes (8 fields)
# dims 10-17: Field types (8 fields, normalized)
# dims 18-49: Field values (8 fields x 4 dims each)
# dims 50-63: Reserved

NUM_FIELDS_OFFSET = 0
SCHEMA_HASH_OFFSET = 1
FIELD_NAMES_OFFSET = 2
FIELD_TYPES_OFFSET = 10
FIELD_VALUES_OFFSET = 18

# Domain tag for data records
DATA_RECORD_TAG = backend.array([0, 0, 0, 0, 1, 0, 0, 1])


class RecordEncoder:
    """
    Encoder for structured records (dict-like objects).

    Encodes records with up to 8 named fields of various types.
    Field names are hashed for compact representation.
    """

    domain_tag = DATA_RECORD_TAG
    domain_name = "data_record"

    def encode(self, record: Dict[str, Any]) -> Any:
        """
        Encode a record (dictionary).

        Args:
            record: Dictionary with string keys and typed values

        Returns:
            128-dim embedding
        """
        backend = get_backend()
        if not record:
            raise ValueError("Cannot encode empty record")

        # Get field names in sorted order for deterministic encoding
        field_names = sorted(record.keys())[:MAX_FIELDS]
        n_fields = len(field_names)

        emb = create_embedding()

        # Domain tag
        emb = backend.at_add(emb, slice(0, 8), self.domain_tag)

        # Number of fields
        emb = backend.at_add(emb, 8 + NUM_FIELDS_OFFSET, n_fields / MAX_FIELDS)

        # Schema hash (hash of field names)
        schema_str = ",".join(field_names)
        schema_hash = (hash(schema_str) % 10000) / 10000.0
        emb = backend.at_add(emb, 8 + SCHEMA_HASH_OFFSET, schema_hash)

        # Encode each field
        for i, name in enumerate(field_names):
            value = record[name]

            # Field name hash
            name_hash = (hash(name) % 10000) / 10000.0
            emb = backend.at_add(emb, 8 + FIELD_NAMES_OFFSET + i, name_hash)

            # Determine and encode field type and value
            ftype, encoded = self._encode_value(value)
            emb = backend.at_add(emb, 8 + FIELD_TYPES_OFFSET + i, ftype.value / 4.0)

            # Store encoded value (4 dims per field)
            base = 8 + FIELD_VALUES_OFFSET + i * 4
            for j, v in enumerate(encoded[:4]):
                emb = backend.at_add(emb, base + j, v)

        return emb

    def _encode_value(self, value: Any) -> Tuple[FieldType, List[float]]:
        """Encode a single field value."""
        if value is None:
            return (FieldType.NULL, [0.0, 0.0, 0.0, 0.0])

        if isinstance(value, bool):
            return (FieldType.BOOL, [1.0 if value else -1.0, 0.0, 0.0, 0.0])

        if isinstance(value, int):
            sign, log_mag = log_encode_value(float(value))
            return (FieldType.INT, [sign, log_mag, 0.0, 0.0])

        if isinstance(value, float):
            sign, log_mag = log_encode_value(value)
            return (FieldType.FLOAT, [sign, log_mag, 0.0, 0.0])

        if isinstance(value, str):
            # Encode string as hash + length
            str_hash = (hash(value) % 10000) / 10000.0
            str_len = min(len(value), 1000) / 1000.0
            return (FieldType.STRING, [str_hash, str_len, 0.0, 0.0])

        # Default: treat as string representation
        str_val = str(value)
        str_hash = (hash(str_val) % 10000) / 10000.0
        return (FieldType.STRING, [str_hash, 0.0, 0.0, 0.0])

    def _decode_value(self, ftype: FieldType, encoded: List[float]) -> Any:
        """Decode a single field value."""
        if ftype == FieldType.NULL:
            return None

        if ftype == FieldType.BOOL:
            return encoded[0] > 0

        if ftype == FieldType.INT:
            val = log_decode_value(encoded[0], encoded[1])
            return int(round(val))

        if ftype == FieldType.FLOAT:
            return log_decode_value(encoded[0], encoded[1])

        if ftype == FieldType.STRING:
            # Cannot fully decode strings, return placeholder
            return f"<string:{encoded[0]:.4f}>"

        return None

    def decode(self, emb: Any) -> Dict[str, Any]:
        """
        Decode embedding to record.

        Note: Field names and string values cannot be fully recovered.
        Returns placeholder names like "field_0", "field_1", etc.

        Returns:
            Dictionary with decoded values
        """
        n_fields = int(round(emb[8 + NUM_FIELDS_OFFSET].item() * MAX_FIELDS))
        n_fields = max(1, min(n_fields, MAX_FIELDS))

        result = {}
        for i in range(n_fields):
            # Use placeholder field name
            name = f"field_{i}"

            # Get field type
            type_val = int(round(emb[8 + FIELD_TYPES_OFFSET + i].item() * 4.0))
            type_val = max(0, min(4, type_val))
            ftype = FieldType(type_val)

            # Get encoded value
            base = 8 + FIELD_VALUES_OFFSET + i * 4
            encoded = [emb[base + j].item() for j in range(4)]

            # Decode value
            result[name] = self._decode_value(ftype, encoded)

        return result

    def is_valid(self, emb: Any) -> bool:
        """Check if embedding is a valid record."""
        backend = get_backend()
        tag = emb[0:8]
        return backend.allclose(tag, self.domain_tag, atol=0.1).item()

    # =========================================================================
    # Record Queries
    # =========================================================================

    def get_num_fields(self, emb: Any) -> int:
        """Get number of fields in the record."""
        return int(round(emb[8 + NUM_FIELDS_OFFSET].item() * MAX_FIELDS))

    def get_schema_hash(self, emb: Any) -> float:
        """Get schema hash for equality checking."""
        return emb[8 + SCHEMA_HASH_OFFSET].item()

    def same_schema(self, emb1: Any, emb2: Any) -> bool:
        """Check if two records have the same schema."""
        hash1 = self.get_schema_hash(emb1)
        hash2 = self.get_schema_hash(emb2)
        n1 = self.get_num_fields(emb1)
        n2 = self.get_num_fields(emb2)
        return n1 == n2 and abs(hash1 - hash2) < 0.0001

    def get_field_type(self, emb: Any, index: int) -> FieldType:
        """Get the type of a specific field."""
        if index < 0 or index >= MAX_FIELDS:
            raise IndexError(f"Field index {index} out of range")

        type_val = int(round(emb[8 + FIELD_TYPES_OFFSET + index].item() * 4.0))
        return FieldType(max(0, min(4, type_val)))

    # =========================================================================
    # Operations
    # =========================================================================

    def merge(self, emb1: Any, emb2: Any) -> Any:
        """
        Merge two records.

        Fields from emb2 override fields from emb1 with same index.
        """
        rec1 = self.decode(emb1)
        rec2 = self.decode(emb2)

        merged = {**rec1, **rec2}
        return self.encode(merged)

    def project(self, emb: Any, indices: List[int]) -> Any:
        """
        Project record to subset of fields by index.

        Args:
            emb: Record embedding
            indices: List of field indices to keep

        Returns:
            New record with only specified fields
        """
        record = self.decode(emb)
        field_names = sorted(record.keys())

        projected = {}
        for i, name in enumerate(field_names):
            if i in indices:
                projected[name] = record[name]

        if not projected:
            # Keep at least one field
            first_name = field_names[0] if field_names else "field_0"
            projected[first_name] = record.get(first_name)

        return self.encode(projected)


# =============================================================================
# Table Encoder (Collection of Records)
# =============================================================================

# Domain tag for data tables
DATA_TABLE_TAG = backend.array([0, 0, 0, 0, 1, 0, 1, 0])

# Maximum number of rows we can encode statistics for
MAX_ROWS = 1000

# Table layout within domain-specific region (dims 8-71):
# dims 0:     Number of rows (normalized, log scale)
# dims 1:     Number of columns (normalized)
# dims 2:     Schema hash
# dims 3-10:  Column types (8 columns)
# dims 11-26: Column statistics (8 columns x 2 dims each: mean/mode, std/cardinality)
# dims 27-34: Column name hashes (8 columns)
# dims 35-63: Reserved / first row sample

TABLE_ROWS_OFFSET = 0
TABLE_COLS_OFFSET = 1
TABLE_SCHEMA_OFFSET = 2
TABLE_COL_TYPES_OFFSET = 3
TABLE_COL_STATS_OFFSET = 11
TABLE_COL_NAMES_OFFSET = 27


class TableEncoder:
    """
    Encoder for tabular data (collection of records).

    Encodes table structure and column statistics.
    Does not store individual row values.
    """

    domain_tag = DATA_TABLE_TAG
    domain_name = "data_table"

    def encode(
        self,
        data: Union[List[Dict[str, Any]], Dict[str, List[Any]]]
    ) -> Any:
        """
        Encode a table.

        Args:
            data: Either list of row dicts or dict of column lists

        Returns:
            128-dim embedding
        """
        # Normalize to column-oriented format
        if isinstance(data, list):
            if not data:
                raise ValueError("Cannot encode empty table")
            columns = self._rows_to_columns(data)
        else:
            columns = data

        if not columns:
            raise ValueError("Cannot encode empty table")

        col_names = sorted(columns.keys())[:MAX_FIELDS]
        n_cols = len(col_names)
        n_rows = max(len(col) for col in columns.values()) if columns else 0

        emb = create_embedding()

        # Domain tag
        emb = backend.at_add(emb, slice(0, 8), self.domain_tag)

        # Number of rows (log scale for large tables)
        if n_rows > 0:
            log_rows = math.log(n_rows + 1) / math.log(MAX_ROWS + 1)
            emb = backend.at_add(emb, 8 + TABLE_ROWS_OFFSET, min(1.0, log_rows))
        else:
            emb = backend.at_add(emb, 8 + TABLE_ROWS_OFFSET, 0.0)

        # Number of columns
        emb = backend.at_add(emb, 8 + TABLE_COLS_OFFSET, n_cols / MAX_FIELDS)

        # Schema hash
        schema_str = ",".join(col_names)
        schema_hash = (hash(schema_str) % 10000) / 10000.0
        emb = backend.at_add(emb, 8 + TABLE_SCHEMA_OFFSET, schema_hash)

        # Encode each column
        for i, name in enumerate(col_names):
            col_data = columns[name]

            # Column name hash
            name_hash = (hash(name) % 10000) / 10000.0
            emb = backend.at_add(emb, 8 + TABLE_COL_NAMES_OFFSET + i, name_hash)

            # Detect column type
            col_type = self._detect_column_type(col_data)
            emb = backend.at_add(emb, 8 + TABLE_COL_TYPES_OFFSET + i, col_type.value / 4.0)

            # Column statistics
            stats = self._compute_column_stats(col_data, col_type)
            base = 8 + TABLE_COL_STATS_OFFSET + i * 2
            emb = backend.at_add(emb, base, stats[0])
            emb = backend.at_add(emb, base + 1, stats[1])

        return emb

    def _rows_to_columns(self, rows: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        """Convert row-oriented data to column-oriented."""
        if not rows:
            return {}

        all_keys = set()
        for row in rows:
            all_keys.update(row.keys())

        columns = {key: [] for key in all_keys}
        for row in rows:
            for key in all_keys:
                columns[key].append(row.get(key))

        return columns

    def _detect_column_type(self, values: List[Any]) -> FieldType:
        """Detect column type from values."""
        non_null = [v for v in values if v is not None]
        if not non_null:
            return FieldType.NULL

        sample = non_null[0]
        if isinstance(sample, bool):
            return FieldType.BOOL
        elif isinstance(sample, int):
            return FieldType.INT
        elif isinstance(sample, float):
            return FieldType.FLOAT
        else:
            return FieldType.STRING

    def _compute_column_stats(
        self,
        values: List[Any],
        col_type: FieldType
    ) -> Tuple[float, float]:
        """Compute summary statistics for a column."""
        non_null = [v for v in values if v is not None]
        if not non_null:
            return (0.0, 0.0)

        if col_type in (FieldType.INT, FieldType.FLOAT):
            # Numeric: mean and normalized std
            numeric = [float(v) for v in non_null]
            mean = sum(numeric) / len(numeric)
            if len(numeric) > 1:
                variance = sum((x - mean) ** 2 for x in numeric) / (len(numeric) - 1)
                std = math.sqrt(variance) if variance > 0 else 0.0
            else:
                std = 0.0

            # Normalize to reasonable range
            mean_norm = math.tanh(mean / 1000.0)  # Compress to [-1, 1]
            std_norm = min(1.0, std / 100.0)  # Normalize std
            return (mean_norm, std_norm)

        elif col_type == FieldType.BOOL:
            # Boolean: fraction of True values
            true_count = sum(1 for v in non_null if v)
            frac = true_count / len(non_null)
            return (frac, 0.0)

        else:
            # String/categorical: cardinality
            cardinality = len(set(str(v) for v in non_null))
            card_norm = min(1.0, cardinality / len(non_null))
            return (card_norm, 0.0)

    def decode(self, emb: Any) -> Dict[str, Any]:
        """
        Decode embedding to table metadata.

        Note: Individual rows cannot be recovered.

        Returns:
            Dictionary with table metadata
        """
        log_rows = emb[8 + TABLE_ROWS_OFFSET].item()
        n_rows = int(math.exp(log_rows * math.log(MAX_ROWS + 1)) - 1)
        n_rows = max(0, n_rows)

        n_cols = int(round(emb[8 + TABLE_COLS_OFFSET].item() * MAX_FIELDS))
        n_cols = max(1, min(n_cols, MAX_FIELDS))

        schema_hash = emb[8 + TABLE_SCHEMA_OFFSET].item()

        columns = []
        for i in range(n_cols):
            type_val = int(round(emb[8 + TABLE_COL_TYPES_OFFSET + i].item() * 4.0))
            col_type = FieldType(max(0, min(4, type_val)))

            base = 8 + TABLE_COL_STATS_OFFSET + i * 2
            stat1 = emb[base].item()
            stat2 = emb[base + 1].item()

            name_hash = emb[8 + TABLE_COL_NAMES_OFFSET + i].item()

            columns.append({
                "name": f"col_{i}",
                "type": col_type.name,
                "name_hash": name_hash,
                "stat1": stat1,
                "stat2": stat2,
            })

        return {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "schema_hash": schema_hash,
            "columns": columns,
        }

    def is_valid(self, emb: Any) -> bool:
        """Check if embedding is a valid table."""
        backend = get_backend()
        tag = emb[0:8]
        return backend.allclose(tag, self.domain_tag, atol=0.1).item()

    # =========================================================================
    # Table Queries
    # =========================================================================

    def get_num_rows(self, emb: Any) -> int:
        """Get approximate number of rows."""
        log_rows = emb[8 + TABLE_ROWS_OFFSET].item()
        return max(0, int(math.exp(log_rows * math.log(MAX_ROWS + 1)) - 1))

    def get_num_cols(self, emb: Any) -> int:
        """Get number of columns."""
        return int(round(emb[8 + TABLE_COLS_OFFSET].item() * MAX_FIELDS))

    def get_schema_hash(self, emb: Any) -> float:
        """Get schema hash."""
        return emb[8 + TABLE_SCHEMA_OFFSET].item()

    def same_schema(self, emb1: Any, emb2: Any) -> bool:
        """Check if two tables have the same schema."""
        hash1 = self.get_schema_hash(emb1)
        hash2 = self.get_schema_hash(emb2)
        n1 = self.get_num_cols(emb1)
        n2 = self.get_num_cols(emb2)
        return n1 == n2 and abs(hash1 - hash2) < 0.0001

    def get_column_type(self, emb: Any, index: int) -> FieldType:
        """Get type of column at index."""
        if index < 0 or index >= MAX_FIELDS:
            raise IndexError(f"Column index {index} out of range")

        type_val = int(round(emb[8 + TABLE_COL_TYPES_OFFSET + index].item() * 4.0))
        return FieldType(max(0, min(4, type_val)))
