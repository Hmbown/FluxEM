"""Math domain: Extended number systems."""

from .complex import ComplexEncoder
from .rational import RationalEncoder
from .polynomial import PolynomialEncoder
from .vector import VectorEncoder
from .matrix import MatrixEncoder
from .arithmetic import ArithmeticEncoder, encode_number, decode_number, compute
from .datetime_encoder import (
    DateTimeEncoder, 
    DurationEncoder,
    encode_date,
    encode_datetime,
    encode_duration,
    days_between,
)

__all__ = [
    "ComplexEncoder",
    "RationalEncoder", 
    "PolynomialEncoder",
    "VectorEncoder",
    "MatrixEncoder",
    "ArithmeticEncoder",
    "encode_number",
    "decode_number",
    "compute",
    "DateTimeEncoder",
    "DurationEncoder",
    "encode_date",
    "encode_datetime",
    "encode_duration",
    "days_between",
]
