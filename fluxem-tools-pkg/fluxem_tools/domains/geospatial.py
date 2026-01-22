"""Geospatial domain - geographic calculations.

This module provides deterministic geographic computations.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

from ..registry import ToolSpec, ToolRegistry


# =============================================================================
# Constants
# =============================================================================

R_EARTH = 6371000  # Earth's radius in meters


# =============================================================================
# Core Functions
# =============================================================================

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two points in meters.

    Uses the Haversine formula.

    Args:
        lat1, lon1: First point (degrees)
        lat2, lon2: Second point (degrees)

    Returns:
        Distance in meters
    """
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R_EARTH * c


def geo_midpoint(lat1: float, lon1: float, lat2: float, lon2: float) -> Tuple[float, float]:
    """Calculate geographic midpoint between two coordinates.

    Args:
        lat1, lon1: First point (degrees)
        lat2, lon2: Second point (degrees)

    Returns:
        (lat, lon) of midpoint in degrees
    """
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    lon1_rad = math.radians(lon1)
    dlon = math.radians(lon2 - lon1)

    bx = math.cos(lat2_rad) * math.cos(dlon)
    by = math.cos(lat2_rad) * math.sin(dlon)

    lat_mid = math.atan2(
        math.sin(lat1_rad) + math.sin(lat2_rad),
        math.sqrt((math.cos(lat1_rad) + bx) ** 2 + by ** 2)
    )
    lon_mid = lon1_rad + math.atan2(by, math.cos(lat1_rad) + bx)

    return (round(math.degrees(lat_mid), 6), round(math.degrees(lon_mid), 6))


def initial_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate initial bearing (azimuth) from point 1 to point 2.

    Args:
        lat1, lon1: Starting point (degrees)
        lat2, lon2: Ending point (degrees)

    Returns:
        Bearing in degrees (0-360, where 0=North, 90=East)
    """
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)

    x = math.sin(dlon) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)

    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360


def destination_point(lat: float, lon: float, bearing: float, distance: float) -> Tuple[float, float]:
    """Calculate destination point given start, bearing, and distance.

    Args:
        lat, lon: Starting point (degrees)
        bearing: Initial bearing in degrees
        distance: Distance in meters

    Returns:
        (lat, lon) of destination in degrees
    """
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    bearing_rad = math.radians(bearing)
    angular_distance = distance / R_EARTH

    lat2 = math.asin(
        math.sin(lat_rad) * math.cos(angular_distance) +
        math.cos(lat_rad) * math.sin(angular_distance) * math.cos(bearing_rad)
    )

    lon2 = lon_rad + math.atan2(
        math.sin(bearing_rad) * math.sin(angular_distance) * math.cos(lat_rad),
        math.cos(angular_distance) - math.sin(lat_rad) * math.sin(lat2)
    )

    return (round(math.degrees(lat2), 6), round(math.degrees(lon2), 6))


def point_in_polygon(lat: float, lon: float, polygon: List[Tuple[float, float]]) -> bool:
    """Check if a point is inside a polygon using ray casting.

    Args:
        lat, lon: Point to check
        polygon: List of (lat, lon) vertices

    Returns:
        True if point is inside polygon
    """
    n = len(polygon)
    inside = False

    j = n - 1
    for i in range(n):
        lat_i, lon_i = polygon[i]
        lat_j, lon_j = polygon[j]

        if ((lon_i > lon) != (lon_j > lon)) and \
           (lat < (lat_j - lat_i) * (lon - lon_i) / (lon_j - lon_i) + lat_i):
            inside = not inside

        j = i

    return inside


def bounding_box(points: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    """Calculate bounding box of a set of points.

    Args:
        points: List of (lat, lon) tuples

    Returns:
        (min_lat, min_lon, max_lat, max_lon)
    """
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    return (min(lats), min(lons), max(lats), max(lons))


# =============================================================================
# Argument Parsing
# =============================================================================

def _parse_two_coords(args) -> Tuple[float, float, float, float]:
    if isinstance(args, dict):
        # Try different key patterns
        if "coord1" in args and "coord2" in args:
            c1, c2 = args["coord1"], args["coord2"]
            return float(c1[0]), float(c1[1]), float(c2[0]), float(c2[1])
        if "lat1" in args:
            return float(args["lat1"]), float(args["lon1"]), float(args["lat2"]), float(args["lon2"])
    if isinstance(args, (list, tuple)) and len(args) >= 2:
        if isinstance(args[0], (list, tuple)):
            c1, c2 = args[0], args[1]
            return float(c1[0]), float(c1[1]), float(c2[0]), float(c2[1])
    raise ValueError(f"Cannot parse two coordinates: {args}")


def _parse_destination(args) -> Tuple[float, float, float, float]:
    if isinstance(args, dict):
        lat = float(args.get("lat", args.get("latitude")))
        lon = float(args.get("lon", args.get("longitude")))
        bearing = float(args.get("bearing", args.get("azimuth")))
        distance = float(args.get("distance", args.get("d")))
        return lat, lon, bearing, distance
    if isinstance(args, (list, tuple)) and len(args) >= 4:
        return float(args[0]), float(args[1]), float(args[2]), float(args[3])
    raise ValueError(f"Cannot parse destination args: {args}")


# =============================================================================
# Tool Registration
# =============================================================================

def register_tools(registry: ToolRegistry) -> None:
    """Register geospatial tools in the registry."""

    registry.register(ToolSpec(
        name="geo_distance",
        function=lambda args: haversine_distance(*_parse_two_coords(args)),
        description="Computes great-circle distance between two geographic points in meters.",
        parameters={
            "type": "object",
            "properties": {
                "coord1": {"type": "array", "description": "[lat, lon] of first point"},
                "coord2": {"type": "array", "description": "[lat, lon] of second point"},
            },
            "required": ["coord1", "coord2"]
        },
        returns="Distance in meters",
        examples=[
            {"input": {"coord1": [40.7128, -74.0060], "coord2": [34.0522, -118.2437]}, "output": 3935746.25},
        ],
        domain="geospatial",
        tags=["distance", "haversine", "great circle"],
    ))

    registry.register(ToolSpec(
        name="geo_midpoint",
        function=lambda args: geo_midpoint(*_parse_two_coords(args)),
        description="Computes geographic midpoint between two coordinates.",
        parameters={
            "type": "object",
            "properties": {
                "coord1": {"type": "array", "description": "[lat, lon] of first point"},
                "coord2": {"type": "array", "description": "[lat, lon] of second point"},
            },
            "required": ["coord1", "coord2"]
        },
        returns="Midpoint as (lat, lon)",
        examples=[
            {"input": {"coord1": [40.7128, -74.0060], "coord2": [34.0522, -118.2437]}, "output": [39.5, -97.5]},
        ],
        domain="geospatial",
        tags=["midpoint", "center"],
    ))

    registry.register(ToolSpec(
        name="geo_bearing",
        function=lambda args: initial_bearing(*_parse_two_coords(args)),
        description="Computes initial bearing (azimuth) from point 1 to point 2 in degrees.",
        parameters={
            "type": "object",
            "properties": {
                "coord1": {"type": "array", "description": "[lat, lon] of starting point"},
                "coord2": {"type": "array", "description": "[lat, lon] of ending point"},
            },
            "required": ["coord1", "coord2"]
        },
        returns="Bearing in degrees (0-360, 0=North, 90=East)",
        examples=[
            {"input": {"coord1": [40.7128, -74.0060], "coord2": [34.0522, -118.2437]}, "output": 273.4},
        ],
        domain="geospatial",
        tags=["bearing", "azimuth", "direction"],
    ))

    registry.register(ToolSpec(
        name="geo_destination",
        function=lambda args: destination_point(*_parse_destination(args)),
        description="Computes destination point given start, bearing, and distance.",
        parameters={
            "type": "object",
            "properties": {
                "lat": {"type": "number", "description": "Starting latitude"},
                "lon": {"type": "number", "description": "Starting longitude"},
                "bearing": {"type": "number", "description": "Bearing in degrees"},
                "distance": {"type": "number", "description": "Distance in meters"},
            },
            "required": ["lat", "lon", "bearing", "distance"]
        },
        returns="Destination as (lat, lon)",
        examples=[
            {"input": {"lat": 40.7128, "lon": -74.0060, "bearing": 270, "distance": 100000}, "output": [40.63, -75.32]},
        ],
        domain="geospatial",
        tags=["destination", "travel"],
    ))
