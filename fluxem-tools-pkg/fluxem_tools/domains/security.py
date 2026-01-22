"""Security domain - RBAC/ABAC permissions, access control.

This module provides deterministic security/permission computations.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ..registry import ToolSpec, ToolRegistry


# =============================================================================
# Core Functions
# =============================================================================

def encode_permissions(permissions: List[str]) -> Set[str]:
    """Encode a list of permissions into a set.

    Args:
        permissions: List of permission strings

    Returns:
        Set of normalized permissions
    """
    return {p.lower().strip() for p in permissions}


def check_access(user_permissions: List[str], required_permissions: List[str]) -> bool:
    """Check if user has all required permissions (RBAC check).

    Args:
        user_permissions: List of permissions the user has
        required_permissions: List of permissions required

    Returns:
        True if user has all required permissions
    """
    user_set = encode_permissions(user_permissions)
    required_set = encode_permissions(required_permissions)
    return required_set <= user_set


def has_permission(permissions: List[str], permission: str) -> bool:
    """Check if a specific permission exists in a permission set.

    Args:
        permissions: List of permissions
        permission: Permission to check

    Returns:
        True if permission exists
    """
    perm_set = encode_permissions(permissions)
    return permission.lower().strip() in perm_set


def combine_roles(roles: List[List[str]]) -> List[str]:
    """Combine multiple roles/permission sets into one (union).

    Args:
        roles: List of permission lists

    Returns:
        Combined permission list
    """
    combined = set()
    for role in roles:
        combined.update(encode_permissions(role))
    return sorted(combined)


def permission_diff(perms_a: List[str], perms_b: List[str]) -> Dict[str, List[str]]:
    """Find permission differences between two sets.

    Args:
        perms_a: First permission set
        perms_b: Second permission set

    Returns:
        Dict with 'only_in_a', 'only_in_b', 'common' keys
    """
    set_a = encode_permissions(perms_a)
    set_b = encode_permissions(perms_b)

    return {
        "only_in_a": sorted(set_a - set_b),
        "only_in_b": sorted(set_b - set_a),
        "common": sorted(set_a & set_b),
    }


def check_role_hierarchy(user_roles: List[str], required_role: str,
                         hierarchy: Dict[str, List[str]]) -> bool:
    """Check if user has required role considering role hierarchy.

    Args:
        user_roles: Roles the user has
        required_role: Role required for access
        hierarchy: Dict mapping roles to their parent roles

    Returns:
        True if user has required role (directly or through inheritance)
    """
    required_lower = required_role.lower().strip()
    user_roles_lower = {r.lower().strip() for r in user_roles}

    # Direct match
    if required_lower in user_roles_lower:
        return True

    # Check hierarchy - find all roles that grant the required role
    def get_all_granted_roles(role: str, visited: Set[str] = None) -> Set[str]:
        if visited is None:
            visited = set()
        if role in visited:
            return set()
        visited.add(role)

        granted = {role}
        for parent in hierarchy.get(role, []):
            granted.update(get_all_granted_roles(parent.lower(), visited))
        return granted

    # Check if any user role grants the required role
    for user_role in user_roles_lower:
        granted = get_all_granted_roles(user_role)
        if required_lower in granted:
            return True

    return False


def validate_permission_string(permission: str) -> Dict[str, Any]:
    """Validate and parse a permission string.

    Supports formats like 'resource:action' or 'resource:action:scope'.

    Args:
        permission: Permission string to validate

    Returns:
        Dict with 'valid', 'resource', 'action', 'scope' keys
    """
    permission = permission.strip()
    if not permission:
        return {"valid": False, "error": "Empty permission string"}

    parts = permission.split(':')

    if len(parts) == 1:
        return {"valid": True, "resource": parts[0], "action": "*", "scope": "*"}
    elif len(parts) == 2:
        return {"valid": True, "resource": parts[0], "action": parts[1], "scope": "*"}
    elif len(parts) == 3:
        return {"valid": True, "resource": parts[0], "action": parts[1], "scope": parts[2]}
    else:
        return {"valid": False, "error": "Too many colons in permission string"}


# =============================================================================
# Argument Parsing
# =============================================================================

def _parse_permissions(args) -> List[str]:
    if isinstance(args, dict):
        perms = args.get("permissions", args.get("perms"))
        if perms is not None:
            return list(perms)
        return list(list(args.values())[0])
    if isinstance(args, (list, tuple)):
        return list(args)
    if isinstance(args, str):
        return [args]
    raise ValueError(f"Cannot parse permissions: {args}")


def _parse_check_access(args) -> Tuple[List[str], List[str]]:
    if isinstance(args, dict):
        user_perms = args.get("user_permissions", args.get("user_perms", args.get("user")))
        required = args.get("required_permissions", args.get("required", args.get("needed")))
        return list(user_perms), list(required)
    if isinstance(args, (list, tuple)) and len(args) >= 2:
        return list(args[0]), list(args[1])
    raise ValueError(f"Cannot parse check_access args: {args}")


def _parse_has_permission(args) -> Tuple[List[str], str]:
    if isinstance(args, dict):
        perms = args.get("permissions", args.get("perms"))
        perm = args.get("permission", args.get("perm", args.get("check")))
        return list(perms), str(perm)
    if isinstance(args, (list, tuple)) and len(args) >= 2:
        return list(args[0]), str(args[1])
    raise ValueError(f"Cannot parse has_permission args: {args}")


def _parse_combine_roles(args) -> List[List[str]]:
    if isinstance(args, dict):
        roles = args.get("roles", list(args.values())[0])
        return [list(r) for r in roles]
    if isinstance(args, (list, tuple)):
        return [list(r) for r in args]
    raise ValueError(f"Cannot parse combine_roles args: {args}")


# =============================================================================
# Tool Registration
# =============================================================================

def register_tools(registry: ToolRegistry) -> None:
    """Register security tools in the registry."""

    registry.register(ToolSpec(
        name="security_encode_permissions",
        function=lambda args: len(encode_permissions(_parse_permissions(args))),
        description="Encodes a list of permissions and returns the count.",
        parameters={
            "type": "object",
            "properties": {
                "permissions": {"type": "array", "items": {"type": "string"}, "description": "List of permissions"}
            },
            "required": ["permissions"]
        },
        returns="Number of unique permissions",
        examples=[
            {"input": {"permissions": ["read", "write", "read"]}, "output": 2},
        ],
        domain="security",
        tags=["permissions", "encode", "rbac"],
    ))

    registry.register(ToolSpec(
        name="security_check_access",
        function=lambda args: check_access(*_parse_check_access(args)),
        description="Checks if user has all required permissions (RBAC check).",
        parameters={
            "type": "object",
            "properties": {
                "user_permissions": {"type": "array", "items": {"type": "string"}, "description": "User's permissions"},
                "required": {"type": "array", "items": {"type": "string"}, "description": "Required permissions"},
            },
            "required": ["user_permissions", "required"]
        },
        returns="Boolean: True if user has all required permissions",
        examples=[
            {"input": {"user_permissions": ["read", "write"], "required": ["read"]}, "output": True},
            {"input": {"user_permissions": ["read"], "required": ["read", "write"]}, "output": False},
        ],
        domain="security",
        tags=["access", "rbac", "authorization"],
    ))

    registry.register(ToolSpec(
        name="security_has_permission",
        function=lambda args: has_permission(*_parse_has_permission(args)),
        description="Checks if a specific permission exists in a permission set.",
        parameters={
            "type": "object",
            "properties": {
                "permissions": {"type": "array", "items": {"type": "string"}, "description": "Permission set"},
                "permission": {"type": "string", "description": "Permission to check"},
            },
            "required": ["permissions", "permission"]
        },
        returns="Boolean: True if permission exists",
        examples=[
            {"input": {"permissions": ["read", "write", "admin"], "permission": "admin"}, "output": True},
        ],
        domain="security",
        tags=["permission", "check", "rbac"],
    ))

    registry.register(ToolSpec(
        name="security_combine_roles",
        function=lambda args: len(set().union(*[encode_permissions(r) for r in _parse_combine_roles(args)])),
        description="Combines multiple roles/permission sets into one (union) and returns count.",
        parameters={
            "type": "object",
            "properties": {
                "roles": {"type": "array", "items": {"type": "array"}, "description": "List of role permission lists"}
            },
            "required": ["roles"]
        },
        returns="Total unique permissions count",
        examples=[
            {"input": {"roles": [["read"], ["write", "delete"]]}, "output": 3},
        ],
        domain="security",
        tags=["roles", "combine", "union"],
    ))
