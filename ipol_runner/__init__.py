"""IPOL Runner - Unified CLI for IPOL image processing methods."""
from .base import IPOLMethod, MethodResult
from .registry import register, get_method, list_methods, get_all_methods

__version__ = "0.1.0"
__all__ = [
    "IPOLMethod",
    "MethodResult",
    "register",
    "get_method",
    "list_methods",
    "get_all_methods",
]

# Import methods to register them
from . import methods  # noqa: F401
