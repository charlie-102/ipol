"""Method registry for IPOL methods."""
from typing import Dict, List, Optional, Type
from .base import IPOLMethod


_registry: Dict[str, Type[IPOLMethod]] = {}


def register(cls: Type[IPOLMethod]) -> Type[IPOLMethod]:
    """Decorator to register a method class.

    Usage:
        @register
        class MyMethod(IPOLMethod):
            name = "my_method"
            ...
    """
    instance = cls()
    _registry[instance.name] = cls
    return cls


def get_method(name: str) -> Optional[IPOLMethod]:
    """Get a method instance by name."""
    cls = _registry.get(name)
    return cls() if cls else None


def list_methods() -> List[str]:
    """List all registered method names."""
    return list(_registry.keys())


def get_all_methods() -> Dict[str, IPOLMethod]:
    """Get all method instances."""
    return {name: cls() for name, cls in _registry.items()}
