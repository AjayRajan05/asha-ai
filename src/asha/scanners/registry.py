# Copyright 2026 Ajay Rajan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Thread-safe ScannerRegistry for registering custom PII and sensitive token scanners.
"""

from __future__ import annotations

import functools
import threading
from typing import Any, Callable, Dict, List, Optional, Type, Union

from .base import BaseScanner


class ScannerRegistry:
    """
    Global thread-safe registry for custom ASHA scanner plugins.
    """

    _scanners: Dict[str, BaseScanner] = {}
    _lock = threading.Lock()

    @classmethod
    def register(
        cls,
        scanner: Union[BaseScanner, Type[BaseScanner]],
        *,
        name: Optional[str] = None,
    ) -> BaseScanner:
        """
        Register a custom scanner instance or class.

        Args:
            scanner: An instance or subclass of :class:`BaseScanner`.
            name: Optional override for the scanner's registered name.

        Returns:
            The registered :class:`BaseScanner` instance.
        """
        instance: BaseScanner
        if isinstance(scanner, type) and issubclass(scanner, BaseScanner):
            instance = scanner()
        elif isinstance(scanner, BaseScanner):
            instance = scanner
        else:
            raise TypeError(
                f"Expected instance or subclass of BaseScanner, got {type(scanner).__name__}"
            )

        reg_name = name or instance.name
        with cls._lock:
            cls._scanners[reg_name] = instance
        return instance

    @classmethod
    def unregister(cls, name: str) -> bool:
        """
        Remove a scanner by name.

        Returns:
            True if removed, False if not found.
        """
        with cls._lock:
            return cls._scanners.pop(name, None) is not None

    @classmethod
    def get(cls, name: str) -> Optional[BaseScanner]:
        """Look up a scanner by name."""
        with cls._lock:
            return cls._scanners.get(name)

    @classmethod
    def get_all(cls) -> List[BaseScanner]:
        """Return a snapshot list of all registered scanners."""
        with cls._lock:
            return list(cls._scanners.values())

    @classmethod
    def clear(cls) -> None:
        """Remove all registered scanners (useful for test isolation)."""
        with cls._lock:
            cls._scanners.clear()


def register_scanner(
    scanner: Optional[Union[BaseScanner, Type[BaseScanner], Callable[..., Any]]] = None,
    *,
    name: Optional[str] = None,
) -> Any:
    """
    Register a scanner instance or decorate a :class:`BaseScanner` class.

    Can be used directly as a function or as a class decorator:

    Examples:
        Direct function call:
            >>> from asha.scanners import PatternScanner, register_scanner
            >>> scanner = PatternScanner("acme_id", r"ACME-\\d{5}")
            >>> register_scanner(scanner)

        Decorator without arguments:
            >>> @register_scanner
            ... class CustomScanner(BaseScanner):
            ...     @property
            ...     def name(self): return "custom"
            ...     def scan(self, text): return []

        Decorator with name override:
            >>> @register_scanner(name="override_name")
            ... class AnotherScanner(BaseScanner):
            ...     ...
    """
    if scanner is None:
        # Called as decorator with arguments: @register_scanner(name="...")
        def decorator(cls: Type[BaseScanner]) -> Type[BaseScanner]:
            ScannerRegistry.register(cls, name=name)
            return cls

        return decorator

    # Called as decorator without arguments: @register_scanner
    if isinstance(scanner, type) and issubclass(scanner, BaseScanner):
        ScannerRegistry.register(scanner, name=name)
        return scanner

    # Called as direct function: register_scanner(my_instance)
    if isinstance(scanner, BaseScanner):
        return ScannerRegistry.register(scanner, name=name)

    raise TypeError(
        f"@register_scanner can only decorate or accept BaseScanner objects, got {type(scanner).__name__}"
    )


def unregister_scanner(name: str) -> bool:
    """Unregister a scanner by name."""
    return ScannerRegistry.unregister(name)


def get_registered_scanners() -> List[BaseScanner]:
    """Get all currently registered scanners."""
    return ScannerRegistry.get_all()


def clear_registered_scanners() -> None:
    """Clear all registered scanners."""
    ScannerRegistry.clear()
