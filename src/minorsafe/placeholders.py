"""Fail-closed registry of CPU placeholder components.

Several components are currently lightweight stand-ins so the CPU smoke pipeline can run end to end
(heuristic guards, fake LoRA, lexicon "MLM" substitution, heuristic judge/PPL, ...). Their outputs are
not scientifically meaningful. To prevent them from silently producing reported numbers, every
placeholder must call :func:`require_real_backend`, which raises unless the active hardware profile
explicitly opts in with ``allow_placeholders: true`` (only ``cpu_debug`` does).
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


class PlaceholderBackendError(RuntimeError):
    """Raised when a placeholder component would run on a profile that forbids placeholders."""


def require_real_backend(hardware: Any, component: str, reason: str) -> None:
    """Raise :class:`PlaceholderBackendError` unless ``hardware.allow_placeholders`` is true.

    ``hardware`` is a :class:`~minorsafe.config.HardwareProfile` (typed loosely to avoid import cycles).
    """
    if bool(getattr(hardware, "allow_placeholders", False)):
        log.warning("Using CPU placeholder for %s (%s); results are NOT meaningful.", component, reason)
        return
    name = getattr(hardware, "name", "?")
    raise PlaceholderBackendError(
        f"{component} is a CPU placeholder ({reason}) and is not allowed on hardware profile {name!r}. "
        "Implement the real backend, or set `allow_placeholders: true` in the profile for debugging only."
    )

