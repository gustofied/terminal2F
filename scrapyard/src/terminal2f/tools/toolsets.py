# maybe rename from saying toolsets tbh..
from __future__ import annotations

from typing import FrozenSet

PAYMENTS: FrozenSet[str] = frozenset({"retrieve_payment_status", "retrieve_payment_date"})
CODE: FrozenSet[str] = frozenset({"read", "write", "edit", "glob", "grep"})
ALL_TOOLS: FrozenSet[str] = frozenset({"*"})
NO_TOOLS: FrozenSet[str] = frozenset()
