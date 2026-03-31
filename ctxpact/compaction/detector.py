"""Sequence detector — classifies messages into compactible vs. preserved.

Uses a sliding window to identify patterns:
  [assistant] → [tool_call] → [tool_result] → [assistant]

These sequences are safe to compact as a unit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ctxpact.config import PreserveConfig

logger = logging.getLogger(__name__)


@dataclass
class SequenceClassification:
    """Result of classifying messages into groups."""

    system_messages: list[dict[str, Any]]
    preserved_user_messages: list[dict[str, Any]]
    compactible_messages: list[dict[str, Any]]
    retention_window: list[dict[str, Any]]

    @property
    def preserved_count(self) -> int:
        return (
            len(self.system_messages)
            + len(self.preserved_user_messages)
            + len(self.retention_window)
        )

    @property
    def compactible_count(self) -> int:
        return len(self.compactible_messages)


class SequenceDetector:
    """Classifies messages into compactible vs. preserved groups."""

    def __init__(
        self,
        retention_window: int = 6,
        eviction_window: float = 0.30,
        merge_strategy: str = "conservative",
        preserve_config: PreserveConfig | None = None,
    ) -> None:
        self.retention_window = retention_window
        self.eviction_window = eviction_window
        self.merge_strategy = merge_strategy
        self.preserve = preserve_config or PreserveConfig()

    def classify(self, messages: list[dict[str, Any]]) -> SequenceClassification:
        """Split messages into system, preserved, compactible, and retained groups.

        The conservative merge strategy (from Forge) ensures that if
        retention_window and eviction_window conflict, we preserve more.
        """
        if len(messages) <= self.retention_window:
            return SequenceClassification(
                system_messages=[],
                preserved_user_messages=[],
                compactible_messages=[],
                retention_window=messages,
            )

        system_msgs: list[dict[str, Any]] = []
        user_msgs: list[dict[str, Any]] = []
        other_msgs: list[dict[str, Any]] = []

        # Separate system messages (always at the start)
        non_system: list[dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") == "system" and self.preserve.system_prompts:
                system_msgs.append(msg)
            else:
                non_system.append(msg)

        if not non_system:
            return SequenceClassification(
                system_messages=system_msgs,
                preserved_user_messages=[],
                compactible_messages=[],
                retention_window=[],
            )

        # Apply retention window (last N messages)
        retention_count = min(self.retention_window, len(non_system))

        # Apply eviction window (compact X% of messages from the front)
        eviction_count = int(len(non_system) * self.eviction_window)

        # Conservative merge (from Forge): when retention_window and eviction_window
        # conflict, preserve MORE messages (smaller split = less compaction).
        split_from_retention = len(non_system) - retention_count
        split_from_eviction = eviction_count

        if self.merge_strategy == "conservative":
            split_point = min(split_from_retention, split_from_eviction)
        else:
            split_point = split_from_retention

        compactible_region = non_system[:split_point]
        retained_region = non_system[split_point:]

        # From the compactible region, extract user messages if they should be preserved
        for msg in compactible_region:
            if msg.get("role") == "user" and self.preserve.user_messages:
                # Preserve the user's intent but it can be summarized
                user_msgs.append(msg)
            else:
                other_msgs.append(msg)

        return SequenceClassification(
            system_messages=system_msgs,
            preserved_user_messages=user_msgs,
            compactible_messages=other_msgs,
            retention_window=retained_region,
        )
