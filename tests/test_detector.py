"""Tests for sequence detection and message classification."""

from ctxpact.compaction.detector import SequenceDetector
from ctxpact.config import PreserveConfig


def _msgs(roles: list[str]) -> list[dict]:
    return [{"role": r, "content": f"msg {i}"} for i, r in enumerate(roles)]


class TestSequenceDetector:
    def test_short_conversations_fully_retained(self):
        detector = SequenceDetector(retention_window=6)
        messages = _msgs(["system", "user", "assistant"])
        result = detector.classify(messages)
        assert len(result.retention_window) == 3
        assert len(result.compactible_messages) == 0

    def test_system_messages_always_preserved(self):
        detector = SequenceDetector(retention_window=2)
        messages = _msgs(["system", "user", "assistant", "user", "assistant",
                          "user", "assistant", "user", "assistant"])
        result = detector.classify(messages)
        assert len(result.system_messages) == 1
        assert result.system_messages[0]["role"] == "system"

    def test_retention_window_preserved(self):
        detector = SequenceDetector(retention_window=4)
        messages = _msgs(["system"] + ["user", "assistant"] * 6)
        result = detector.classify(messages)
        assert len(result.retention_window) == 4

    def test_user_messages_extracted_from_compactible(self):
        detector = SequenceDetector(
            retention_window=2,
            preserve_config=PreserveConfig(user_messages=True),
        )
        messages = _msgs(["system", "user", "assistant", "tool",
                          "assistant", "user", "assistant", "user", "assistant"])
        result = detector.classify(messages)
        # User messages in compactible region should be in preserved_user_messages
        assert all(m["role"] == "user" for m in result.preserved_user_messages)

    def test_conservative_merge_preserves_more(self):
        detector_conservative = SequenceDetector(
            retention_window=4,
            eviction_window=0.5,
            merge_strategy="conservative",
        )
        detector_aggressive = SequenceDetector(
            retention_window=4,
            eviction_window=0.5,
            merge_strategy="aggressive",
        )
        messages = _msgs(["user", "assistant"] * 10)

        result_c = detector_conservative.classify(messages)
        result_a = detector_aggressive.classify(messages)

        # Conservative should preserve at least as many as aggressive
        assert result_c.preserved_count >= result_a.preserved_count
