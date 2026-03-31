"""Tests for Dynamic Context Pruning (Stage 1)."""

import json

from ctxpact.compaction.pruner import DynamicContextPruner
from ctxpact.config import DcpConfig


def _tool_call_msg(name: str, args: dict) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": f"call_{name}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        ],
    }


def _tool_result_msg(content: str, call_id: str = "call_test") -> dict:
    return {"role": "tool", "content": content, "tool_call_id": call_id}


class TestDeduplicateToolCalls:
    def test_removes_duplicate_tool_calls(self):
        pruner = DynamicContextPruner(DcpConfig(dedup_tool_calls=True))
        messages = [
            {"role": "user", "content": "search for X"},
            _tool_call_msg("search", {"query": "X"}),
            _tool_result_msg("result 1"),
            _tool_call_msg("search", {"query": "X"}),  # Duplicate
            _tool_result_msg("result 2"),
        ]
        result = pruner.prune(messages)
        # Should remove the first duplicate, keep the last
        assert result.deduped_tool_calls >= 1
        assert len(result.messages) < len(messages)

    def test_keeps_different_tool_calls(self):
        pruner = DynamicContextPruner(DcpConfig(dedup_tool_calls=True))
        messages = [
            _tool_call_msg("search", {"query": "X"}),
            _tool_result_msg("result 1"),
            _tool_call_msg("search", {"query": "Y"}),  # Different args
            _tool_result_msg("result 2"),
        ]
        result = pruner.prune(messages)
        assert result.deduped_tool_calls == 0
        assert len(result.messages) == len(messages)


class TestStripSupersededWrites:
    def test_keeps_only_latest_write(self):
        pruner = DynamicContextPruner(DcpConfig(strip_superseded_writes=True))
        messages = [
            _tool_call_msg("write_file", {"path": "main.py", "content": "v1"}),
            _tool_result_msg("written"),
            _tool_call_msg("write_file", {"path": "main.py", "content": "v2"}),
            _tool_result_msg("written"),
            _tool_call_msg("write_file", {"path": "main.py", "content": "v3"}),
            _tool_result_msg("written"),
        ]
        result = pruner.prune(messages)
        assert result.superseded_writes >= 2  # Removed 2 earlier writes + results


class TestTruncateErrors:
    def test_truncates_long_tracebacks(self):
        pruner = DynamicContextPruner(DcpConfig(truncate_errors=True))
        long_trace = "Traceback (most recent call last):\n" + "\n".join(
            [f"  File line {i}" for i in range(50)]
        ) + "\nValueError: something broke"

        messages = [{"role": "tool", "content": long_trace}]
        result = pruner.prune(messages)
        assert result.truncated_errors == 1
        content = result.messages[0]["content"]
        assert "truncated by ctxpact" in content
        assert len(content) < len(long_trace)


class TestStripToolPayloads:
    def test_strips_verbose_tool_results(self):
        pruner = DynamicContextPruner(DcpConfig(strip_tool_payloads=True))
        verbose_output = "x" * 1000
        messages = [{"role": "tool", "content": verbose_output}]
        result = pruner.prune(messages)
        assert result.stripped_payloads == 1
        assert len(result.messages[0]["content"]) < 500

    def test_preserves_short_tool_results(self):
        pruner = DynamicContextPruner(DcpConfig(strip_tool_payloads=True))
        messages = [{"role": "tool", "content": "OK"}]
        result = pruner.prune(messages)
        assert result.stripped_payloads == 0
        assert result.messages[0]["content"] == "OK"
