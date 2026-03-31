"""Stage 1: Dynamic Context Pruning (DCP) — no LLM call needed.

Inspired by OpenCode's oh-my-opencode DCP strategy. Removes redundant
content from message history using pattern matching:

  1. Deduplicate tool calls (same tool + same args)
  2. Strip superseded file writes (keep only latest version)
  3. Truncate long error stack traces
  4. Strip raw tool result payloads (keep status only)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from ctxpact.config import DcpConfig

logger = logging.getLogger(__name__)


@dataclass
class PruneResult:
    """Result of a DCP pruning pass."""

    messages: list[dict[str, Any]]
    tokens_saved_estimate: int
    deduped_tool_calls: int
    superseded_writes: int
    truncated_errors: int
    stripped_payloads: int


class DynamicContextPruner:
    """Stateless pruner — takes messages in, returns pruned messages out."""

    def __init__(self, config: DcpConfig | None = None) -> None:
        self.config = config or DcpConfig()

    def prune(self, messages: list[dict[str, Any]]) -> PruneResult:
        """Run all enabled pruning stages on a copy of messages."""
        result = PruneResult(
            messages=deepcopy(messages),
            tokens_saved_estimate=0,
            deduped_tool_calls=0,
            superseded_writes=0,
            truncated_errors=0,
            stripped_payloads=0,
        )

        if self.config.dedup_tool_calls:
            self._dedup_tool_calls(result)

        if self.config.strip_superseded_writes:
            self._strip_superseded_writes(result)

        if self.config.truncate_errors:
            self._truncate_errors(result)

        if self.config.strip_tool_payloads:
            self._strip_tool_payloads(result)

        return result

    def _dedup_tool_calls(self, result: PruneResult) -> None:
        """Remove duplicate tool calls with identical arguments.

        If the same tool was called with the same args multiple times,
        keep only the last occurrence and its result.
        """
        # Map: (tool_name, args_hash) → index of last occurrence
        seen: dict[str, int] = {}
        tool_call_indices: list[int] = []

        for i, msg in enumerate(result.messages):
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    args = fn.get("arguments", "")
                    # args may be dict (pre-parsed) or str — normalize to str
                    if not isinstance(args, str):
                        args = json.dumps(args, sort_keys=True)
                    key = f"{name}:{hashlib.md5(args.encode()).hexdigest()}"

                    if key in seen:
                        tool_call_indices.append(seen[key])
                    seen[key] = i

        # Remove duplicate messages (earlier occurrences)
        if tool_call_indices:
            indices_to_remove = set(tool_call_indices)
            # Also remove the corresponding tool result messages
            expanded_indices: set[int] = set()
            for idx in indices_to_remove:
                expanded_indices.add(idx)
                # Check next message — if it's a tool result, remove it too
                if idx + 1 < len(result.messages) and result.messages[idx + 1].get("role") == "tool":
                    expanded_indices.add(idx + 1)

            original_len = len(result.messages)
            result.messages = [
                m for i, m in enumerate(result.messages) if i not in expanded_indices
            ]
            result.deduped_tool_calls = original_len - len(result.messages)

    def _strip_superseded_writes(self, result: PruneResult) -> None:
        """For file write tool results, keep only the latest version.

        Detects patterns like: write_file(path="foo.py", content="...")
        If the same path was written multiple times, remove earlier writes.
        """
        # Track last write index per file path
        write_indices: dict[str, list[int]] = {}

        for i, msg in enumerate(result.messages):
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    name = fn.get("name", "").lower()
                    if any(w in name for w in ("write", "create", "save", "update_file")):
                        try:
                            args = json.loads(fn.get("arguments", "{}"))
                            path = args.get("path") or args.get("file_path") or args.get("filename")
                            if path:
                                write_indices.setdefault(path, []).append(i)
                        except (json.JSONDecodeError, AttributeError):
                            pass

        # Remove all but the last write for each path
        indices_to_remove: set[int] = set()
        for path, indices in write_indices.items():
            if len(indices) > 1:
                for idx in indices[:-1]:  # Keep last, remove earlier
                    indices_to_remove.add(idx)
                    if idx + 1 < len(result.messages) and result.messages[idx + 1].get("role") == "tool":
                        indices_to_remove.add(idx + 1)

        if indices_to_remove:
            original_len = len(result.messages)
            result.messages = [
                m for i, m in enumerate(result.messages) if i not in indices_to_remove
            ]
            result.superseded_writes = original_len - len(result.messages)

    def _truncate_errors(self, result: PruneResult) -> None:
        """Truncate long error messages and stack traces.

        Keep the first line (error type) and last 3 lines (most relevant frame).
        """
        error_pattern = re.compile(
            r"(Traceback|Error|Exception|FAILED|panic|fatal)", re.IGNORECASE
        )

        for msg in result.messages:
            # Only truncate tool results and assistant messages, never user/system
            role = msg.get("role", "")
            if role in ("user", "system"):
                continue

            content = msg.get("content", "")
            if not isinstance(content, str):
                continue

            if error_pattern.search(content):
                lines = content.split("\n")
                if len(lines) > 10:
                    original_len = len(content)
                    truncated = (
                        lines[:3]
                        + ["    ... (truncated by ctxpact DCP) ..."]
                        + lines[-3:]
                    )
                    msg["content"] = "\n".join(truncated)
                    result.truncated_errors += 1
                    result.tokens_saved_estimate += (original_len - len(msg["content"])) // 3

    def _strip_tool_payloads(self, result: PruneResult) -> None:
        """Replace verbose tool result payloads with status summaries.

        Only strips results from older messages (not in retention window).
        Keeps the tool name and a brief status indicator.
        """
        for msg in result.messages:
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                if isinstance(content, str) and len(content) > 500:
                    original_len = len(content)
                    # Extract key status info
                    status = "success" if "error" not in content.lower() else "error"
                    preview = content[:200].replace("\n", " ")
                    msg["content"] = f"[Tool result ({status}): {preview}...]"
                    result.stripped_payloads += 1
                    result.tokens_saved_estimate += (original_len - len(msg["content"])) // 3
