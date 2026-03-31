"""Core context isolation orchestrator.

Scans incoming messages for file contents, uses the dependency graph
to determine which files are structurally relevant to the user's query,
and strips irrelevant file contents from messages.

This runs BEFORE ctxpact's DCP compaction stage.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from ctxpact.isolation.seed_finder import find_seeds

logger = logging.getLogger(__name__)

# Patterns to detect file content in messages
_FILE_CONTENT_MARKERS = [
    # Tool result with file path
    re.compile(r"(?:File|Reading|Content of|Contents of)\s+[`'\"]?([^\s`'\"]+\.\w+)"),
    # Markdown code block with filename
    re.compile(r"```\w*\s*#\s*(\S+\.\w+)"),
    # Path-like strings
    re.compile(r"(?:^|[\s])(/?\w[\w/.-]+\.(?:py|ts|vue|js|tsx|jsx))\b"),
]


@dataclass
class IsolationResult:
    """Result of context isolation."""
    messages: list[dict[str, Any]]
    isolated_files: set[str] = field(default_factory=set)
    stripped_files: set[str] = field(default_factory=set)
    tokens_before: int = 0
    tokens_after: int = 0
    seeds_found: int = 0
    applied: bool = False


def _extract_file_refs_from_message(msg: dict) -> set[str]:
    """Extract file path references from a single message."""
    refs: set[str] = set()
    content = msg.get("content", "")

    if isinstance(content, str):
        for pattern in _FILE_CONTENT_MARKERS:
            for match in pattern.finditer(content):
                refs.add(match.group(1))

    # Check tool calls
    tool_calls = msg.get("tool_calls", [])
    if tool_calls:
        for tc in tool_calls:
            func = tc.get("function", {})
            args_str = func.get("arguments", "")
            if isinstance(args_str, str):
                try:
                    args = json.loads(args_str)
                    for key in ("path", "file_path", "filename", "file"):
                        if key in args:
                            refs.add(args[key])
                except (json.JSONDecodeError, TypeError):
                    pass

    return refs


def _estimate_content_tokens(text: str) -> int:
    """Quick token estimate (words / 0.75)."""
    return len(text.split())


def _is_large_file_content(content: str, min_lines: int = 10) -> bool:
    """Check if content looks like a large file dump."""
    return content.count("\n") >= min_lines


def isolate_context(
    graph: nx.DiGraph,
    messages: list[dict[str, Any]],
    prompt: str,
    repo_path: str,
    min_file_lines: int = 10,
) -> IsolationResult:
    """Isolate structurally relevant context from messages.

    Steps:
      1. Find seed nodes from the user's prompt
      2. Traverse graph to get the isolated file set
      3. Scan messages for file contents
      4. Strip file contents that aren't in the isolated set

    Only strips content from tool results and assistant messages that
    contain large file dumps. User messages and system messages are
    never modified.
    """
    result = IsolationResult(messages=list(messages))

    if graph.number_of_nodes() == 0:
        return result

    # Step 1: Find seeds
    seeds = find_seeds(graph, prompt, messages, repo_path)
    result.seeds_found = len(seeds)

    if not seeds:
        logger.debug("GOG: No seed nodes found, skipping isolation")
        return result

    seed_nodes = [s.node for s in seeds]
    logger.info(
        f"GOG: Found {len(seeds)} seeds: "
        + ", ".join(f"{os.path.basename(s.node)}({s.score:.1f})" for s in seeds[:5])
    )

    # Step 2: Traverse graph to get isolated set
    isolated: set[str] = set()

    if len(seed_nodes) == 1:
        # Single seed: seed + all descendants
        isolated.add(seed_nodes[0])
        isolated.update(nx.descendants(graph, seed_nodes[0]))
        # Also add ancestors (files that import this one)
        isolated.update(nx.ancestors(graph, seed_nodes[0]))
    else:
        # Multiple seeds: find paths between them + descendants
        for i in range(len(seed_nodes)):
            for j in range(i + 1, len(seed_nodes)):
                src, dst = seed_nodes[i], seed_nodes[j]
                # Try both directions
                for s, d in [(src, dst), (dst, src)]:
                    try:
                        path = nx.shortest_path(graph, source=s, target=d)
                        isolated.update(path)
                    except nx.NetworkXNoPath:
                        pass

            # Add descendants of each seed
            isolated.add(seed_nodes[i])
            try:
                isolated.update(nx.descendants(graph, seed_nodes[i]))
            except nx.NetworkXError:
                pass

        # Fallback if no paths found
        if not isolated:
            for seed in seed_nodes:
                isolated.add(seed)
                isolated.update(nx.descendants(graph, seed))

    result.isolated_files = isolated

    # Convert to basenames for matching against message content
    isolated_basenames = {os.path.basename(f).lower() for f in isolated}
    isolated_relpaths = set()
    if repo_path:
        isolated_relpaths = {
            os.path.relpath(f, repo_path) for f in isolated if f.startswith(repo_path)
        }

    logger.info(f"GOG: Isolated {len(isolated)} files from {graph.number_of_nodes()}-node graph")

    # Step 3+4: Scan and strip messages
    tokens_before = 0
    tokens_after = 0
    stripped: set[str] = set()
    new_messages: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        tokens_before += _estimate_content_tokens(str(content))

        # Never touch system or user messages
        if role in ("system", "user"):
            new_messages.append(msg)
            tokens_after += _estimate_content_tokens(str(content))
            continue

        # For tool results and assistant messages with file content,
        # check if the file is in our isolated set
        if role == "tool" and isinstance(content, str) and _is_large_file_content(content, min_file_lines):
            file_refs = _extract_file_refs_from_message(msg)
            # Check if ANY referenced file is in the isolated set
            is_relevant = False
            for ref in file_refs:
                ref_basename = os.path.basename(ref).lower()
                if (ref_basename in isolated_basenames
                    or ref in isolated_relpaths
                    or any(ref in f for f in isolated)):
                    is_relevant = True
                    break

            if not is_relevant and file_refs:
                # Strip this file content — replace with a note
                stripped_msg = dict(msg)
                stripped_names = ", ".join(os.path.basename(r) for r in file_refs)
                stripped_msg["content"] = f"[File content stripped by GOG: {stripped_names} — not structurally connected to current query]"
                new_messages.append(stripped_msg)
                stripped.update(file_refs)
                tokens_after += 15  # The replacement note
                continue

        new_messages.append(msg)
        tokens_after += _estimate_content_tokens(str(content))

    result.messages = new_messages
    result.stripped_files = stripped
    result.tokens_before = tokens_before
    result.tokens_after = tokens_after
    result.applied = len(stripped) > 0

    if stripped:
        saved = tokens_before - tokens_after
        pct = (saved / tokens_before * 100) if tokens_before > 0 else 0
        logger.info(
            f"GOG: Stripped {len(stripped)} files, "
            f"saved ~{saved} tokens ({pct:.0f}%)"
        )

    return result
