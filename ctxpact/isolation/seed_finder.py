"""Enhanced seed finding for GOG context isolation.

Improved over GOG's original keyword-only matching:
  1. Extract code identifiers from the user's prompt (class names, function names)
  2. Match against file names AND file content (exported symbols)
  3. Score candidates by relevance
  4. Fall back to keyword matching
"""

from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass

import networkx as nx

# Common Python identifier pattern (class names, function names, variables)
_IDENTIFIER_PATTERN = re.compile(r"\b([A-Z][a-zA-Z0-9]+|[a-z_][a-z0-9_]{2,})\b")

# File path pattern (looks like a path with slashes or dots)
_FILE_PATH_PATTERN = re.compile(
    r"(?:^|[\s`'\"])([a-zA-Z0-9_/.-]+\.(?:py|ts|vue|js|tsx|jsx))\b"
)

# Common noise words to exclude from identifier matching
_NOISE_WORDS = frozenset({
    "the", "and", "for", "from", "with", "that", "this", "have", "are",
    "was", "will", "can", "not", "all", "but", "how", "why", "what",
    "where", "when", "which", "some", "any", "each", "every", "into",
    "about", "between", "through", "just", "also", "then", "than",
    "here", "there", "been", "being", "would", "could", "should",
    "does", "did", "has", "had", "let", "may", "might", "must",
    "need", "use", "used", "using", "make", "made", "get", "set",
    "add", "new", "old", "see", "try", "run", "fix", "bug", "code",
    "file", "line", "error", "issue", "problem", "help", "please",
    "look", "check", "show", "find", "read", "write", "test",
    "def", "class", "import", "return", "self", "none", "true", "false",
    "str", "int", "float", "bool", "list", "dict", "set", "tuple",
})


@dataclass
class SeedResult:
    """A seed node with a confidence score."""
    node: str
    score: float
    reason: str


def extract_identifiers_from_text(text: str) -> set[str]:
    """Extract meaningful code identifiers from natural language text.

    Finds CamelCase names, snake_case names, and file paths.
    Filters out common English words.
    """
    identifiers: set[str] = set()

    # Extract identifiers
    for match in _IDENTIFIER_PATTERN.finditer(text):
        word = match.group(1)
        if word.lower() not in _NOISE_WORDS and len(word) > 2:
            identifiers.add(word.lower())

    return identifiers


def extract_file_paths_from_text(text: str) -> set[str]:
    """Extract file paths mentioned in text."""
    paths: set[str] = set()
    for match in _FILE_PATH_PATTERN.finditer(text):
        paths.add(match.group(1))
    return paths


def extract_symbols_from_python(file_path: str) -> set[str]:
    """Extract top-level class and function names from a Python file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
    except (SyntaxError, OSError, UnicodeDecodeError):
        return set()

    symbols: set[str] = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            symbols.add(node.name.lower())
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            symbols.add(node.name.lower())
    return symbols


def find_seeds(
    graph: nx.DiGraph,
    prompt: str,
    messages: list[dict] | None = None,
    repo_path: str = "",
) -> list[SeedResult]:
    """Find seed nodes in the dependency graph for a given prompt.

    Strategy:
      1. Exact file path match (highest confidence)
      2. File name keyword match
      3. Identifier match against file content (class/function names)
    """
    seeds: list[SeedResult] = []
    seen: set[str] = set()

    # Combine prompt with latest user messages for richer context
    full_text = prompt
    if messages:
        for msg in reversed(messages[-5:]):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    full_text += " " + content

    # Extract signals from text
    mentioned_paths = extract_file_paths_from_text(full_text)
    identifiers = extract_identifiers_from_text(full_text)

    for node in graph.nodes():
        basename = os.path.basename(node).lower()
        relpath = os.path.relpath(node, repo_path) if repo_path else node

        # Strategy 1: Exact file path match
        for path in mentioned_paths:
            if relpath.endswith(path) or basename == os.path.basename(path).lower():
                if node not in seen:
                    seeds.append(SeedResult(node, 1.0, f"path_match:{path}"))
                    seen.add(node)

        # Strategy 2: Filename keyword match
        name_without_ext = os.path.splitext(basename)[0]
        name_parts = set(re.split(r"[_.\-]", name_without_ext))
        overlap = identifiers & name_parts
        if overlap and node not in seen:
            score = min(len(overlap) * 0.4, 0.8)
            seeds.append(SeedResult(node, score, f"name_match:{overlap}"))
            seen.add(node)

        # Strategy 3: Symbol match (only for Python files, more expensive)
        if node.endswith(".py") and node not in seen and identifiers:
            symbols = extract_symbols_from_python(node)
            sym_overlap = identifiers & symbols
            if sym_overlap:
                score = min(len(sym_overlap) * 0.3, 0.7)
                seeds.append(SeedResult(node, score, f"symbol_match:{sym_overlap}"))
                seen.add(node)

    # Sort by confidence (highest first)
    seeds.sort(key=lambda s: s.score, reverse=True)
    return seeds
