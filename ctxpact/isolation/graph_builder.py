"""Build a NetworkX dependency graph from any codebase.

Auto-detects language from file extensions and uses the appropriate parser.
The graph is a DiGraph where:
  - Nodes = absolute file paths
  - Edges = file_A → file_B (A imports B)
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import networkx as nx

from ctxpact.isolation.language_parser import LanguageParser
from ctxpact.isolation.python_parser import PythonParser
from ctxpact.isolation.ts_parser import TypeScriptParser

logger = logging.getLogger(__name__)

# Default directories to skip when walking
DEFAULT_EXCLUDE_DIRS = frozenset({
    "node_modules", ".venv", "venv", "__pycache__", ".git", ".hg",
    ".svn", "dist", "build", ".tox", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", "egg-info", ".eggs", "site-packages",
})

# Registry of parsers
_PARSERS: list[LanguageParser] = [
    PythonParser(),
    TypeScriptParser(),
]


def get_parser_for_file(file_path: str) -> LanguageParser | None:
    """Find the appropriate parser for a given file."""
    for parser in _PARSERS:
        if parser.can_parse(file_path):
            return parser
    return None


def discover_files(
    root_dir: str,
    include_extensions: list[str] | None = None,
    exclude_dirs: set[str] | None = None,
) -> list[str]:
    """Walk a directory tree and return all source files.

    Args:
        root_dir: Root directory to scan.
        include_extensions: If set, only include files with these extensions.
                          If None, include all files that have a registered parser.
        exclude_dirs: Directory names to skip. Defaults to common non-source dirs.
    """
    if exclude_dirs is None:
        exclude_dirs = DEFAULT_EXCLUDE_DIRS

    # Build the set of extensions we care about
    if include_extensions:
        valid_exts = set(include_extensions)
    else:
        valid_exts = set()
        for parser in _PARSERS:
            valid_exts.update(parser.file_extensions)

    files: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Prune excluded directories in-place
        dirnames[:] = [
            d for d in dirnames
            if d not in exclude_dirs and not d.startswith(".")
        ]

        for filename in filenames:
            if any(filename.endswith(ext) for ext in valid_exts):
                files.append(os.path.join(dirpath, filename))

    return files


def build_graph(
    root_dir: str,
    include_extensions: list[str] | None = None,
    exclude_dirs: set[str] | None = None,
) -> nx.DiGraph:
    """Build a dependency graph for a codebase.

    Returns a NetworkX DiGraph where nodes are absolute file paths
    and edges represent import dependencies.
    """
    root_dir = os.path.abspath(root_dir)
    start = time.monotonic()

    G = nx.DiGraph()
    files = discover_files(root_dir, include_extensions, exclude_dirs)

    # Add all files as nodes
    for file_path in files:
        abs_path = os.path.abspath(file_path)
        G.add_node(abs_path)

    # Add edges from imports
    edge_count = 0
    parse_errors = 0

    for file_path in files:
        abs_file = os.path.abspath(file_path)
        parser = get_parser_for_file(file_path)
        if parser is None:
            continue

        try:
            imports = parser.extract_imports(file_path)
            for imp in imports:
                resolved = parser.resolve_import(imp, abs_file, root_dir)
                if resolved and os.path.isfile(resolved):
                    resolved = os.path.abspath(resolved)
                    # Only add edges to files within our graph
                    if resolved in G:
                        G.add_edge(abs_file, resolved)
                        edge_count += 1
        except Exception as e:
            parse_errors += 1
            logger.debug(f"Parse error in {file_path}: {e}")

    elapsed = time.monotonic() - start
    logger.info(
        f"Graph built: {G.number_of_nodes()} nodes, {edge_count} edges, "
        f"{parse_errors} parse errors, {elapsed:.2f}s"
    )

    return G


def update_graph_for_file(
    graph: nx.DiGraph,
    file_path: str,
    root_dir: str,
) -> None:
    """Incrementally update the graph when a file is created or modified.

    This is the 'synaptic plasticity' from the GOG paper — O(1) update
    instead of O(N) full rebuild.
    """
    abs_file = os.path.abspath(file_path)
    root_dir = os.path.abspath(root_dir)

    parser = get_parser_for_file(file_path)
    if parser is None:
        return

    # Remove existing outgoing edges from this file (imports may have changed)
    if abs_file in graph:
        old_edges = list(graph.out_edges(abs_file))
        graph.remove_edges_from(old_edges)
    else:
        graph.add_node(abs_file)

    # Re-parse and add new edges
    if os.path.isfile(file_path):
        try:
            imports = parser.extract_imports(file_path)
            for imp in imports:
                resolved = parser.resolve_import(imp, abs_file, root_dir)
                if resolved and os.path.isfile(resolved):
                    resolved = os.path.abspath(resolved)
                    if resolved not in graph:
                        graph.add_node(resolved)
                    graph.add_edge(abs_file, resolved)
        except Exception as e:
            logger.debug(f"Update parse error for {file_path}: {e}")


def remove_file_from_graph(graph: nx.DiGraph, file_path: str) -> None:
    """Remove a file node and all its edges from the graph."""
    abs_file = os.path.abspath(file_path)
    if abs_file in graph:
        graph.remove_node(abs_file)
