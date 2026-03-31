"""Graph lifecycle manager — build, cache, and incrementally update.

Handles:
  - Loading graph from pickle cache
  - Building graph from scratch when cache is stale or missing
  - Incremental O(1) updates when files change
  - Exposing graph status for debug endpoints
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from typing import Any

import networkx as nx

from ctxpact.isolation.graph_builder import (
    build_graph,
    remove_file_from_graph,
    update_graph_for_file,
)

logger = logging.getLogger(__name__)


class GraphManager:
    """Manages the dependency graph lifecycle."""

    def __init__(
        self,
        repo_path: str,
        cache_path: str = "./gog_cache.pkl",
        include_extensions: list[str] | None = None,
        exclude_dirs: list[str] | None = None,
        rebuild_on_startup: bool = False,
    ) -> None:
        self.repo_path = os.path.abspath(repo_path)
        self.cache_path = cache_path
        self.include_extensions = include_extensions
        self.exclude_dirs = set(exclude_dirs) if exclude_dirs else None
        self.rebuild_on_startup = rebuild_on_startup

        self._graph: nx.DiGraph | None = None
        self._build_time: float = 0
        self._cache_loaded: bool = False

    @property
    def graph(self) -> nx.DiGraph | None:
        return self._graph

    async def ensure_graph(self) -> nx.DiGraph:
        """Load or build the graph. Safe to call multiple times."""
        if self._graph is not None:
            return self._graph

        if not self.rebuild_on_startup and os.path.isfile(self.cache_path):
            try:
                self._graph = self._load_cache()
                self._cache_loaded = True
                logger.info(
                    f"GOG: Loaded graph from cache: "
                    f"{self._graph.number_of_nodes()} nodes, "
                    f"{self._graph.number_of_edges()} edges"
                )
                return self._graph
            except Exception as e:
                logger.warning(f"GOG: Cache load failed ({e}), rebuilding")

        self._graph = self._build_and_cache()
        return self._graph

    def _load_cache(self) -> nx.DiGraph:
        """Load graph from pickle cache."""
        with open(self.cache_path, "rb") as f:
            graph = pickle.load(f)
        if not isinstance(graph, nx.DiGraph):
            raise TypeError(f"Expected DiGraph, got {type(graph)}")
        return graph

    def _build_and_cache(self) -> nx.DiGraph:
        """Build graph from repo and save to cache."""
        start = time.monotonic()

        graph = build_graph(
            root_dir=self.repo_path,
            include_extensions=self.include_extensions,
            exclude_dirs=self.exclude_dirs,
        )

        self._build_time = time.monotonic() - start
        self._cache_loaded = False

        # Save cache
        try:
            cache_dir = os.path.dirname(self.cache_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            with open(self.cache_path, "wb") as f:
                pickle.dump(graph, f)
            logger.info(
                f"GOG: Graph built and cached in {self._build_time:.2f}s: "
                f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
            )
        except Exception as e:
            logger.warning(f"GOG: Failed to cache graph: {e}")

        return graph

    def handle_file_change(self, file_path: str) -> None:
        """Incrementally update graph when a file changes.

        O(1) update instead of O(N) full rebuild.
        """
        if self._graph is None:
            return

        abs_path = os.path.abspath(file_path)

        if os.path.isfile(abs_path):
            # File created or modified
            update_graph_for_file(self._graph, file_path, self.repo_path)
            logger.debug(f"GOG: Updated graph for {os.path.basename(file_path)}")
        else:
            # File deleted
            remove_file_from_graph(self._graph, file_path)
            logger.debug(f"GOG: Removed {os.path.basename(file_path)} from graph")

    def rebuild(self) -> nx.DiGraph:
        """Force full rebuild of the graph."""
        self._graph = self._build_and_cache()
        return self._graph

    def status(self) -> dict[str, Any]:
        """Return graph status for debug endpoints."""
        if self._graph is None:
            return {
                "ready": False,
                "repo_path": self.repo_path,
                "cache_path": self.cache_path,
            }

        # Count files by extension
        ext_counts: dict[str, int] = {}
        for node in self._graph.nodes():
            ext = os.path.splitext(node)[1]
            ext_counts[ext] = ext_counts.get(ext, 0) + 1

        return {
            "ready": True,
            "repo_path": self.repo_path,
            "nodes": self._graph.number_of_nodes(),
            "edges": self._graph.number_of_edges(),
            "build_time_seconds": round(self._build_time, 3),
            "from_cache": self._cache_loaded,
            "cache_path": self.cache_path,
            "languages": ext_counts,
        }
