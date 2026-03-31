"""Python import extraction using stdlib ast — zero external dependencies.

Handles:
  - import os
  - import os.path
  - from os import path
  - from . import sibling
  - from ..parent import thing
  - from .utils.helpers import func
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path

from ctxpact.isolation.language_parser import LanguageParser


class PythonParser(LanguageParser):
    """Extract and resolve Python imports using the stdlib ast module."""

    @property
    def file_extensions(self) -> list[str]:
        return [".py"]

    def extract_imports(self, file_path: str) -> list[str]:
        """Extract all import module names from a Python file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
        except (OSError, UnicodeDecodeError):
            return []

        try:
            tree = ast.parse(source, filename=file_path)
        except SyntaxError:
            # Fallback to regex for files with syntax errors
            return self._regex_fallback(source)

        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                level = node.level or 0
                if level > 0:
                    # Relative import: encode as dots + module
                    prefix = "." * level
                    imports.append(f"{prefix}{module}" if module else prefix)
                elif module:
                    imports.append(module)

        return imports

    def resolve_import(
        self, import_path: str, current_file: str, root_dir: str
    ) -> str | None:
        """Resolve a Python import to an absolute file path.

        Handles relative imports (leading dots) and absolute imports
        that map to files within root_dir.
        """
        root_dir = os.path.abspath(root_dir)
        current_dir = os.path.dirname(os.path.abspath(current_file))

        if import_path.startswith("."):
            return self._resolve_relative(import_path, current_dir, root_dir)
        else:
            return self._resolve_absolute(import_path, root_dir)

    def _resolve_relative(
        self, import_path: str, current_dir: str, root_dir: str
    ) -> str | None:
        """Resolve a relative import like '.utils' or '..config'."""
        # Count leading dots
        level = 0
        for ch in import_path:
            if ch == ".":
                level += 1
            else:
                break

        module_part = import_path[level:]

        # Go up 'level' directories from current_dir
        # level=1 means current package, level=2 means parent package, etc.
        base_dir = current_dir
        for _ in range(level - 1):
            base_dir = os.path.dirname(base_dir)

        if module_part:
            # Convert dotted module to path
            rel_path = module_part.replace(".", os.sep)
            candidate = os.path.join(base_dir, rel_path)
        else:
            # Just dots with no module — refers to the package itself
            candidate = base_dir

        return self._find_python_file(candidate, root_dir)

    def _resolve_absolute(self, import_path: str, root_dir: str) -> str | None:
        """Resolve an absolute import like 'ctxpact.config' to a file in root_dir."""
        rel_path = import_path.replace(".", os.sep)
        candidate = os.path.join(root_dir, rel_path)
        return self._find_python_file(candidate, root_dir)

    def _find_python_file(self, candidate: str, root_dir: str) -> str | None:
        """Check common Python file patterns for a candidate path."""
        # Direct .py file
        if os.path.isfile(candidate + ".py"):
            return os.path.abspath(candidate + ".py")

        # Package directory with __init__.py
        init_path = os.path.join(candidate, "__init__.py")
        if os.path.isfile(init_path):
            return os.path.abspath(init_path)

        # Already a .py file
        if candidate.endswith(".py") and os.path.isfile(candidate):
            return os.path.abspath(candidate)

        return None

    def _regex_fallback(self, source: str) -> list[str]:
        """Fallback regex extraction for files with syntax errors."""
        imports: list[str] = []

        # import X
        for match in re.finditer(r"^import\s+([\w.]+)", source, re.MULTILINE):
            imports.append(match.group(1))

        # from X import Y
        for match in re.finditer(
            r"^from\s+([\w.]+)\s+import", source, re.MULTILINE
        ):
            imports.append(match.group(1))

        return imports
