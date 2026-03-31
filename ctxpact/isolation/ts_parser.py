"""TypeScript/Vue import extraction — ported from GOG's srm_engine.

Uses regex-based extraction (no tree-sitter dependency required).
Tree-sitter can be added as an optional enhancement.

Handles:
  - import { X } from './module'
  - import X from '../module'
  - import type { X } from './module'
  - Vue <script> block extraction
"""

from __future__ import annotations

import os
import re

from ctxpact.isolation.language_parser import LanguageParser

# Regex for TypeScript/JavaScript imports
_IMPORT_PATTERN = re.compile(
    r"""import\s+(?:type\s+)?(?:.*?\s+from\s+)?['"](.+?)['"]""",
    re.MULTILINE,
)

# Extract <script> block from Vue SFCs
_VUE_SCRIPT_PATTERN = re.compile(
    r"<script.*?>\s*(.*?)\s*</script>", re.DOTALL
)


class TypeScriptParser(LanguageParser):
    """Extract and resolve TypeScript/Vue imports."""

    @property
    def file_extensions(self) -> list[str]:
        return [".ts", ".tsx", ".js", ".jsx", ".vue"]

    def extract_imports(self, file_path: str) -> list[str]:
        """Extract import paths from a TypeScript or Vue file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except (OSError, UnicodeDecodeError):
            return []

        # For Vue files, extract just the <script> block
        if file_path.endswith(".vue"):
            match = _VUE_SCRIPT_PATTERN.search(content)
            if match:
                content = match.group(1)
            else:
                return []

        return _IMPORT_PATTERN.findall(content)

    def resolve_import(
        self, import_path: str, current_file: str, root_dir: str
    ) -> str | None:
        """Resolve a TS/JS import to an absolute file path."""
        if not import_path.startswith("."):
            # Non-relative = node_modules package, skip
            return None

        current_dir = os.path.dirname(os.path.abspath(current_file))
        candidate = os.path.normpath(os.path.join(current_dir, import_path))

        # Check common extensions
        for ext in [".ts", ".tsx", ".js", ".jsx", ".vue", "/index.ts", "/index.js"]:
            full = candidate + ext
            if os.path.isfile(full):
                return os.path.abspath(full)

        # Maybe already has extension
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)

        return None
