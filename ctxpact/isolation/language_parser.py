"""Abstract base class for language-specific import parsers.

Each language parser must implement:
  - extract_imports(file_path) → list of raw import strings
  - resolve_import(import_path, current_file, root_dir) → absolute file path or None
  - file_extensions → list of extensions this parser handles
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class LanguageParser(ABC):
    """Base class for language-specific import extraction."""

    @property
    @abstractmethod
    def file_extensions(self) -> list[str]:
        """Return file extensions this parser handles (e.g., ['.py'])."""
        ...

    @abstractmethod
    def extract_imports(self, file_path: str) -> list[str]:
        """Extract import paths/modules from a source file.

        Returns a list of raw import strings (module names or relative paths).
        Should not raise on malformed files — return empty list instead.
        """
        ...

    @abstractmethod
    def resolve_import(
        self, import_path: str, current_file: str, root_dir: str
    ) -> str | None:
        """Resolve an import string to an absolute file path within root_dir.

        Returns None if the import is external (third-party) or unresolvable.
        """
        ...

    def can_parse(self, file_path: str) -> bool:
        """Check if this parser handles the given file."""
        return any(file_path.endswith(ext) for ext in self.file_extensions)
