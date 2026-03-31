"""ConversationBook — stores full conversation as searchable text.

Each message in a session becomes a section in the book. The book provides:
- Section-based indexing for RLM orientation
- Full text retrieval by section number
- Token counting for budget management
- Persistence to disk (optional)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ctxpact.compaction.tokens import count_tokens

logger = logging.getLogger(__name__)


@dataclass
class BookSection:
    """A single section in the conversation book."""

    index: int
    role: str  # system, user, assistant, tool
    turn: int  # conversation turn number (user+assistant = 1 turn)
    content: str
    timestamp: float = field(default_factory=time.time)
    token_count: int = 0

    def __post_init__(self) -> None:
        if not self.token_count:
            self.token_count = count_tokens(self.content)

    @property
    def header(self) -> str:
        return f"=== Section {self.index} [{self.role}, turn {self.turn}] ==="

    def to_text(self) -> str:
        return f"{self.header}\n{self.content}"

    @property
    def index_entry(self) -> str:
        """One-line summary for the table of contents."""
        preview = self.content[:120].replace("\n", " ").strip()
        if len(self.content) > 120:
            preview += "..."
        return (
            f"  [{self.index}] {self.role} (turn {self.turn}, "
            f"{self.token_count} tokens): {preview}"
        )

    def header_text(self, max_chars: int = 600) -> str:
        """Return section header with extended preview for extraction.

        Captures structural markers (chapter titles, headings) that appear
        near the start of the section, plus key metadata like names and dates.
        """
        text = self.content[:max_chars].rstrip()
        if len(self.content) > max_chars:
            text += f"\n[... {self.token_count} tokens total ...]"
        return f"{self.header}\n{text}"


class ConversationBook:
    """Accumulates conversation messages as a searchable book."""

    def __init__(self, session_id: str = "") -> None:
        self.session_id = session_id
        self.sections: list[BookSection] = []
        self._turn_counter = 0
        self._last_role = ""

    # Large messages get split into chunks of this many tokens
    MAX_SECTION_TOKENS = 2000

    def append_message(self, role: str, content: str) -> None:
        """Add a message to the book as one or more sections.

        Large messages are automatically split into multiple sections so that
        each section fits within the RLM's per-tool-call reading budget.
        """
        if not isinstance(content, str) or not content.strip():
            return

        # Track turns: a user message starts a new turn
        if role == "user":
            self._turn_counter += 1
        turn = max(self._turn_counter, 1)

        token_count = count_tokens(content)
        if token_count <= self.MAX_SECTION_TOKENS:
            section = BookSection(
                index=len(self.sections) + 1,
                role=role,
                turn=turn,
                content=content,
            )
            self.sections.append(section)
        else:
            # Split into chunks, preferring paragraph boundaries
            chunks = self._split_content(content, self.MAX_SECTION_TOKENS)
            for i, chunk in enumerate(chunks):
                section = BookSection(
                    index=len(self.sections) + 1,
                    role=role,
                    turn=turn,
                    content=chunk,
                )
                self.sections.append(section)

        self._last_role = role

    @staticmethod
    def _split_content(content: str, max_tokens: int) -> list[str]:
        """Split content into chunks, preferring structural boundaries.

        Splits at heading markers (===, ---, # ) when possible, falling
        back to paragraph boundaries. This keeps chapter/section structure
        intact within chunks.
        """
        import re

        # First, try to split at structural boundaries
        # Match lines like "===...===", "---...---", "# Heading", "Chapter N:"
        heading_pattern = re.compile(
            r'\n(?=={3,}|#{1,3}\s|Chapter\s+\d|---{3,})',
            re.MULTILINE,
        )

        structural_parts = heading_pattern.split(content)

        # If structural splitting produces good-sized chunks, use them
        if len(structural_parts) > 1:
            chunks: list[str] = []
            current_parts: list[str] = []
            current_tokens = 0

            for part in structural_parts:
                part = part.strip()
                if not part:
                    continue
                part_tokens = count_tokens(part)

                if current_parts and current_tokens + part_tokens > max_tokens:
                    chunks.append("\n\n".join(current_parts))
                    current_parts = [part]
                    current_tokens = part_tokens
                else:
                    current_parts.append(part)
                    current_tokens += part_tokens

            if current_parts:
                chunks.append("\n\n".join(current_parts))

            if len(chunks) > 1:
                return chunks

        # Fallback: split at paragraph boundaries
        paragraphs = content.split("\n\n")
        chunks = []
        current: list[str] = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = count_tokens(para)
            if current and current_tokens + para_tokens > max_tokens:
                chunks.append("\n\n".join(current))
                current = [para]
                current_tokens = para_tokens
            else:
                current.append(para)
                current_tokens += para_tokens

        if current:
            chunks.append("\n\n".join(current))

        return chunks

    def to_searchable_text(self) -> str:
        """Full book text with section markers."""
        return "\n\n".join(s.to_text() for s in self.sections)

    def to_section_index(self) -> str:
        """Table of contents showing all sections."""
        lines = [
            f"Conversation Book — {len(self.sections)} sections, "
            f"{self.total_tokens} tokens, {self._turn_counter} turns",
            "",
        ]
        for s in self.sections:
            lines.append(s.index_entry)
        return "\n".join(lines)

    def get_section(self, index: int) -> BookSection | None:
        """Retrieve a section by 1-based index."""
        if 1 <= index <= len(self.sections):
            return self.sections[index - 1]
        return None

    def get_sections_by_range(self, start: int, end: int) -> list[BookSection]:
        """Get sections by 1-based inclusive range."""
        return [
            s for s in self.sections
            if start <= s.index <= end
        ]

    def get_sections_by_role(self, role: str) -> list[BookSection]:
        """Get all sections with a given role."""
        return [s for s in self.sections if s.role == role]

    def get_sections_text(self, indices: list[int]) -> str:
        """Get combined text for specific section indices."""
        parts = []
        for idx in sorted(indices):
            section = self.get_section(idx)
            if section:
                parts.append(section.to_text())
        return "\n\n".join(parts)

    @property
    def total_tokens(self) -> int:
        return sum(s.token_count for s in self.sections)

    @property
    def section_count(self) -> int:
        return len(self.sections)

    def to_section_dict(self) -> dict[str, str]:
        """Return dict mapping section ID to content (for RLM directory input)."""
        return {
            f"section_{s.index}": s.content
            for s in self.sections
        }

    def build_from_messages(self, messages: list[dict[str, Any]]) -> None:
        """Rebuild book from OpenAI-format messages."""
        self.sections.clear()
        self._turn_counter = 0
        self._last_role = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, str):
                self.append_message(role, content)
            elif isinstance(content, list):
                # Multi-part content — extract text parts
                text_parts = [
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("text")
                ]
                if text_parts:
                    self.append_message(role, "\n".join(text_parts))

    # ---- Persistence ----

    def save(self, path: str | Path) -> None:
        """Save book to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "session_id": self.session_id,
            "turn_counter": self._turn_counter,
            "sections": [
                {
                    "index": s.index,
                    "role": s.role,
                    "turn": s.turn,
                    "content": s.content,
                    "timestamp": s.timestamp,
                    "token_count": s.token_count,
                }
                for s in self.sections
            ],
        }
        path.write_text(json.dumps(data, ensure_ascii=False))
        logger.debug(f"Book saved: {len(self.sections)} sections to {path}")

    @classmethod
    def load(cls, path: str | Path) -> ConversationBook:
        """Load book from JSON file."""
        path = Path(path)
        data = json.loads(path.read_text())
        book = cls(session_id=data.get("session_id", ""))
        book._turn_counter = data.get("turn_counter", 0)
        for s_data in data.get("sections", []):
            section = BookSection(
                index=s_data["index"],
                role=s_data["role"],
                turn=s_data["turn"],
                content=s_data["content"],
                timestamp=s_data.get("timestamp", 0),
                token_count=s_data.get("token_count", 0),
            )
            book.sections.append(section)
        return book
