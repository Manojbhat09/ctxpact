"""Data models for sessions, messages, and compaction events."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ctxpact.compaction.book import ConversationBook


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """A single message in a conversation."""

    role: MessageRole
    content: Any  # str | list[dict] for multi-part
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    timestamp: float = field(default_factory=time.time)
    is_summary: bool = False
    token_estimate: int = 0

    def to_openai_dict(self) -> dict[str, Any]:
        """Convert to OpenAI API message format."""
        msg: dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.name:
            msg["name"] = self.name
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        return msg

    @classmethod
    def from_openai_dict(cls, data: dict[str, Any]) -> Message:
        """Create from OpenAI API message format."""
        return cls(
            role=MessageRole(data["role"]),
            content=data.get("content", ""),
            name=data.get("name"),
            tool_calls=data.get("tool_calls"),
            tool_call_id=data.get("tool_call_id"),
        )


@dataclass
class CompactionEvent:
    """Record of a compaction that occurred."""

    timestamp: float = field(default_factory=time.time)
    trigger_reason: str = ""
    tokens_before: int = 0
    tokens_after: int = 0
    messages_before: int = 0
    messages_after: int = 0
    dcp_tokens_saved: int = 0
    summary_tokens_used: int = 0
    stage: str = ""  # "dcp_only" | "dcp_and_summarize" | "summarize_only"


@dataclass
class Session:
    """Per-conversation state."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: list[Message] = field(default_factory=list)
    compaction_events: list[CompactionEvent] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    user_turn_count: int = 0
    _book: Any = field(default=None, repr=False)

    @property
    def book(self) -> "ConversationBook":
        """Lazy-initialized conversation book."""
        if self._book is None:
            from ctxpact.compaction.book import ConversationBook
            self._book = ConversationBook(session_id=self.session_id)
        return self._book

    def append_message(self, msg: Message) -> None:
        self.messages.append(msg)
        self.last_active = time.time()
        if msg.role == MessageRole.USER and not msg.is_summary:
            self.user_turn_count += 1
        # Also append to the conversation book
        content = msg.content
        if isinstance(content, str) and content.strip():
            self.book.append_message(msg.role.value, content)

    def get_openai_messages(self) -> list[dict[str, Any]]:
        return [m.to_openai_dict() for m in self.messages]

    @property
    def message_count(self) -> int:
        return len(self.messages)
