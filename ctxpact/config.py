"""Configuration management — loads from config.yaml with env var interpolation."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class HealthCheckConfig(BaseModel):
    endpoint: str = "/health"
    interval_seconds: int = 10
    timeout_seconds: int = 3


class ProviderConfig(BaseModel):
    name: str
    url: str
    model: str
    api_key: str = "dummy"
    max_context: int = 32000
    priority: int = 1
    timeout_seconds: float = 180.0
    connect_timeout_seconds: float = 5.0
    stream_timeout_seconds: float | None = None
    health_check: HealthCheckConfig = Field(default_factory=HealthCheckConfig)


class CircuitBreakerConfig(BaseModel):
    failure_threshold: int = 3
    recovery_timeout_seconds: int = 30
    half_open_max_calls: int = 1


class DcpConfig(BaseModel):
    """Stage 1: Dynamic Context Pruning (no LLM call)."""

    enabled: bool = True
    dedup_tool_calls: bool = True
    strip_superseded_writes: bool = True
    truncate_errors: bool = True
    strip_tool_payloads: bool = True


class SummarizeConfig(BaseModel):
    """Stage 2: LLM-based summarization."""

    provider: str | None = None  # None = use active provider
    model: str | None = None  # None = use active provider
    max_summary_tokens: int = 2000
    retention_window: int = 6
    eviction_window: float = 0.30
    merge_strategy: str = "conservative"
    parallel: bool = True


class TriggersConfig(BaseModel):
    token_ratio: float = 0.70
    message_count: int | None = None
    turn_count: int | None = None


class PreserveConfig(BaseModel):
    system_prompts: bool = True
    user_messages: bool = True
    code_blocks_under_lines: int = 50
    file_paths: bool = True
    error_messages: bool = True


class OversizedConfig(BaseModel):
    """Strategy for handling input that exceeds context window.

    Strategies:
      "header"     — Section previews + recent full sections (no LLM, fast)
      "autosearch" — Heuristic keyword extraction → grep → assemble (no LLM)
      "rlm"        — LLM generates search terms → grep → assemble (1 LLM call)
      "rlm_v2"     — Fixed RLM: word matching, IDF ranking, no summary bloat (1 LLM call)
      "rlm_v3"     — DSPy RLM: model writes Python to search the book (N LLM calls)
      "toolcall"   — Multi-turn tool-calling loop (N LLM calls)
      "chunking"   — Map-reduce chunked processing (separate ChunkedProcessor)
    """

    strategy: str = "rlm_v2"  # "header" | "autosearch" | "rlm" | "rlm_v2" | "rlm_v3" | "toolcall" | "chunking"
    rlm_max_iterations: int = 10
    rlm_max_llm_calls: int = 15
    book_storage: str = "memory"  # "memory" | "disk"
    book_path: str = "./books/"


class CompactionConfig(BaseModel):
    enabled: bool = True
    triggers: TriggersConfig = Field(default_factory=TriggersConfig)
    stage1_dcp: DcpConfig = Field(default_factory=DcpConfig)
    stage2_summarize: SummarizeConfig = Field(default_factory=SummarizeConfig)
    preserve: PreserveConfig = Field(default_factory=PreserveConfig)
    oversized: OversizedConfig = Field(default_factory=OversizedConfig)


class ContextIsolationConfig(BaseModel):
    """GOG (Graph-Oriented Generation) context isolation."""

    enabled: bool = False
    repo_path: str | None = None
    graph_cache_path: str = "./gog_cache.pkl"
    include_extensions: list[str] | None = None  # None = auto-detect
    exclude_dirs: list[str] = Field(
        default_factory=lambda: [
            "node_modules", ".venv", "venv", "__pycache__", ".git",
            "dist", "build", ".mypy_cache", ".pytest_cache",
        ]
    )
    rebuild_on_startup: bool = False
    min_file_lines: int = 10  # Only strip file contents larger than this
    debug: bool = False


class SessionConfig(BaseModel):
    store: str = "memory"  # "memory" | "sqlite"
    db_path: str = "./ctxpact_sessions.db"
    ttl_hours: int = 24


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

class CtxpactConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    providers: list[ProviderConfig] = Field(default_factory=list)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    compaction: CompactionConfig = Field(default_factory=CompactionConfig)
    context_isolation: ContextIsolationConfig = Field(default_factory=ContextIsolationConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_ENV_VAR_PATTERN = re.compile(r"\$\{(\w+)\}")


def _interpolate_env(obj: Any) -> Any:
    """Recursively replace ${VAR_NAME} with environment variable values."""
    if isinstance(obj, str):
        return _ENV_VAR_PATTERN.sub(lambda m: os.environ.get(m.group(1), m.group(0)), obj)
    if isinstance(obj, dict):
        return {k: _interpolate_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interpolate_env(v) for v in obj]
    return obj


def load_config(path: str | Path = "config.yaml") -> CtxpactConfig:
    """Load configuration from YAML file with env var interpolation."""
    config_path = Path(path)
    if config_path.exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f)
        raw = _interpolate_env(raw or {})
        return CtxpactConfig(**raw)

    # Fallback: sensible defaults for local dev
    return CtxpactConfig(
        providers=[
            ProviderConfig(
                name="local",
                url="http://localhost:8080/v1",
                model="Nanbeige/Nanbeige4.1-3B",
                priority=1,
            )
        ]
    )
