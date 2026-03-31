"""ctxpact — FastAPI server with OpenAI-compatible endpoints.

This is the main entry point. It:
  1. Accepts OpenAI-format chat completion requests
  2. Manages per-conversation sessions
  3. Runs the two-stage compaction pipeline when thresholds are hit
  4. Routes to the best available provider (with circuit breaker failover)
  5. Streams responses back to the client
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from ctxpact.compaction.engine import CompactionEngine
from ctxpact.config import CtxpactConfig, load_config
from ctxpact.isolation.graph_manager import GraphManager
from ctxpact.isolation.isolator import isolate_context
from ctxpact.routing.health import HealthChecker
from ctxpact.routing.router import ProviderRouter
from ctxpact.session.models import CompactionEvent, Message, MessageRole, Session
from ctxpact.session.store import MemorySessionStore, SessionStore, SqliteSessionStore

logger = logging.getLogger("ctxpact")


# ---------------------------------------------------------------------------
# Application State
# ---------------------------------------------------------------------------

class AppState:
    """Holds all runtime dependencies."""

    def __init__(self, config: CtxpactConfig) -> None:
        self.config = config
        self.router = ProviderRouter(config.providers, config.circuit_breaker)
        self.compaction = CompactionEngine(config.compaction) if config.compaction.enabled else None
        self.health_checker = HealthChecker()

        # GOG context isolation
        iso = config.context_isolation
        self.graph_manager: GraphManager | None = None
        if iso.enabled and iso.repo_path:
            self.graph_manager = GraphManager(
                repo_path=iso.repo_path,
                cache_path=iso.graph_cache_path,
                include_extensions=iso.include_extensions,
                exclude_dirs=iso.exclude_dirs,
                rebuild_on_startup=iso.rebuild_on_startup,
            )

        # Session store
        if config.session.store == "sqlite":
            self.sessions: SessionStore = SqliteSessionStore(config.session.db_path)
        else:
            self.sessions = MemorySessionStore()

    async def start(self) -> None:
        await self.health_checker.start(self.config.providers, self.router.breakers)

    async def stop(self) -> None:
        await self.health_checker.stop()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    """Start/stop background tasks."""
    state: AppState = app.state.app_state
    # Initialize GOG graph if enabled
    if state.graph_manager:
        graph = await state.graph_manager.ensure_graph()
        logger.info(
            f"GOG graph ready: {graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} edges"
        )
    await state.start()
    logger.info(
        f"ctxpact started — {len(state.config.providers)} providers, "
        f"compaction={'enabled' if state.config.compaction.enabled else 'disabled'}, "
        f"gog={'enabled' if state.graph_manager else 'disabled'}"
    )
    yield
    await state.stop()
    logger.info("ctxpact stopped")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

def create_app(config: CtxpactConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if config is None:
        config = load_config()

    app = FastAPI(
        title="ctxpact",
        description="Resilient context compaction gateway for local LLM inference",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.app_state = AppState(config)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---- Routes ----

    @app.get("/health")
    async def health() -> dict:
        state: AppState = app.state.app_state
        return {
            "status": "ok",
            "providers": state.router.status(),
            "compaction_enabled": state.config.compaction.enabled,
        }

    @app.get("/v1/models")
    async def list_models() -> dict:
        """List available models (OpenAI-compatible)."""
        state: AppState = app.state.app_state
        models = []
        for p in state.config.providers:
            models.append({
                "id": p.model,
                "object": "model",
                "created": int(time.time()),
                "owned_by": p.name,
            })
        return {"object": "list", "data": models}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        """OpenAI-compatible chat completions endpoint.

        Handles:
          - Session management (via X-Session-ID header or metadata)
          - Compaction checks and execution
          - Provider routing with failover
          - Streaming and non-streaming responses
        """
        state: AppState = app.state.app_state
        body = await request.json()

        messages = body.get("messages", [])
        stream = body.get("stream", False)

        # Session management
        session_id = (
            request.headers.get("X-Session-ID")
            or body.get("metadata", {}).get("session_id")
            or body.get("session_id")
            or str(uuid.uuid4())
        )

        session = await state.sessions.get(session_id)
        if session is None:
            session = Session(session_id=session_id)

        # Merge incoming messages into session.
        # Most OpenAI-compatible clients send FULL history on every request.
        # If incoming messages >= existing count, treat as full replacement
        # to avoid doubling messages. Otherwise append only new messages.
        if messages:
            existing_count = session.message_count
            if len(messages) >= existing_count:
                # Client sent full history — replace session state
                session.messages = []
                session.user_turn_count = 0
                for msg_dict in messages:
                    session.append_message(Message.from_openai_dict(msg_dict))
            else:
                # Client sent only new messages — append
                for msg_dict in messages:
                    session.append_message(Message.from_openai_dict(msg_dict))

        # Build the messages to send
        outgoing_messages = session.get_openai_messages()

        # ---- GOG Context Isolation (runs before compaction) ----
        if state.graph_manager:
            try:
                graph = await state.graph_manager.ensure_graph()
                # Extract the latest user prompt for seed finding
                user_prompt = ""
                for msg in reversed(outgoing_messages):
                    if msg.get("role") == "user":
                        content = msg.get("content", "")
                        if isinstance(content, str):
                            user_prompt = content
                        break

                if user_prompt:
                    iso_result = isolate_context(
                        graph=graph,
                        messages=outgoing_messages,
                        prompt=user_prompt,
                        repo_path=state.config.context_isolation.repo_path or "",
                        min_file_lines=state.config.context_isolation.min_file_lines,
                    )
                    if iso_result.applied:
                        outgoing_messages = iso_result.messages
                        logger.info(
                            f"[{session_id[:8]}] GOG: kept {len(iso_result.isolated_files)} files, "
                            f"stripped {len(iso_result.stripped_files)}, "
                            f"~{iso_result.tokens_before - iso_result.tokens_after} tokens saved"
                        )
            except Exception as e:
                logger.warning(f"GOG isolation failed (continuing without): {e}")

        # ---- Compaction Check ----
        if state.compaction:
            active_provider = state.router.get_active_provider()
            should_compact, reason = state.compaction.should_compact(
                messages=outgoing_messages,
                max_context=active_provider.max_context,
                user_turn_count=session.user_turn_count,
            )

            if should_compact:
                logger.info(
                    f"[{session_id[:8]}] Compaction triggered: {reason}"
                )
                try:
                    summarize_provider = active_provider
                    override_provider = state.config.compaction.stage2_summarize.provider
                    if override_provider:
                        match = next(
                            (p for p in state.config.providers if p.name == override_provider),
                            None,
                        )
                        if match:
                            summarize_provider = match
                        else:
                            logger.warning(
                                f"Summarize provider '{override_provider}' not found; "
                                f"using active provider {active_provider.name}"
                            )

                    summarize_model = (
                        state.config.compaction.stage2_summarize.model
                        or summarize_provider.model
                    )

                    result = await state.compaction.compact(
                        messages=outgoing_messages,
                        backend_url=summarize_provider.url,
                        model=summarize_model,
                        max_context=summarize_provider.max_context,
                        api_key=summarize_provider.api_key,
                    )

                    if result.compacted:
                        outgoing_messages = result.messages
                        # Update session with compacted history
                        session.messages = [
                            Message.from_openai_dict(m) for m in outgoing_messages
                        ]
                        session.compaction_events.append(CompactionEvent(
                            trigger_reason=reason,
                            tokens_before=result.tokens_before,
                            tokens_after=result.tokens_after,
                            messages_before=result.messages_before,
                            messages_after=result.messages_after,
                            dcp_tokens_saved=result.dcp_tokens_saved,
                            stage=result.stage,
                        ))
                        logger.info(
                            f"[{session_id[:8]}] Compacted: "
                            f"{result.tokens_before}→{result.tokens_after} tokens, "
                            f"{result.messages_before}→{result.messages_after} msgs, "
                            f"stage={result.stage}"
                        )

                except Exception as e:
                    logger.error(f"Compaction failed: {e}", exc_info=True)
                    # Continue with uncompacted messages — don't block the request

        # ---- Oversized Input Handling (RLM or Chunking) ----
        from ctxpact.compaction.tokens import count_messages_tokens

        active_provider = state.router.get_active_provider()
        max_ctx = active_provider.max_context
        max_tokens_param = body.get("max_tokens", 1024)
        input_budget = int(max_ctx * 0.85) - max_tokens_param

        total_tokens = count_messages_tokens(outgoing_messages)
        if total_tokens > input_budget:
            oversized_cfg = state.config.compaction.oversized
            strategy = oversized_cfg.strategy

            logger.info(
                f"[{session_id[:8]}] Input oversized: {total_tokens} tokens > "
                f"{input_budget} budget. Strategy: {strategy}"
            )

            if strategy in ("header", "autosearch", "rlm", "rlm_v2", "rlm_v3", "toolcall",
                           "embed", "compress", "adaptive", "icl", "rlm_v4", "rlm_v5",
                           "rlm_v6", "agentic", "subagent", "readagent"):
                # ---- Book-based extraction (header/autosearch/rlm/toolcall) ----
                from ctxpact.compaction.rlm_extractor import get_extractor

                # Build book from current messages if not already populated
                if session.book.section_count == 0:
                    session.book.build_from_messages(outgoing_messages)

                # Extract user query (last user message)
                user_query = ""
                for msg in reversed(outgoing_messages):
                    if msg.get("role") == "user":
                        user_query = msg.get("content", "")
                        if isinstance(user_query, str):
                            break
                        user_query = ""

                try:
                    extractor = get_extractor(
                        strategy=strategy,
                        provider_url=active_provider.url,
                        model=active_provider.model,
                        api_key=active_provider.api_key,
                        max_context=active_provider.max_context,
                        max_iterations=oversized_cfg.rlm_max_iterations,
                        max_llm_calls=oversized_cfg.rlm_max_llm_calls,
                    )
                    extracted = await extractor.extract(
                        book=session.book,
                        query=user_query,
                        token_budget=input_budget,
                    )
                    # Rebuild messages: system + extracted context + short query
                    # The full document is already in the extracted context,
                    # so only include the question/instruction part of user_query
                    short_query = user_query[:2000]
                    if len(user_query) > 2000:
                        short_query += "\n\n[Full document content was extracted above]"

                    system_msgs = [
                        m for m in outgoing_messages if m.get("role") == "system"
                    ]
                    outgoing_messages = list(system_msgs)
                    if extracted:
                        outgoing_messages.append({
                            "role": "user",
                            "content": extracted + "\n\n---\n\n" + short_query,
                        })
                    else:
                        outgoing_messages.append({
                            "role": "user", "content": short_query,
                        })

                    logger.info(
                        f"[{session_id[:8]}] {strategy} extraction complete: "
                        f"{total_tokens}→{count_messages_tokens(outgoing_messages)} tokens"
                    )
                except Exception as e:
                    logger.error(
                        f"{strategy} extraction failed, falling back to chunking: {e}",
                        exc_info=True,
                    )
                    strategy = "chunking"  # Fall through to chunking

            if strategy == "chunking":
                # ---- Chunked map-reduce fallback ----
                from ctxpact.compaction.chunker import ChunkedProcessor

                chunker = ChunkedProcessor()
                try:
                    outgoing_messages = await chunker.process(
                        messages=outgoing_messages,
                        input_budget=input_budget,
                        router=state.router,
                        forward_kwargs={
                            k: body[k] for k in ("temperature", "top_p") if k in body
                        },
                    )
                    logger.info(
                        f"[{session_id[:8]}] Chunked processing complete: "
                        f"{total_tokens}→{count_messages_tokens(outgoing_messages)} tokens"
                    )
                except Exception as e:
                    logger.error(f"Chunked processing failed: {e}", exc_info=True)

        # ---- Forward to Provider ----
        # Strip session-specific fields, forward standard OpenAI params
        forward_kwargs: dict[str, Any] = {}
        for key in ("max_tokens", "temperature", "top_p", "stop", "tools",
                     "tool_choice", "response_format", "seed"):
            if key in body:
                forward_kwargs[key] = body[key]

        try:
            if stream:
                sse_iter, provider = await state.router.chat_completion_stream(
                    messages=outgoing_messages, **forward_kwargs
                )

                async def stream_with_session():
                    """Wrap SSE stream, capture assistant response for session."""
                    full_content = []
                    async for line in sse_iter:
                        yield f"{line}\n\n"
                        # Try to extract content delta for session tracking
                        if line.startswith("data: ") and line != "data: [DONE]":
                            try:
                                chunk = json.loads(line[6:])
                                delta = chunk.get("choices", [{}])[0].get("delta", {})
                                c = delta.get("content", "")
                                if c:
                                    full_content.append(c)
                            except (json.JSONDecodeError, IndexError):
                                pass

                    # Save assistant response to session
                    if full_content:
                        assistant_msg = Message(
                            role=MessageRole.ASSISTANT,
                            content="".join(full_content),
                        )
                        session.append_message(assistant_msg)
                        await state.sessions.put(session)

                return StreamingResponse(
                    stream_with_session(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Session-ID": session_id,
                        "X-Provider": provider.name,
                    },
                )

            else:
                response, provider = await state.router.chat_completion(
                    messages=outgoing_messages, **forward_kwargs
                )

                # Save assistant response to session
                choice = response.get("choices", [{}])[0]
                assistant_content = choice.get("message", {}).get("content", "")
                if assistant_content:
                    session.append_message(Message(
                        role=MessageRole.ASSISTANT,
                        content=assistant_content,
                        tool_calls=choice.get("message", {}).get("tool_calls"),
                    ))

                # Track token usage
                usage = response.get("usage", {})
                session.total_input_tokens += usage.get("prompt_tokens", 0)
                session.total_output_tokens += usage.get("completion_tokens", 0)

                await state.sessions.put(session)

                # Add session metadata to response
                response["_ctxpact"] = {
                    "session_id": session_id,
                    "provider": provider.name,
                    "compaction_events": len(session.compaction_events),
                    "message_count": session.message_count,
                }

                return JSONResponse(
                    content=response,
                    headers={
                        "X-Session-ID": session_id,
                        "X-Provider": provider.name,
                    },
                )

        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            logger.error(f"Request failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/v1/sessions/{session_id}")
    async def get_session(session_id: str) -> dict:
        """Inspect a session's state (debug endpoint)."""
        state: AppState = app.state.app_state
        session = await state.sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return {
            "session_id": session.session_id,
            "message_count": session.message_count,
            "user_turn_count": session.user_turn_count,
            "total_input_tokens": session.total_input_tokens,
            "total_output_tokens": session.total_output_tokens,
            "compaction_events": [
                {
                    "timestamp": e.timestamp,
                    "trigger": e.trigger_reason,
                    "tokens_before": e.tokens_before,
                    "tokens_after": e.tokens_after,
                    "stage": e.stage,
                }
                for e in session.compaction_events
            ],
        }

    @app.get("/v1/graph/status")
    async def graph_status() -> dict:
        """GOG dependency graph status (debug endpoint)."""
        state: AppState = app.state.app_state
        if not state.graph_manager:
            return {"enabled": False}
        return {"enabled": True, **state.graph_manager.status()}

    @app.post("/v1/graph/rebuild")
    async def graph_rebuild() -> dict:
        """Force rebuild of the GOG dependency graph."""
        state: AppState = app.state.app_state
        if not state.graph_manager:
            raise HTTPException(status_code=400, detail="GOG not enabled")
        graph = state.graph_manager.rebuild()
        return {
            "status": "rebuilt",
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
        }

    @app.delete("/v1/sessions/{session_id}")
    async def delete_session(session_id: str) -> dict:
        """Delete a session."""
        state: AppState = app.state.app_state
        await state.sessions.delete(session_id)
        return {"status": "deleted", "session_id": session_id}

    return app


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the server from CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="ctxpact — context compaction gateway")
    parser.add_argument("--config", "-c", default="config.yaml", help="Config file path")
    parser.add_argument("--host", default=None, help="Override host")
    parser.add_argument("--port", "-p", type=int, default=None, help="Override port")
    parser.add_argument("--local", action="store_true",
                        help="Local-only mode: remove all external providers, never call OpenRouter etc.")
    parser.add_argument("--strategy", choices=["header", "autosearch", "rlm", "rlm_v2", "rlm_v3", "toolcall", "chunking",
                                              "embed", "compress", "adaptive", "icl", "rlm_v4", "rlm_v5", "rlm_v6", "agentic", "subagent", "readagent"],
                        default=None,
                        help="Override oversized input strategy (default: from config)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.host:
        config.server.host = args.host
    if args.port:
        config.server.port = args.port

    if args.local:
        # Keep only the highest-priority (local) provider
        if config.providers:
            local_provider = min(config.providers, key=lambda p: p.priority)
            removed = [p.name for p in config.providers if p.name != local_provider.name]
            config.providers = [local_provider]
            # Also clear summarize provider override if it references a removed provider
            if config.compaction.stage2_summarize.provider in removed:
                config.compaction.stage2_summarize.provider = None
            logger.info(
                f"Local-only mode: keeping {local_provider.name}, "
                f"removed {removed}"
            )

    if args.strategy:
        config.compaction.oversized.strategy = args.strategy
        logger.info(f"Strategy override: {args.strategy}")

    logging.basicConfig(
        level=getattr(logging, config.server.log_level.upper()),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    app = create_app(config)
    uvicorn.run(app, host=config.server.host, port=config.server.port)


if __name__ == "__main__":
    main()
