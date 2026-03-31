"""Compaction engine — orchestrates the two-stage pipeline.

Stage 1 (DCP): Prune duplicates, superseded writes, truncate errors (no LLM)
Stage 2 (Summarize): If still over threshold, LLM-summarize compactible region

The engine is stateless — it takes messages + config, returns compacted messages.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ctxpact.compaction.detector import SequenceDetector
from ctxpact.compaction.pruner import DynamicContextPruner
from ctxpact.compaction.summarizer import Summarizer
from ctxpact.compaction.tokens import count_messages_tokens
from ctxpact.config import CompactionConfig

logger = logging.getLogger(__name__)


@dataclass
class CompactionResult:
    """Outcome of a compaction pass."""

    messages: list[dict[str, Any]]
    compacted: bool
    stage: str  # "none" | "dcp_only" | "dcp_and_summarize" | "summarize_only"
    tokens_before: int
    tokens_after: int
    dcp_tokens_saved: int
    messages_before: int
    messages_after: int


class CompactionEngine:
    """Orchestrates DCP + Summarization pipeline."""

    def __init__(self, config: CompactionConfig) -> None:
        self.config = config
        self.pruner = DynamicContextPruner(config.stage1_dcp)
        self.detector = SequenceDetector(
            retention_window=config.stage2_summarize.retention_window,
            eviction_window=config.stage2_summarize.eviction_window,
            merge_strategy=config.stage2_summarize.merge_strategy,
            preserve_config=config.preserve,
        )
        self.summarizer = Summarizer(
            max_summary_tokens=config.stage2_summarize.max_summary_tokens,
            code_line_limit=config.preserve.code_blocks_under_lines,
        )

    def should_compact(
        self,
        messages: list[dict[str, Any]],
        max_context: int,
        user_turn_count: int = 0,
    ) -> tuple[bool, str]:
        """Check if compaction should trigger based on configured thresholds.

        Returns (should_compact, reason).
        """
        triggers = self.config.triggers

        # Token ratio trigger
        if triggers.token_ratio:
            token_count = count_messages_tokens(messages)
            threshold = int(max_context * triggers.token_ratio)
            if token_count >= threshold:
                return True, f"token_count={token_count} >= threshold={threshold}"

        # Message count trigger
        if triggers.message_count and len(messages) >= triggers.message_count:
            return True, f"message_count={len(messages)} >= {triggers.message_count}"

        # Turn count trigger
        if triggers.turn_count and user_turn_count >= triggers.turn_count:
            return True, f"turn_count={user_turn_count} >= {triggers.turn_count}"

        return False, ""

    async def compact(
        self,
        messages: list[dict[str, Any]],
        backend_url: str,
        model: str,
        max_context: int,
        api_key: str = "dummy",
    ) -> CompactionResult:
        """Run the two-stage compaction pipeline.

        1. DCP (prune) — cheap, no LLM call
        2. Check if still over threshold
        3. Summarize if needed — LLM call
        4. Reconstruct: [system] + [summary] + [retained] + [preserved_user]
        """
        tokens_before = count_messages_tokens(messages)
        messages_before = len(messages)
        stage = "none"
        dcp_tokens_saved = 0

        # ---- Stage 1: Dynamic Context Pruning ----
        if self.config.stage1_dcp.enabled:
            prune_result = self.pruner.prune(messages)
            working_messages = prune_result.messages
            dcp_tokens_saved = prune_result.tokens_saved_estimate

            logger.info(
                f"DCP: deduped={prune_result.deduped_tool_calls}, "
                f"superseded={prune_result.superseded_writes}, "
                f"truncated={prune_result.truncated_errors}, "
                f"stripped={prune_result.stripped_payloads}"
            )
            stage = "dcp_only"
        else:
            working_messages = messages

        # Check if DCP was enough
        tokens_after_dcp = count_messages_tokens(working_messages)
        threshold = int(max_context * self.config.triggers.token_ratio)

        if tokens_after_dcp < threshold:
            logger.info(
                f"DCP sufficient: {tokens_before} → {tokens_after_dcp} tokens "
                f"(threshold={threshold})"
            )
            return CompactionResult(
                messages=working_messages,
                compacted=True,
                stage=stage,
                tokens_before=tokens_before,
                tokens_after=tokens_after_dcp,
                dcp_tokens_saved=dcp_tokens_saved,
                messages_before=messages_before,
                messages_after=len(working_messages),
            )

        # ---- Stage 2: LLM Summarization ----
        logger.info(
            f"DCP not sufficient ({tokens_after_dcp} >= {threshold}), "
            "proceeding to LLM summarization"
        )

        # Classify messages
        classification = self.detector.classify(working_messages)

        if not classification.compactible_messages:
            logger.warning("No compactible messages found — skipping summarization")
            return CompactionResult(
                messages=working_messages,
                compacted=False,
                stage=stage,
                tokens_before=tokens_before,
                tokens_after=tokens_after_dcp,
                dcp_tokens_saved=dcp_tokens_saved,
                messages_before=messages_before,
                messages_after=len(working_messages),
            )

        # Summarize the compactible region
        summary_msg = await self.summarizer.summarize(
            messages=classification.compactible_messages
            + classification.preserved_user_messages,
            backend_url=backend_url,
            model=model,
            api_key=api_key,
        )

        # Reconstruct: [system] + [summary] + [retention_window]
        reconstructed: list[dict[str, Any]] = []
        reconstructed.extend(classification.system_messages)
        reconstructed.append(summary_msg)
        reconstructed.extend(classification.retention_window)

        tokens_after = count_messages_tokens(reconstructed)
        stage = "dcp_and_summarize" if self.config.stage1_dcp.enabled else "summarize_only"

        logger.info(
            f"Compaction complete: {tokens_before} → {tokens_after} tokens, "
            f"{messages_before} → {len(reconstructed)} messages"
        )

        return CompactionResult(
            messages=reconstructed,
            compacted=True,
            stage=stage,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            dcp_tokens_saved=dcp_tokens_saved,
            messages_before=messages_before,
            messages_after=len(reconstructed),
        )
