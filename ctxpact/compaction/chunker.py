"""Chunked processing for oversized inputs via map-reduce.

When a user message exceeds the model's context window, this module:

1. SPLIT: Breaks the content into overlapping chunks at paragraph/sentence
   boundaries. Chunks are sized so that chunk + map prompt fits in context.
2. MAP: Sends each chunk sequentially to the model with an extraction prompt,
   getting back condensed key information from that chunk.
3. REDUCE: Combines all chunk extractions into a single condensed context
   message, paired with the user's original question for the final response.

This preserves information across the full input rather than arbitrarily
truncating, at the cost of additional inference calls.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from ctxpact.compaction.tokens import count_tokens, count_messages_tokens

logger = logging.getLogger(__name__)

# Overlap between chunks to avoid losing context at boundaries
OVERLAP_RATIO = 0.05
# Reserve tokens for the map system+user prompt framing
MAP_PROMPT_OVERHEAD = 200
# Target chunk size — small enough to leave room for a generous extraction.
# Use ~40% of context for the chunk, leaving 60% for prompt overhead + response.
CHUNK_CONTEXT_RATIO = 0.40
# Map response should be proportional to chunk size to avoid over-compression.
# Target ~25% of chunk tokens as extraction length.
MAP_EXTRACTION_RATIO = 0.25
# Max concurrent map requests (avoid rate limits on free tiers)
MAX_CONCURRENCY = 2
# Retry config for failed chunks
MAX_RETRIES = 2
RETRY_DELAY_S = 3.0


class ChunkedProcessor:
    """Map-reduce processor for oversized inputs."""

    def _extract_question(self, content: str) -> tuple[str, str]:
        """Separate the user's question from the bulk content.

        Heuristic: the actual question is usually at the end, after
        the pasted content. Look for common patterns like a trailing
        question, instruction after a separator, etc.
        """
        paragraphs = content.strip().split("\n\n")
        if len(paragraphs) < 2:
            lines = content.strip().split("\n")
            if len(lines) > 5:
                question_lines = []
                for line in reversed(lines):
                    line_s = line.strip()
                    if not line_s:
                        continue
                    question_lines.insert(0, line)
                    if len(question_lines) >= 3:
                        break
                question = "\n".join(question_lines)
                bulk = "\n".join(lines[: -len(question_lines)])
                return bulk, question
            return content, ""

        last = paragraphs[-1].strip()
        if len(last) < 500 and (
            "?" in last
            or last.lower().startswith(("summarize", "explain", "what", "how",
                                        "why", "can you", "please", "tell me",
                                        "analyze", "review", "describe"))
        ):
            bulk = "\n\n".join(paragraphs[:-1])
            return bulk, last

        return content, ""

    def _split_into_chunks(self, text: str, chunk_token_budget: int) -> list[str]:
        """Split text into chunks at paragraph boundaries."""
        chars_per_token = max(len(text) / max(count_tokens(text), 1), 2.5)
        chunk_char_budget = int(chunk_token_budget * chars_per_token)
        overlap_chars = int(chunk_char_budget * OVERLAP_RATIO)

        paragraphs = re.split(r"\n\n+", text)

        chunks: list[str] = []
        current_chunk: list[str] = []
        current_len = 0

        for para in paragraphs:
            para_len = len(para)

            if current_len + para_len > chunk_char_budget and current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                chunks.append(chunk_text)

                overlap_text = chunk_text[-overlap_chars:] if overlap_chars > 0 else ""
                newline_pos = overlap_text.find("\n")
                if newline_pos > 0:
                    overlap_text = overlap_text[newline_pos + 1:]

                current_chunk = [overlap_text] if overlap_text else []
                current_len = len(overlap_text)

            current_chunk.append(para)
            current_len += para_len + 2

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    async def _map_chunk_with_retry(
        self,
        chunk: str,
        chunk_index: int,
        total_chunks: int,
        question: str,
        router: Any,
        forward_kwargs: dict[str, Any],
        max_extraction_tokens: int,
    ) -> str:
        """Send a single chunk to the model with retries."""
        chunk_tokens = count_tokens(chunk)
        context = (
            f"You are extracting key information from section {chunk_index + 1} "
            f"of {total_chunks} of a large document."
        )
        if question:
            map_prompt = (
                f"{context}\n\n"
                f"The user's question about this document is: \"{question}\"\n\n"
                f"Extract ALL key information from this section that could be "
                f"relevant. Preserve specific details: names, numbers, dates, "
                f"chapter/section titles, lists, code snippets, and structure. "
                f"Do NOT answer the question — just extract information.\n\n"
                f"--- SECTION {chunk_index + 1}/{total_chunks} ---\n{chunk}"
            )
        else:
            map_prompt = (
                f"{context}\n\n"
                f"Extract and condense the key information from this section. "
                f"Preserve specific details: names, numbers, dates, "
                f"chapter/section titles, lists, code snippets, and structure. "
                f"Be thorough — do not skip sections or summarize too aggressively.\n\n"
                f"--- SECTION {chunk_index + 1}/{total_chunks} ---\n{chunk}"
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise information extractor. Your job is to "
                    "condense document sections while preserving ALL important "
                    "details. Never skip items in lists. Never omit headings or "
                    "structure. Err on the side of including too much rather "
                    "than too little."
                ),
            },
            {"role": "user", "content": map_prompt},
        ]

        for attempt in range(MAX_RETRIES + 1):
            try:
                response, provider = await router.chat_completion(
                    messages=messages,
                    max_tokens=max_extraction_tokens,
                    **forward_kwargs,
                )
                content = (
                    response.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                logger.info(
                    f"Chunk {chunk_index + 1}/{total_chunks}: "
                    f"{chunk_tokens} tokens → {count_tokens(content)} tokens "
                    f"(via {provider.name})"
                )
                return content
            except Exception as e:
                if attempt < MAX_RETRIES:
                    delay = RETRY_DELAY_S * (attempt + 1)
                    logger.warning(
                        f"Chunk {chunk_index + 1} attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Chunk {chunk_index + 1} failed after {MAX_RETRIES + 1} attempts: {e}"
                    )
                    # Last resort: return a meaningful portion of the chunk
                    # rather than almost nothing
                    target_chars = max_extraction_tokens * 3
                    if len(chunk) <= target_chars:
                        return chunk
                    # Keep start and end with a marker
                    half = target_chars // 2
                    return (
                        chunk[:half]
                        + f"\n[... section {chunk_index + 1}: extraction failed, "
                        f"{len(chunk) - target_chars} chars omitted ...]\n"
                        + chunk[-half:]
                    )

    async def process(
        self,
        messages: list[dict[str, Any]],
        input_budget: int,
        router: Any,
        forward_kwargs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Process oversized messages via chunked map-reduce."""
        system_msgs = [m for m in messages if m.get("role") == "system"]
        other_msgs = [m for m in messages if m.get("role") != "system"]

        system_tokens = sum(count_tokens(m.get("content", "")) for m in system_msgs)
        content_budget = input_budget - system_tokens - 100

        # Find the large user message
        large_msg_idx = None
        large_content = ""
        for i, msg in enumerate(other_msgs):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str) and count_tokens(content) > content_budget:
                    large_msg_idx = i
                    large_content = content

        if large_msg_idx is None:
            return messages

        # Separate question from bulk content
        bulk, question = self._extract_question(large_content)
        bulk_tokens = count_tokens(bulk)
        logger.info(
            f"Chunked: bulk={bulk_tokens} tokens, "
            f"question={count_tokens(question) if question else 0} tokens"
        )

        # Calculate chunk size: use CHUNK_CONTEXT_RATIO of the provider's context
        # so there's ample room for the prompt + extraction response
        active_provider = router.get_active_provider()
        max_ctx = active_provider.max_context
        chunk_token_budget = int(max_ctx * CHUNK_CONTEXT_RATIO) - MAP_PROMPT_OVERHEAD

        # Extraction response tokens: proportional to chunk size
        max_extraction_tokens = max(
            int(chunk_token_budget * MAP_EXTRACTION_RATIO), 1024
        )

        # Split bulk content into chunks
        chunks = self._split_into_chunks(bulk, chunk_token_budget)
        logger.info(
            f"Split into {len(chunks)} chunks "
            f"(budget: {chunk_token_budget} tokens/chunk, "
            f"extraction: {max_extraction_tokens} tokens/chunk)"
        )

        # Map phase: process chunks with limited concurrency
        semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

        async def throttled_map(idx: int, chunk: str) -> str:
            async with semaphore:
                return await self._map_chunk_with_retry(
                    chunk, idx, len(chunks), question,
                    router, forward_kwargs, max_extraction_tokens,
                )

        tasks = [throttled_map(i, c) for i, c in enumerate(chunks)]
        extractions = await asyncio.gather(*tasks)

        # Reduce phase: combine extractions
        combined = "\n\n".join(
            f"[Section {i + 1}/{len(extractions)}]\n{ext}"
            for i, ext in enumerate(extractions)
            if ext.strip()
        )

        combined_tokens = count_tokens(combined)
        question_tokens = count_tokens(question) if question else 0
        logger.info(
            f"Map complete: {len(extractions)} extractions, "
            f"{combined_tokens} tokens total"
        )

        # If combined extractions still too large, run a reduce pass
        if combined_tokens > content_budget - question_tokens - 200:
            logger.info(
                f"Extractions too large ({combined_tokens} tokens > "
                f"{content_budget - question_tokens - 200} budget), "
                "running reduce pass"
            )
            reduce_prompt = (
                "Combine these extracted sections into a single coherent summary. "
                "Preserve ALL key facts, data, structure, lists, chapter titles, "
                "and details. Do not omit items.\n\n" + combined
            )
            reduce_messages = [
                {
                    "role": "system",
                    "content": (
                        "You combine information extractions into coherent "
                        "summaries. Preserve all details and structure."
                    ),
                },
                {"role": "user", "content": reduce_prompt},
            ]
            try:
                response, _ = await router.chat_completion(
                    messages=reduce_messages,
                    max_tokens=min(content_budget - question_tokens - 100, 4096),
                    **forward_kwargs,
                )
                combined = (
                    response.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", combined)
                )
                logger.info(
                    f"Reduce complete: {combined_tokens} → "
                    f"{count_tokens(combined)} tokens"
                )
            except Exception as e:
                logger.warning(f"Reduce pass failed: {e}")

        # Build the final user message
        if question:
            final_content = (
                f"The following is extracted information from a large document "
                f"({len(chunks)} sections, {bulk_tokens} tokens processed):\n\n"
                f"{combined}\n\n"
                f"---\n\n{question}"
            )
        else:
            final_content = (
                f"The following is extracted information from a large document "
                f"({len(chunks)} sections, {bulk_tokens} tokens processed):\n\n"
                f"{combined}"
            )

        # Reconstruct messages
        result = list(system_msgs)
        for i, msg in enumerate(other_msgs):
            if i == large_msg_idx:
                result.append({"role": "user", "content": final_content})
            else:
                result.append(msg)

        return result
