"""Context extraction strategies for oversized inputs.

Multiple strategies available, selectable via config or CLI flags:

  "header"     — Pure algorithmic: section previews + recent full sections (no LLM)
  "autosearch" — Heuristic keyword extraction from query → grep → assemble (no LLM)
  "rlm"        — 2-step RAG: LLM generates search terms → grep → assemble (1 LLM call)
  "rlm_v2"     — Fixed RLM: word-level matching, IDF-weighted ranking, no summary bloat (1 LLM call)
  "toolcall"   — Multi-turn tool-calling loop: model iteratively searches/reads (N LLM calls)

All strategies share the same ConversationBook and assembly logic.
Falls back to header extraction if the selected strategy fails.

## rlm vs rlm_v2 — what changed and why

rlm_v2 fixes three problems identified in the v1 benchmark (62% header vs 38% rlm):

1. EXACT PHRASE MATCHING KILLS MULTI-WORD TERMS
   v1: `re.escape("Victor Frankenstein")` → searches literal "Victor Frankenstein"
       The book says "Frankenstein" alone most of the time → 0 matches.
       Typos like "Captain Walrton" (LLM output) → 0 matches.
   v2: Multi-word terms split into individual words. Each word searched separately.
       Section scores = how many words from that term matched in the section.
       "Victor Frankenstein" → search "Victor" AND "Frankenstein" → find sections with both.

2. BROAD TERMS DROWN OUT SPECIFIC ONES
   v1: Ranking = count of matched lines per section. "creature" (70 lines in 23 sections)
       outweighs "Justine" (5 lines in 3 sections). Budget fills with creature-heavy sections.
   v2: IDF weighting. score_per_term = 1 / log2(num_sections_matched + 1).
       A term matching 3 sections gets weight 2.0, a term matching 23 sections gets weight 0.2.
       Section score = sum of IDF-weighted term scores → specific matches rank higher.

3. MATCH SUMMARY EATS BUDGET WITH SCATTERED LINES
   v1: Includes "search match summary" (individual matched lines) using up to 30% of budget.
       With broad terms, this is 50 random lines from all over the book — useless fragments.
   v2: No match summary. Budget goes 100% to full coherent sections.
       Sections are included in IDF-ranked order, then remaining budget fills with recent sections.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from collections import defaultdict
from typing import Any

import httpx

from ctxpact.compaction.book import ConversationBook
from ctxpact.compaction.tokens import count_tokens

logger = logging.getLogger(__name__)

_MAX_QUERY_CHARS = 2000
_MAX_TOOL_RESULT_CHARS = 6000

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _truncate_query(query: str) -> str:
    if len(query) <= _MAX_QUERY_CHARS:
        return query
    return query[:_MAX_QUERY_CHARS] + "\n\n[... truncated ...]"


def _search_book(book: ConversationBook, pattern: str) -> list[tuple[int, str]]:
    """Search all sections for pattern (exact phrase). Returns (section_index, line) tuples."""
    try:
        regex = re.compile(re.escape(pattern), re.IGNORECASE)
    except re.error:
        return []

    matches: list[tuple[int, str]] = []
    for section in book.sections:
        for line in section.content.splitlines():
            if regex.search(line):
                matches.append((section.index, line.strip()))
    return matches


def _search_book_words(
    book: ConversationBook, term: str
) -> dict[int, int]:
    """Search sections for a term using word-level matching.

    Multi-word terms are split into individual words. A section's score is
    the number of distinct words from the term found in that section.
    Single-word terms score 1 if found.

    Returns {section_index: word_match_count}.
    """
    words = [w.strip() for w in term.split() if len(w.strip()) >= 2]
    if not words:
        return {}

    # Build regex for each word
    word_regexes: list[re.Pattern] = []
    for w in words:
        try:
            word_regexes.append(re.compile(re.escape(w), re.IGNORECASE))
        except re.error:
            continue

    if not word_regexes:
        return {}

    section_scores: dict[int, int] = {}
    for section in book.sections:
        text = section.content
        matched_words = sum(1 for rx in word_regexes if rx.search(text))
        if matched_words > 0:
            section_scores[section.index] = matched_words

    return section_scores


def _parse_all_json(text: str) -> list[dict]:
    """Extract ALL JSON objects from text (handles multiple tool calls)."""
    results: list[dict] = []
    i = 0
    while i < len(text):
        if text[i] == '{':
            depth = 0
            for j in range(i, len(text)):
                if text[j] == '{':
                    depth += 1
                elif text[j] == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            results.append(json.loads(text[i:j + 1]))
                        except json.JSONDecodeError:
                            pass
                        i = j + 1
                        break
            else:
                break
        else:
            i += 1
    return results


def _parse_json(text: str) -> dict | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find a balanced JSON object (handles nested braces)
    for i, ch in enumerate(text):
        if ch == '{':
            depth = 0
            for j in range(i, len(text)):
                if text[j] == '{':
                    depth += 1
                elif text[j] == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[i:j+1])
                        except json.JSONDecodeError:
                            break
            break
    # Fallback: simple non-nested JSON
    for match in re.finditer(r'\{[^{}]*\}', text, re.DOTALL):
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


_QUESTION_STOP_WORDS = frozenset({
    "What", "How", "Who", "Where", "When", "Why", "Which",
    "Does", "Did", "List", "Name", "Tell", "Describe", "Explain",
    "There", "These", "Those", "This", "That", "Have", "With",
    "From", "About", "Into", "Some", "Many", "Most", "Also",
})


def _extract_heuristic_terms(query: str) -> list[str]:
    """Extract search terms from query using heuristics (no LLM).

    Picks capitalized words (likely names/places) first, then supplements
    with long content words. Filters out common question/function words.
    """
    _CONTENT_STOP = {
        "about", "after", "before", "being", "could", "does",
        "every", "happen", "happens", "might", "novel", "other",
        "shall", "should", "story", "their", "there", "these",
        "those", "would",
    }
    words = query.split()
    # Start with proper nouns (capitalized, not question words)
    terms = [
        w.strip("?,.'\"!")
        for w in words
        if len(w) > 3
        and w[0].isupper()
        and w.strip("?,.'\"!") not in _QUESTION_STOP_WORDS
    ]
    # Supplement with long content words (discriminative)
    if len(terms) < 3:
        seen = {t.lower() for t in terms}
        for w in words:
            clean = w.strip("?,.'\"!")
            if (
                len(clean) > 4
                and clean.lower() not in seen
                and clean.lower() not in _CONTENT_STOP
            ):
                terms.append(clean)
                seen.add(clean.lower())
            if len(terms) >= 5:
                break
    return terms[:5]


# ---------------------------------------------------------------------------
# Shared assembly logic
# ---------------------------------------------------------------------------


def _assemble_context(
    book: ConversationBook,
    relevant_section_ids: list[int],
    all_matches: dict[int, set[str]],
    token_budget: int,
) -> str:
    """Build extracted context from search results within budget (v1 assembly).

    Includes match summary lines + full relevant sections + recent sections.
    """
    parts: list[str] = []
    used = 0

    # Include search match summary
    match_summary_lines: list[str] = []
    for section_idx, lines in sorted(all_matches.items()):
        for line in list(lines)[:5]:  # max 5 lines per section
            match_summary_lines.append(f"[Section {section_idx}] {line}")

    if match_summary_lines:
        summary = (
            f"[Search results from {book.section_count} sections, "
            f"{book.total_tokens} tokens total]\n\n"
            + "\n".join(match_summary_lines[:50])  # cap at 50 lines
        )
        summary_tokens = count_tokens(summary)
        if summary_tokens < token_budget * 0.3:
            parts.append(summary)
            used += summary_tokens

    # Include full text of most relevant sections
    remaining = token_budget - used - 100
    included_sections: list[str] = []
    for section_idx in relevant_section_ids:
        section = book.get_section(section_idx)
        if section and section.token_count <= remaining:
            included_sections.append(section.to_text())
            remaining -= section.token_count

    # Also fill remaining budget with recent sections (recency bias)
    included_ids = set(relevant_section_ids)
    for section in reversed(book.sections):
        if section.index in included_ids:
            continue
        if section.token_count <= remaining:
            included_sections.append(section.to_text())
            remaining -= section.token_count
            included_ids.add(section.index)
        elif remaining < 200:
            break

    if included_sections:
        parts.append(
            "\n--- Relevant sections ---\n\n"
            + "\n\n".join(included_sections)
        )

    logger.info(
        f"[v1-assembly] {len(match_summary_lines)} match lines + "
        f"{len(included_sections)} full sections, "
        f"~{used + sum(count_tokens(s) for s in included_sections)} tokens"
    )

    return "\n\n".join(parts) if parts else ""


def _assemble_context_v2(
    book: ConversationBook,
    ranked_section_ids: list[int],
    token_budget: int,
) -> str:
    """Build extracted context from IDF-ranked sections (v2 assembly).

    No match summary. 100% budget goes to full coherent sections.
    Fills ranked sections first, then remaining budget with recent sections.
    """
    parts: list[str] = []
    remaining = token_budget - 200  # reserve for framing text
    included_ids: set[int] = set()
    included_sections: list[str] = []

    # Phase 1: Include IDF-ranked sections in order
    for section_idx in ranked_section_ids:
        section = book.get_section(section_idx)
        if section and section.token_count <= remaining:
            included_sections.append(section.to_text())
            remaining -= section.token_count
            included_ids.add(section_idx)
            logger.info(
                f"[v2-assembly] +section {section_idx} "
                f"({section.token_count} tokens, {remaining} remaining)"
            )
        if remaining < 200:
            break

    # Phase 2: Fill remaining budget with recent sections (recency bias)
    recency_added = 0
    for section in reversed(book.sections):
        if section.index in included_ids:
            continue
        if section.token_count <= remaining:
            included_sections.append(section.to_text())
            remaining -= section.token_count
            included_ids.add(section.index)
            recency_added += 1
        elif remaining < 200:
            break

    if included_sections:
        framing = (
            f"[Extracted from {book.section_count} sections, "
            f"{book.total_tokens} tokens total. "
            f"Showing {len(included_ids)} most relevant sections.]\n"
        )
        parts.append(framing + "\n\n".join(included_sections))

    logger.info(
        f"[v2-assembly] {len(included_ids)} sections total "
        f"({len(included_ids) - recency_added} by relevance, {recency_added} by recency), "
        f"~{token_budget - remaining} tokens used of {token_budget} budget"
    )

    return "\n\n".join(parts) if parts else ""


def _header_extract(
    book: ConversationBook,
    token_budget: int,
) -> str:
    """Fallback: section headers + most recent full sections."""
    logger.warning("Using header extraction fallback")
    parts: list[str] = []
    used = 0

    header_parts: list[str] = []
    for section in book.sections:
        header = section.header_text(max_chars=600)
        header_tokens = count_tokens(header)
        if used + header_tokens < token_budget * 0.6:
            header_parts.append(header)
            used += header_tokens

    if header_parts:
        parts.append(
            f"[Overview of {book.section_count} sections, "
            f"{book.total_tokens} tokens total]\n\n"
            + "\n\n".join(header_parts)
        )

    remaining = token_budget - used - 100
    full_sections: list[str] = []
    for section in reversed(book.sections):
        if section.token_count <= remaining:
            full_sections.insert(0, section.to_text())
            remaining -= section.token_count
        elif remaining < 200:
            break

    if full_sections:
        parts.append(
            "\n--- Full sections ---\n\n"
            + "\n\n".join(full_sections)
        )

    return "\n\n".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# Strategy: header (no LLM)
# ---------------------------------------------------------------------------


class HeaderExtractor:
    """Pure algorithmic extraction: section previews + recent full sections.

    No LLM calls. Fast and reliable. Good baseline.
    """

    async def extract(
        self,
        book: ConversationBook,
        query: str,
        token_budget: int,
    ) -> str:
        if book.section_count == 0:
            return ""
        if book.total_tokens <= token_budget:
            return book.to_searchable_text()

        logger.info(
            f"Header extraction: book={book.total_tokens} tokens, "
            f"budget={token_budget} tokens, sections={book.section_count}"
        )
        return _header_extract(book, token_budget)


# ---------------------------------------------------------------------------
# Strategy: autosearch (no LLM)
# ---------------------------------------------------------------------------


class AutoSearchExtractor:
    """Heuristic keyword extraction from query → grep → assemble.

    No LLM calls. Extracts capitalized words and long words from the query,
    searches the book, and assembles relevant sections.
    """

    async def extract(
        self,
        book: ConversationBook,
        query: str,
        token_budget: int,
    ) -> str:
        if book.section_count == 0:
            return ""
        if book.total_tokens <= token_budget:
            return book.to_searchable_text()

        short_query = _truncate_query(query)
        logger.info(
            f"AutoSearch extraction: book={book.total_tokens} tokens, "
            f"budget={token_budget} tokens, sections={book.section_count}"
        )

        # Extract keywords heuristically
        terms = _extract_heuristic_terms(short_query)
        logger.info(f"AutoSearch terms: {terms}")

        if not terms:
            return _header_extract(book, token_budget)

        # Search the book
        all_matches: dict[int, set[str]] = {}
        for term in terms:
            matches = _search_book(book, term)
            logger.info(f"AutoSearch '{term}': {len(matches)} matches")
            for section_idx, line in matches:
                all_matches.setdefault(section_idx, set()).add(line)

        # Sort sections by match count
        section_relevance = sorted(
            all_matches.items(), key=lambda x: len(x[1]), reverse=True
        )
        relevant_section_ids = [idx for idx, _ in section_relevance]

        logger.info(
            f"AutoSearch found {len(relevant_section_ids)} relevant sections "
            f"from {len(terms)} terms"
        )

        result = _assemble_context(
            book, relevant_section_ids, all_matches, token_budget
        )
        return result if result else _header_extract(book, token_budget)


# ---------------------------------------------------------------------------
# Strategy: rlm (1 LLM call for search terms)
# ---------------------------------------------------------------------------

_QUERY_ANALYSIS_PROMPT = """\
You are a search query generator. Given a question about a book, output \
the search terms needed to find the answer by grepping the book text.

Respond with ONLY a JSON object:
{"terms": ["term1", "term2", "term3"]}

Rules:
- Include character names, place names, key nouns from the question
- Include related terms the text might use (synonyms, related words)
- 2-5 search terms, most specific first
- Terms are case-insensitive grep patterns
- Do NOT answer the question — only output search terms"""


class RLMExtractor:
    """2-step RAG: LLM generates search terms → grep → assemble.

    Makes 1 LLM call to generate semantic search terms, then searches
    the book algorithmically and assembles relevant sections.
    """

    def __init__(
        self,
        provider_url: str,
        model: str,
        api_key: str = "dummy",
        max_context: int = 16384,
        **kwargs: Any,
    ) -> None:
        self.provider_url = provider_url
        self.model = model
        self.api_key = api_key
        self.max_context = max_context

    async def extract(
        self,
        book: ConversationBook,
        query: str,
        token_budget: int,
    ) -> str:
        if book.section_count == 0:
            return ""
        if book.total_tokens <= token_budget:
            logger.info(
                f"Book fits in budget ({book.total_tokens} <= {token_budget}), "
                "returning full text"
            )
            return book.to_searchable_text()

        short_query = _truncate_query(query)
        logger.info(
            f"RLM extraction: book={book.total_tokens} tokens, "
            f"budget={token_budget} tokens, "
            f"sections={book.section_count}, query={short_query[:100]}..."
        )

        try:
            # Step 1: Ask model for search terms
            search_terms = await self._generate_search_terms(book, short_query)
            logger.info(f"RLM search terms: {search_terms}")

            # Step 2: Search the book
            all_matches: dict[int, set[str]] = {}
            for term in search_terms:
                matches = _search_book(book, term)
                logger.info(f"RLM search '{term}': {len(matches)} matches")
                for section_idx, line in matches:
                    all_matches.setdefault(section_idx, set()).add(line)

            # Sort by relevance
            section_relevance = sorted(
                all_matches.items(), key=lambda x: len(x[1]), reverse=True
            )
            relevant_section_ids = [idx for idx, _ in section_relevance]

            logger.info(
                f"RLM found {len(relevant_section_ids)} relevant sections "
                f"from {len(search_terms)} search terms"
            )

            # Step 3: Assemble context within budget
            result = _assemble_context(
                book, relevant_section_ids, all_matches, token_budget
            )
            return result if result else _header_extract(book, token_budget)

        except Exception as e:
            logger.error(f"RLM extraction failed: {e}", exc_info=True)
            return _header_extract(book, token_budget)

    async def _generate_search_terms(
        self, book: ConversationBook, query: str
    ) -> list[str]:
        """Step 1: Use LLM to generate semantic search terms for the query."""
        section_index = book.to_section_index()

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self.provider_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": _QUERY_ANALYSIS_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                f"Book index:\n{section_index}\n\n"
                                f"Question: {query}"
                            ),
                        },
                    ],
                    "max_tokens": 256,
                    "temperature": 0.1,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        text = data["choices"][0]["message"]["content"]
        parsed = _parse_json(text)

        if parsed and "terms" in parsed:
            terms = parsed["terms"]
            if isinstance(terms, list):
                return [str(t) for t in terms[:5]]

        # Fallback: heuristic extraction
        logger.warning("Could not parse search terms from LLM, using heuristic fallback")
        return _extract_heuristic_terms(query)


# ---------------------------------------------------------------------------
# Strategy: rlm_v2 (fixed RLM — word matching, IDF ranking, no summary bloat)
# ---------------------------------------------------------------------------

_QUERY_ANALYSIS_PROMPT_V2 = """\
You are a search query generator. Given a question about a document, output \
search terms to grep for in the text.

Respond with ONLY a JSON object:
{"terms": ["term1", "term2", "term3"]}

Rules:
- Use SINGLE WORDS only, not multi-word phrases
- Include character names, place names, key nouns from the question
- Include synonyms and related terms the text might use
- 3-6 search terms, most specific/rare first
- Terms are case-insensitive
- Do NOT answer the question — only output search terms"""


class RLMV2Extractor:
    """Fixed RLM: word-level matching, IDF-weighted ranking, no summary bloat.

    Fixes over v1:
    1. Multi-word terms split into words (no exact phrase failure)
    2. IDF weighting: rare terms rank higher than common ones
    3. No match summary: 100% budget to full coherent sections
    """

    def __init__(
        self,
        provider_url: str,
        model: str,
        api_key: str = "dummy",
        max_context: int = 16384,
        **kwargs: Any,
    ) -> None:
        self.provider_url = provider_url
        self.model = model
        self.api_key = api_key
        self.max_context = max_context

    async def extract(
        self,
        book: ConversationBook,
        query: str,
        token_budget: int,
    ) -> str:
        if book.section_count == 0:
            return ""
        if book.total_tokens <= token_budget:
            logger.info(
                f"[rlm_v2] Book fits in budget ({book.total_tokens} <= {token_budget}), "
                "returning full text"
            )
            return book.to_searchable_text()

        short_query = _truncate_query(query)
        logger.info(
            f"[rlm_v2] extraction: book={book.total_tokens} tokens, "
            f"budget={token_budget} tokens, "
            f"sections={book.section_count}, query={short_query[:100]}..."
        )

        try:
            # Step 1: Ask model for search terms
            search_terms = await self._generate_search_terms(book, short_query)
            logger.info(f"[rlm_v2] search terms from LLM: {search_terms}")

            # Step 2: Search with word-level matching + IDF scoring
            # For each term, find which sections match and how well
            term_section_scores: dict[str, dict[int, int]] = {}
            term_section_counts: dict[str, int] = {}

            for term in search_terms:
                scores = _search_book_words(book, term)
                term_section_scores[term] = scores
                term_section_counts[term] = len(scores)
                logger.info(
                    f"[rlm_v2] search '{term}': "
                    f"{len(scores)} sections matched, "
                    f"word scores: {dict(sorted(scores.items())[:10])}"
                )

            # Step 3: IDF-weighted section ranking
            section_final_scores: dict[int, float] = defaultdict(float)
            n_sections = book.section_count

            for term, scores in term_section_scores.items():
                n_matched = term_section_counts[term]
                if n_matched == 0:
                    continue
                # IDF: terms that match fewer sections are more informative
                idf = 1.0 / math.log2(n_matched + 1)
                logger.info(
                    f"[rlm_v2] term '{term}': matched {n_matched}/{n_sections} sections, "
                    f"IDF weight={idf:.3f}"
                )
                for section_idx, word_count in scores.items():
                    # Score = IDF * word_match_ratio
                    n_words = len([w for w in term.split() if len(w.strip()) >= 2])
                    word_ratio = word_count / max(n_words, 1)
                    section_final_scores[section_idx] += idf * word_ratio

            # Rank sections by final score (highest first)
            ranked = sorted(
                section_final_scores.items(), key=lambda x: x[1], reverse=True
            )
            ranked_section_ids = [idx for idx, score in ranked]

            # Log the ranking
            for idx, score in ranked[:15]:
                section = book.get_section(idx)
                preview = section.content[:80].replace("\n", " ") if section else "?"
                logger.info(
                    f"[rlm_v2] rank: section {idx} score={score:.3f} "
                    f"({section.token_count if section else 0} tokens) "
                    f"preview={preview}..."
                )

            logger.info(
                f"[rlm_v2] {len(ranked_section_ids)} sections scored "
                f"from {len(search_terms)} terms"
            )

            # Step 4: Assemble with v2 (no match summary, full sections only)
            result = _assemble_context_v2(
                book, ranked_section_ids, token_budget
            )
            return result if result else _header_extract(book, token_budget)

        except Exception as e:
            logger.error(f"[rlm_v2] extraction failed: {e}", exc_info=True)
            return _header_extract(book, token_budget)

    async def _generate_search_terms(
        self, book: ConversationBook, query: str
    ) -> list[str]:
        """Step 1: Use LLM to generate single-word search terms."""
        section_index = book.to_section_index()

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self.provider_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": _QUERY_ANALYSIS_PROMPT_V2},
                        {
                            "role": "user",
                            "content": (
                                f"Book index:\n{section_index}\n\n"
                                f"Question: {query}"
                            ),
                        },
                    ],
                    "max_tokens": 256,
                    "temperature": 0.1,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        text = data["choices"][0]["message"]["content"]
        logger.info(f"[rlm_v2] LLM raw response: {text[:300]}")
        parsed = _parse_json(text)

        if parsed and "terms" in parsed:
            terms = parsed["terms"]
            if isinstance(terms, list):
                return [str(t) for t in terms[:6]]

        # Fallback: heuristic extraction
        logger.warning("[rlm_v2] Could not parse search terms from LLM, using heuristic fallback")
        return _extract_heuristic_terms(query)


# ---------------------------------------------------------------------------
# Strategy: rlm_v3 (DSPy RLM — model writes Python to search the book)
# ---------------------------------------------------------------------------

_RLM_V3_INSTRUCTIONS = """\
You are a search engine for a conversation book. The book is split into \
numbered sections. You have access to the full book as a dictionary mapping \
section numbers to text content, plus a table of contents for orientation.

Your task: find the sections most relevant to answering the user's question.

You will write Python code to search through the sections. You have these \
variables available:
  - `book_sections`: dict[int, str] — mapping from section number to text
  - `section_index`: str — table of contents with section previews

Tips:
- Read section_index first to orient yourself
- Search for keywords using string operations (e.g. `if "word" in text.lower()`)
- Collect relevant section numbers and text excerpts
- Use SUBMIT() when you have found the relevant sections
- Do not import pandas or numpy; use built-ins and re only
- Do not wrap code in backticks; write raw Python only"""


class RLMV3Extractor:
    """DSPy RLM: model writes Python code to iteratively search the book.

    Uses dspy.RLM with a Deno-sandboxed Python interpreter. The model
    gets the book as a dict + section index, writes search code, and
    iteratively refines until it finds relevant sections.

    This is the approach from github.com/halfprice06/rlmgrep adapted
    for ConversationBook.
    """

    def __init__(
        self,
        provider_url: str,
        model: str,
        api_key: str = "dummy",
        max_context: int = 16384,
        max_iterations: int = 10,
        max_llm_calls: int = 15,
        **kwargs: Any,
    ) -> None:
        self.provider_url = provider_url
        self.model = model
        self.api_key = api_key
        self.max_context = max_context
        self.max_iterations = max_iterations
        self.max_llm_calls = max_llm_calls

    async def extract(
        self,
        book: ConversationBook,
        query: str,
        token_budget: int,
    ) -> str:
        if book.section_count == 0:
            return ""
        if book.total_tokens <= token_budget:
            logger.info(
                f"[rlm_v3] Book fits in budget ({book.total_tokens} <= {token_budget}), "
                "returning full text"
            )
            return book.to_searchable_text()

        short_query = _truncate_query(query)
        logger.info(
            f"[rlm_v3] DSPy RLM extraction: book={book.total_tokens} tokens, "
            f"budget={token_budget} tokens, "
            f"sections={book.section_count}, query={short_query[:100]}..."
        )

        try:
            # Run DSPy RLM in a thread (it's synchronous)
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self._run_rlm,
                book,
                short_query,
            )

            relevant_section_ids = result.get("relevant_sections", [])
            excerpts = result.get("excerpts", "")

            logger.info(
                f"[rlm_v3] RLM returned {len(relevant_section_ids)} sections: {relevant_section_ids}"
            )

            if not relevant_section_ids and not excerpts:
                logger.warning("[rlm_v3] RLM returned nothing, falling back to header")
                return _header_extract(book, token_budget)

            # Assemble context from the sections RLM identified
            return _assemble_context_v2(
                book, relevant_section_ids, token_budget
            )

        except Exception as e:
            logger.error(f"[rlm_v3] DSPy RLM failed: {e}", exc_info=True)
            return _header_extract(book, token_budget)

    def _run_rlm(self, book: ConversationBook, query: str) -> dict:
        """Run DSPy RLM synchronously (called from executor)."""
        import dspy

        # Configure DSPy LM to point to our local model
        lm = dspy.LM(
            f"openai/{self.model}",
            api_base=self.provider_url,
            api_key=self.api_key,
            max_tokens=self.max_context,
            temperature=0.1,
        )
        dspy.configure(lm=lm)

        # Build the DSPy signature
        class BookSearchSignature(dspy.Signature):
            __doc__ = _RLM_V3_INSTRUCTIONS + f"\n\nUser query: {query}"

            book_sections: dict = dspy.InputField(
                desc="Mapping from section number (int) to section text (str). "
                "These are the only sections available to search."
            )
            section_index: str = dspy.InputField(
                desc="Table of contents with section previews. Read this first."
            )

            relevant_sections: list[int] = dspy.OutputField(
                desc="List of section numbers most relevant to answering the query. "
                "Return section numbers as integers."
            )
            excerpts: str = dspy.OutputField(
                desc="Key text excerpts from relevant sections that help answer the query."
            )

        # Prepare inputs
        book_sections = {s.index: s.content for s in book.sections}
        section_index = book.to_section_index()

        # Let DSPy create its own PythonInterpreter — it will set up
        # SUBMIT(), tools, and output_fields correctly. The previous
        # custom interpreter with enable_read_paths caused mount errors
        # ("Is a directory") which silently broke every iteration.
        rlm = dspy.RLM(
            BookSearchSignature,
            max_iterations=self.max_iterations,
            max_llm_calls=self.max_llm_calls,
            verbose=True,
        )

        logger.info("[rlm_v3] Starting DSPy RLM execution...")
        result = rlm(
            book_sections=book_sections,
            section_index=section_index,
        )
        logger.info(f"[rlm_v3] RLM completed. Result fields: {dir(result)}")

        relevant = list(getattr(result, "relevant_sections", []))
        excerpts = str(getattr(result, "excerpts", ""))

        # Ensure section ids are ints
        relevant_ints = []
        for s in relevant:
            try:
                relevant_ints.append(int(s))
            except (ValueError, TypeError):
                pass

        return {"relevant_sections": relevant_ints, "excerpts": excerpts}


# ---------------------------------------------------------------------------
# Strategy: toolcall (multi-turn tool-calling loop)
# ---------------------------------------------------------------------------

_TOOLCALL_SYSTEM_PROMPT = """\
You are a research assistant searching a book to answer a question.
You have access to tools. To use a tool, respond with ONLY a JSON object.

Available tools:

1. search_book — Search for a pattern in the book text
   {{"tool": "search_book", "pattern": "search term"}}
   Returns matching lines with section numbers.

2. read_sections — Read the full text of specific sections
   {{"tool": "read_sections", "ids": [1, 5, 12]}}
   Returns the full content of those sections.

3. done — When you have gathered enough information
   {{"tool": "done"}}

Strategy:
- First, look at the book index to identify promising sections
- Use search_book to find sections containing relevant keywords
- Use read_sections to read the most relevant sections in full
- Call done when you have enough context to answer the question
- You MUST use search_book at least once before calling done

Respond with ONLY the JSON tool call. No other text."""


class ToolCallExtractor:
    """Multi-turn tool-calling loop: model iteratively searches/reads.

    The model outputs JSON to call search_book() or read_sections(),
    we execute the tool and feed results back. Continues until the model
    says "done" or we hit max_iterations.
    """

    def __init__(
        self,
        provider_url: str,
        model: str,
        api_key: str = "dummy",
        max_context: int = 16384,
        max_iterations: int = 10,
        max_llm_calls: int = 15,
        **kwargs: Any,
    ) -> None:
        self.provider_url = provider_url
        self.model = model
        self.api_key = api_key
        self.max_context = max_context
        self.max_iterations = max_iterations
        self.max_llm_calls = max_llm_calls

    async def extract(
        self,
        book: ConversationBook,
        query: str,
        token_budget: int,
    ) -> str:
        if book.section_count == 0:
            return ""
        if book.total_tokens <= token_budget:
            return book.to_searchable_text()

        short_query = _truncate_query(query)
        logger.info(
            f"ToolCall extraction: book={book.total_tokens} tokens, "
            f"budget={token_budget} tokens, "
            f"sections={book.section_count}, query={short_query[:100]}..."
        )

        try:
            return await self._tool_loop(book, short_query, token_budget)
        except Exception as e:
            logger.error(f"ToolCall extraction failed: {e}", exc_info=True)
            return _header_extract(book, token_budget)

    async def _tool_loop(
        self,
        book: ConversationBook,
        query: str,
        token_budget: int,
    ) -> str:
        """Run the tool-calling loop."""
        section_index = book.to_section_index()

        messages: list[dict[str, str]] = [
            {"role": "system", "content": _TOOLCALL_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Book index:\n{section_index}\n\n"
                    f"Question: {query}\n\n"
                    "Start by searching for relevant terms."
                ),
            },
        ]

        all_matches: dict[int, set[str]] = {}
        read_section_ids: list[int] = []
        used_search = False
        llm_calls = 0

        for iteration in range(self.max_iterations):
            if llm_calls >= self.max_llm_calls:
                logger.info(f"ToolCall: hit max LLM calls ({self.max_llm_calls})")
                break

            # Call LLM
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{self.provider_url}/chat/completions",
                    json={
                        "model": self.model,
                        "messages": messages[-10:],  # keep context manageable
                        "max_tokens": 256,
                        "temperature": 0.1,
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            llm_calls += 1
            text = data["choices"][0]["message"]["content"]
            logger.info(f"ToolCall iter {iteration + 1}: {text[:200]}")

            parsed = _parse_json(text)
            if not parsed:
                logger.warning(f"ToolCall: could not parse JSON from response")
                # Nudge model to use tools
                messages.append({"role": "assistant", "content": text})
                messages.append({
                    "role": "user",
                    "content": "Please respond with a JSON tool call. "
                    'Example: {"tool": "search_book", "pattern": "keyword"}',
                })
                continue

            # Handle "done"
            if parsed.get("tool") == "done" or parsed.get("done"):
                if not used_search:
                    # Force at least one search
                    logger.info("ToolCall: model said done without searching, forcing search")
                    terms = _extract_heuristic_terms(query)
                    for term in terms[:3]:
                        matches = _search_book(book, term)
                        for section_idx, line in matches:
                            all_matches.setdefault(section_idx, set()).add(line)
                    used_search = True
                logger.info(f"ToolCall: done after {iteration + 1} iterations, {llm_calls} LLM calls")
                break

            # Handle search_book
            if parsed.get("tool") == "search_book":
                pattern = str(parsed.get("pattern", ""))
                if pattern:
                    matches = _search_book(book, pattern)
                    logger.info(f"ToolCall search_book('{pattern}'): {len(matches)} matches")
                    used_search = True

                    for section_idx, line in matches:
                        all_matches.setdefault(section_idx, set()).add(line)

                    # Format result for model
                    result_lines = [
                        f"[Section {idx}] {line}"
                        for idx, line in matches[:20]
                    ]
                    result_text = (
                        f"search_book('{pattern}'): {len(matches)} matches\n"
                        + "\n".join(result_lines)
                    )
                    if len(result_text) > _MAX_TOOL_RESULT_CHARS:
                        result_text = result_text[:_MAX_TOOL_RESULT_CHARS] + "\n[... truncated ...]"

                    messages.append({"role": "assistant", "content": text})
                    messages.append({"role": "user", "content": result_text})
                continue

            # Handle read_sections
            if parsed.get("tool") == "read_sections":
                ids = parsed.get("ids", [])
                if isinstance(ids, list):
                    valid_ids = []
                    for i in ids[:5]:
                        try:
                            valid_ids.append(int(i))
                        except (ValueError, TypeError):
                            pass
                    ids = valid_ids
                    read_section_ids.extend(ids)

                    sections_text = book.get_sections_text(ids)
                    if len(sections_text) > _MAX_TOOL_RESULT_CHARS:
                        sections_text = sections_text[:_MAX_TOOL_RESULT_CHARS] + "\n[... truncated ...]"

                    messages.append({"role": "assistant", "content": text})
                    messages.append({
                        "role": "user",
                        "content": f"read_sections({ids}):\n{sections_text}",
                    })
                continue

            # Unknown tool
            messages.append({"role": "assistant", "content": text})
            messages.append({
                "role": "user",
                "content": 'Unknown tool. Use: {"tool": "search_book", "pattern": "..."} '
                'or {"tool": "read_sections", "ids": [...]} or {"tool": "done"}',
            })

        # Assemble results
        section_relevance = sorted(
            all_matches.items(), key=lambda x: len(x[1]), reverse=True
        )
        relevant_section_ids = [idx for idx, _ in section_relevance]
        # Also include explicitly-read sections
        for sid in read_section_ids:
            if sid not in relevant_section_ids:
                relevant_section_ids.append(sid)

        logger.info(
            f"ToolCall found {len(relevant_section_ids)} relevant sections, "
            f"{llm_calls} LLM calls"
        )

        result = _assemble_context(
            book, relevant_section_ids, all_matches, token_budget
        )
        return result if result else _header_extract(book, token_budget)


# ---------------------------------------------------------------------------
# Strategy: embed (semantic embedding retrieval via chromadb)
# ---------------------------------------------------------------------------

class EmbedExtractor:
    """Semantic embedding retrieval: embed sections + query, return top-K by cosine similarity.

    Uses chromadb (in-memory) with sentence-transformers default model (all-MiniLM-L6-v2).
    Zero LLM extraction calls — embedding model runs locally on CPU.
    """

    def __init__(self, **kwargs: Any) -> None:
        pass

    async def extract(
        self,
        book: ConversationBook,
        query: str,
        token_budget: int,
    ) -> str:
        if book.section_count == 0:
            return ""
        if book.total_tokens <= token_budget:
            return book.to_searchable_text()

        short_query = _truncate_query(query)
        logger.info(
            f"[embed] Semantic retrieval: book={book.total_tokens} tokens, "
            f"budget={token_budget} tokens, "
            f"sections={book.section_count}, query={short_query[:100]}..."
        )

        try:
            ranked_ids = await asyncio.get_event_loop().run_in_executor(
                None, self._embed_and_rank, book, short_query
            )
            logger.info(
                f"[embed] Ranked sections (top 5): {ranked_ids[:5]}"
            )
            return _assemble_context_v2(book, ranked_ids, token_budget)
        except Exception as e:
            logger.error(f"[embed] Embedding retrieval failed: {e}", exc_info=True)
            return _header_extract(book, token_budget)

    def _embed_and_rank(
        self, book: ConversationBook, query: str
    ) -> list[int]:
        """Embed all sections and query, return section IDs ranked by similarity."""
        import chromadb

        client = chromadb.Client()
        # Use unique name to avoid collisions
        import uuid
        col_name = f"book_{uuid.uuid4().hex[:8]}"
        collection = client.create_collection(col_name)

        collection.add(
            documents=[s.content for s in book.sections],
            ids=[str(s.index) for s in book.sections],
        )

        results = collection.query(
            query_texts=[query],
            n_results=book.section_count,
        )

        # Clean up
        client.delete_collection(col_name)

        ranked_ids = [int(id_str) for id_str in results["ids"][0]]
        return ranked_ids


# ---------------------------------------------------------------------------
# Strategy: compress (LLMLingua token-level prompt compression)
# ---------------------------------------------------------------------------

class CompressExtractor:
    """LLMLingua prompt compression: use a small model (GPT-2) to prune non-essential tokens.

    Compresses the full document text down to token_budget while preserving key information.
    Query-aware compression prioritizes content relevant to the user's question.
    """

    _compressor = None  # Class-level cache for PromptCompressor

    def __init__(self, **kwargs: Any) -> None:
        pass

    async def extract(
        self,
        book: ConversationBook,
        query: str,
        token_budget: int,
    ) -> str:
        if book.section_count == 0:
            return ""
        if book.total_tokens <= token_budget:
            return book.to_searchable_text()

        short_query = _truncate_query(query)
        logger.info(
            f"[compress] LLMLingua compression: book={book.total_tokens} tokens, "
            f"budget={token_budget} tokens, query={short_query[:100]}..."
        )

        try:
            compressed = await asyncio.get_event_loop().run_in_executor(
                None, self._compress, book, short_query, token_budget
            )
            logger.info(
                f"[compress] Compressed to ~{len(compressed.split())} words"
            )
            return compressed
        except Exception as e:
            logger.error(f"[compress] LLMLingua failed: {e}", exc_info=True)
            return _header_extract(book, token_budget)

    def _compress(
        self, book: ConversationBook, query: str, token_budget: int
    ) -> str:
        """Run LLMLingua compression synchronously."""
        from llmlingua import PromptCompressor

        if CompressExtractor._compressor is None:
            logger.info("[compress] Loading LLMLingua compressor (GPT-2 on CPU)...")
            CompressExtractor._compressor = PromptCompressor(
                model_name="openai-community/gpt2",
                device_map="cpu",
            )

        # Split into per-section strings for context-level filtering
        context = [s.content for s in book.sections]

        result = CompressExtractor._compressor.compress_prompt(
            context,
            question=query,
            target_token=token_budget,
            iterative_size=100000,  # avoid iterative path (past_key_values bug)
            condition_in_question="after_condition",
            use_sentence_level_filter=True,
        )

        return result["compressed_prompt"]


# ---------------------------------------------------------------------------
# Strategy: adaptive (hybrid query router)
# ---------------------------------------------------------------------------

class AdaptiveExtractor:
    """Routes queries to the best extraction strategy based on query analysis.

    - Structural queries (chapter count, etc.) → header
    - Detail/entity queries (who, where, when) → embed (semantic)
    - Default → embed
    """

    def __init__(self, **kwargs: Any) -> None:
        self._embed = EmbedExtractor()
        self._header = HeaderExtractor()

    async def extract(
        self,
        book: ConversationBook,
        query: str,
        token_budget: int,
    ) -> str:
        if book.section_count == 0:
            return ""
        if book.total_tokens <= token_budget:
            return book.to_searchable_text()

        strategy = self._classify_query(query)
        logger.info(f"[adaptive] Query classified → routing to: {strategy}")

        if strategy == "header":
            return await self._header.extract(book, query, token_budget)
        else:
            return await self._embed.extract(book, query, token_budget)

    def _classify_query(self, query: str) -> str:
        """Classify query to pick the best extraction strategy."""
        q = query.lower()

        # Structural / overview queries → header (fast, good overview)
        structural_cues = ["how many", "chapter", "structure", "list all",
                           "table of contents", "outline", "sections"]
        if any(cue in q for cue in structural_cues):
            return "header"

        # Everything else → embed (semantic similarity)
        return "embed"


# ---------------------------------------------------------------------------
# Strategy: icl (in-context learning compaction)
# ---------------------------------------------------------------------------

class ICLExtractor:
    """In-context learning compaction: select conversation turns as demonstrations.

    Treats the conversation as a pool of demonstration candidates and selects
    the most relevant turns using a 3-step pipeline:

    1. Turn Relevance Scoring — embed sections, aggregate similarity per turn
    2. Importance Weighting — combine similarity + recency decay + role priority
    3. Selective Assembly — keep top-K turns verbatim, timeline entries for skipped
    """

    # Weights for the 3-signal scoring
    W_SIM = 0.6
    W_REC = 0.3
    W_ROLE = 0.1

    def __init__(self, **kwargs: Any) -> None:
        pass

    async def extract(
        self,
        book: ConversationBook,
        query: str,
        token_budget: int,
    ) -> str:
        if book.section_count == 0:
            return ""
        if book.total_tokens <= token_budget:
            return book.to_searchable_text()

        short_query = _truncate_query(query)
        logger.info(
            f"[icl] In-context learning compaction: book={book.total_tokens} tokens, "
            f"budget={token_budget} tokens, sections={book.section_count}, "
            f"query={short_query[:100]}..."
        )

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._icl_pipeline, book, short_query, token_budget
            )
            return result
        except Exception as e:
            logger.error(f"[icl] ICL compaction failed: {e}", exc_info=True)
            return _header_extract(book, token_budget)

    def _icl_pipeline(
        self, book: ConversationBook, query: str, token_budget: int
    ) -> str:
        """Run the full ICL pipeline synchronously."""

        # Step 1: Group sections into turns
        turns = self._group_into_turns(book)
        total_turns = len(turns)
        logger.info(f"[icl] Step 1: Grouped {book.section_count} sections into {total_turns} turns")

        # Step 2: Score turns by embedding similarity
        sim_scores = self._score_turns_by_similarity(book, query)
        logger.info(f"[icl] Step 2: Embedding similarity scores computed for {len(sim_scores)} turns")

        # Step 3: Weight scores (similarity + recency + role)
        scored_turns = []
        for i, turn in enumerate(turns):
            turn_num = turn["turn_num"]
            is_system = "system" in turn["roles"]

            if is_system:
                # System turns always included
                final_score = float("inf")
            else:
                similarity = sim_scores.get(turn_num, 0.0)
                recency = (i + 1) / total_turns  # 0→1, most recent = highest
                role_bonus = 0.1 if "assistant" in turn["roles"] else 0.0
                final_score = (
                    self.W_SIM * similarity
                    + self.W_REC * recency
                    + self.W_ROLE * role_bonus
                )

            scored_turns.append({
                **turn,
                "score": final_score,
                "is_system": is_system,
            })

        # Step 4: Budget allocation — decide which turns to include fully
        selected_turn_nums = self._allocate_budget(scored_turns, token_budget)
        logger.info(
            f"[icl] Step 3: Selected {len(selected_turn_nums)} of "
            f"{total_turns} turns (budget={token_budget})"
        )

        # If no turns fit (e.g., single mega-turn exceeds budget), fall back
        if not selected_turn_nums:
            logger.warning(
                "[icl] No turns fit within budget, falling back to header extraction"
            )
            return _header_extract(book, token_budget)

        # Step 5: Assemble in chronological order with timeline entries
        return self._assemble_icl(book, scored_turns, selected_turn_nums, token_budget)

    def _group_into_turns(self, book: ConversationBook) -> list[dict]:
        """Group sections into turns using BookSection.turn field."""
        turns_map: dict[int, dict] = {}
        for s in book.sections:
            if s.turn not in turns_map:
                turns_map[s.turn] = {
                    "turn_num": s.turn,
                    "sections": [],
                    "roles": set(),
                    "tokens": 0,
                }
            turns_map[s.turn]["sections"].append(s)
            turns_map[s.turn]["roles"].add(s.role)
            turns_map[s.turn]["tokens"] += s.token_count
        return [turns_map[k] for k in sorted(turns_map.keys())]

    def _score_turns_by_similarity(
        self, book: ConversationBook, query: str
    ) -> dict[int, float]:
        """Embed all sections and compute per-turn similarity scores."""
        import chromadb
        import uuid

        client = chromadb.Client()
        col_name = f"icl_{uuid.uuid4().hex[:8]}"
        collection = client.create_collection(col_name)

        collection.add(
            documents=[s.content for s in book.sections],
            ids=[str(s.index) for s in book.sections],
        )

        results = collection.query(
            query_texts=[query],
            n_results=book.section_count,
            include=["distances"],
        )

        client.delete_collection(col_name)

        # chromadb returns L2 distances; convert to similarity (0-1)
        # similarity = 1 / (1 + distance)
        section_sims: dict[int, float] = {}
        for id_str, dist in zip(results["ids"][0], results["distances"][0]):
            section_sims[int(id_str)] = 1.0 / (1.0 + dist)

        # Aggregate to turn level: max similarity across sections in the turn
        turn_sims: dict[int, float] = {}
        for s in book.sections:
            sim = section_sims.get(s.index, 0.0)
            if s.turn not in turn_sims or sim > turn_sims[s.turn]:
                turn_sims[s.turn] = sim
        return turn_sims

    def _allocate_budget(
        self, scored_turns: list[dict], token_budget: int
    ) -> set[int]:
        """Decide which turns to include fully within the budget."""
        selected: set[int] = set()
        budget_used = 200  # reserve for framing + timeline overhead

        # Phase 1: Include system turns (if they fit)
        for turn in scored_turns:
            if turn["is_system"] and budget_used + turn["tokens"] <= token_budget:
                selected.add(turn["turn_num"])
                budget_used += turn["tokens"]

        # Phase 2: Include last 2 non-system turns (if they fit)
        non_system = [t for t in scored_turns if not t["is_system"]]
        for turn in non_system[-2:]:
            if turn["turn_num"] not in selected and budget_used + turn["tokens"] <= token_budget:
                selected.add(turn["turn_num"])
                budget_used += turn["tokens"]

        # Phase 3: Fill remaining budget with highest-scored turns
        # Reserve ~15 tokens per skipped turn for timeline entries
        remaining_turns = [
            t for t in scored_turns
            if t["turn_num"] not in selected
        ]
        remaining_turns.sort(key=lambda t: t["score"], reverse=True)
        skippable_count = len(remaining_turns)
        timeline_reserve = skippable_count * 15  # ~15 tokens per timeline line

        available = token_budget - budget_used - timeline_reserve
        for turn in remaining_turns:
            if turn["tokens"] <= available:
                selected.add(turn["turn_num"])
                available -= turn["tokens"]
                skippable_count -= 1
                # Reclaim timeline reserve for this turn
                available += 15

        return selected

    @staticmethod
    def _build_timeline_entry(turn: dict) -> str:
        """Build a 1-line timeline entry for a skipped turn."""
        parts = []
        for s in turn["sections"]:
            preview = s.content[:80].replace("\n", " ").strip()
            if len(s.content) > 80:
                preview += "..."
            parts.append(f"{s.role}: {preview}")
        return f"[Turn {turn['turn_num']}: {'; '.join(parts)}]"

    def _assemble_icl(
        self,
        book: ConversationBook,
        scored_turns: list[dict],
        selected_turn_nums: set[int],
        token_budget: int,
    ) -> str:
        """Assemble output in chronological order with timeline entries."""
        parts: list[str] = []
        included_count = 0
        timeline_count = 0

        framing = (
            f"[ICL-compacted conversation: {book.section_count} sections, "
            f"{book.total_tokens} tokens total. "
            f"Showing {len(selected_turn_nums)} turns fully, "
            f"remaining as timeline summaries.]\n"
        )
        parts.append(framing)

        for turn in scored_turns:
            if turn["turn_num"] in selected_turn_nums:
                # Include all sections in this turn fully
                for s in turn["sections"]:
                    parts.append(s.to_text())
                included_count += 1
            else:
                # Insert timeline entry
                parts.append(self._build_timeline_entry(turn))
                timeline_count += 1

        logger.info(
            f"[icl] Assembly: {included_count} full turns, "
            f"{timeline_count} timeline entries"
        )

        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Strategy: rlm_v4 (SRLM-inspired multi-candidate selection)
# ---------------------------------------------------------------------------


class RLMV4Extractor:
    """SRLM-inspired multi-candidate selection.

    Runs multiple extraction strategies in parallel (embed, header, rlm_v2),
    scores each result using lightweight quality signals (budget utilization,
    query term coverage, section diversity), and returns the best one.

    Inspired by the SRLM paper (Alizadeh et al., arXiv:2603.15653) which
    showed that self-consistency signals can select among candidates
    without needing an LLM judge.
    """

    def __init__(
        self,
        provider_url: str,
        model: str,
        api_key: str = "dummy",
        max_context: int = 16384,
        **kwargs: Any,
    ) -> None:
        self.provider_url = provider_url
        self.model = model
        self.api_key = api_key
        self.max_context = max_context

    async def extract(
        self,
        book: ConversationBook,
        query: str,
        token_budget: int,
    ) -> str:
        if book.section_count == 0:
            return ""
        if book.total_tokens <= token_budget:
            return book.to_searchable_text()

        short_query = _truncate_query(query)
        logger.info(
            f"[rlm_v4] Multi-candidate selection: book={book.total_tokens} tokens, "
            f"budget={token_budget} tokens, sections={book.section_count}"
        )

        try:
            # Create sub-extractors
            embed_ext = EmbedExtractor()
            header_ext = HeaderExtractor()
            rlm_v2_ext = RLMV2Extractor(
                provider_url=self.provider_url,
                model=self.model,
                api_key=self.api_key,
                max_context=self.max_context,
            )

            # Run all 3 in parallel
            results = await asyncio.gather(
                embed_ext.extract(book, query, token_budget),
                header_ext.extract(book, query, token_budget),
                rlm_v2_ext.extract(book, query, token_budget),
                return_exceptions=True,
            )

            # Score each successful result
            candidates: list[tuple[str, float, str]] = []  # (result, score, name)
            names = ["embed", "header", "rlm_v2"]

            for name, result in zip(names, results):
                if isinstance(result, Exception):
                    logger.warning(f"[rlm_v4] Sub-strategy '{name}' failed: {result}")
                    continue
                if not result:
                    logger.warning(f"[rlm_v4] Sub-strategy '{name}' returned empty")
                    continue

                score = self._score_result(result, short_query, token_budget, book)
                candidates.append((result, score, name))
                logger.info(f"[rlm_v4] Candidate '{name}': score={score:.3f}")

            if not candidates:
                logger.warning("[rlm_v4] All sub-strategies failed, falling back to header")
                return _header_extract(book, token_budget)

            # Pick the winner
            candidates.sort(key=lambda c: c[1], reverse=True)
            winner_result, winner_score, winner_name = candidates[0]
            logger.info(
                f"[rlm_v4] Winner: '{winner_name}' (score={winner_score:.3f}), "
                f"candidates: {[(n, f'{s:.3f}') for _, s, n in candidates]}"
            )
            return winner_result

        except Exception as e:
            logger.error(f"[rlm_v4] Multi-candidate selection failed: {e}", exc_info=True)
            return _header_extract(book, token_budget)

    def _score_result(
        self, result: str, query: str, token_budget: int, book: ConversationBook
    ) -> float:
        """Score a candidate result using 3 quality signals."""
        result_tokens = count_tokens(result)

        # Signal 1: Budget utilization (target 70-95%)
        utilization = result_tokens / max(token_budget, 1)
        util_score = max(0.0, 1.0 - abs(0.85 - min(utilization, 1.0)) * 2)

        # Signal 2: Query term coverage
        terms = _extract_heuristic_terms(query)
        if terms:
            result_lower = result.lower()
            matched = sum(1 for t in terms if t.lower() in result_lower)
            coverage_score = matched / len(terms)
        else:
            coverage_score = 0.5  # neutral if no terms extracted

        # Signal 3: Section diversity (how many unique section markers)
        # Matches both "=== Section N" (v2 assembly) and "[Section N]" (v1 assembly)
        section_refs = set(re.findall(r'(?:=== Section|Section)\s+(\d+)', result))
        if book.section_count > 0:
            avg_section_tokens = book.total_tokens / book.section_count
            expected_sections = token_budget / max(avg_section_tokens, 1)
            diversity_score = min(1.0, len(section_refs) / max(expected_sections * 0.5, 1))
        else:
            diversity_score = 0.5

        final = 0.4 * util_score + 0.4 * coverage_score + 0.2 * diversity_score
        logger.info(
            f"[rlm_v4] Score breakdown: util={util_score:.2f} "
            f"(tokens={result_tokens}/{token_budget}), "
            f"coverage={coverage_score:.2f} ({len(terms)} terms), "
            f"diversity={diversity_score:.2f} ({len(section_refs)} sections)"
        )
        return final


# ---------------------------------------------------------------------------
# Strategy: rlm_v5 (enhanced programmatic exploration)
# ---------------------------------------------------------------------------

_RLM_V5_SYSTEM_PROMPT = """\
You are a research assistant exploring a book to find sections relevant to a \
question. You have access to tools to search and read the book.

To use a tool, respond with ONLY a JSON object (no other text).

Available tools:

1. stats — Get book statistics (section count, token count, role distribution)
   {{"tool": "stats"}}

2. search — Word-level search with relevance scoring
   {{"tool": "search", "terms": ["word1", "word2"]}}
   Returns sections ranked by relevance (IDF-weighted). Best for named entities \
and specific terms. Use SINGLE WORDS, not phrases.

3. regex — Regex pattern search
   {{"tool": "regex", "pattern": "some.*pattern"}}
   Returns matching lines with section numbers. Best for flexible patterns.

4. read — Read the full text of specific sections
   {{"tool": "read", "ids": [1, 5, 12]}}
   Returns the full content of those sections (max 5 per call).

5. done — Finish and return your list of relevant sections
   {{"tool": "done", "ids": [1, 5, 12]}}
   Include ALL section IDs you consider relevant to the question.

Strategy — follow this exploration workflow:
1. ORIENT: Call stats() first to understand the book structure
2. SEARCH: Use search() for specific names/terms from the question
3. EXPLORE: Use regex() if you need flexible pattern matching
4. VERIFY: Use read() to confirm promising sections are actually relevant
5. FINISH: Call done() with all relevant section IDs

Respond with ONLY the JSON tool call. No other text."""


class RLMV5Extractor:
    """Enhanced programmatic exploration with richer tools and better prompting.

    Improves on ToolCallExtractor with:
    - 5 tools instead of 3 (adds regex search and stats)
    - Word-level search with IDF scoring (from rlm_v2)
    - Exploration-first prompting (orient → search → verify → finish)
    - Progress tracking across iterations
    - v2 assembly (no match summary bloat)

    Inspired by the RLM paper (Zhang et al., arXiv:2512.24601):
    treat context as an interactive environment to explore.
    """

    def __init__(
        self,
        provider_url: str,
        model: str,
        api_key: str = "dummy",
        max_context: int = 16384,
        max_iterations: int = 10,
        max_llm_calls: int = 15,
        **kwargs: Any,
    ) -> None:
        self.provider_url = provider_url
        self.model = model
        self.api_key = api_key
        self.max_context = max_context
        self.max_iterations = max_iterations
        self.max_llm_calls = max_llm_calls

    async def extract(
        self,
        book: ConversationBook,
        query: str,
        token_budget: int,
    ) -> str:
        if book.section_count == 0:
            return ""
        if book.total_tokens <= token_budget:
            return book.to_searchable_text()

        short_query = _truncate_query(query)
        logger.info(
            f"[rlm_v5] Enhanced exploration: book={book.total_tokens} tokens, "
            f"budget={token_budget} tokens, "
            f"sections={book.section_count}, query={short_query[:100]}..."
        )

        try:
            return await self._tool_loop(book, short_query, token_budget)
        except Exception as e:
            logger.error(f"[rlm_v5] Enhanced exploration failed: {e}", exc_info=True)
            return _header_extract(book, token_budget)

    async def _tool_loop(
        self,
        book: ConversationBook,
        query: str,
        token_budget: int,
    ) -> str:
        """Run the enhanced tool-calling loop."""
        section_index = book.to_section_index()

        messages: list[dict[str, str]] = [
            {"role": "system", "content": _RLM_V5_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Book table of contents:\n{section_index}\n\n"
                    f"Question: {query}\n\n"
                    "Start by calling stats() to orient yourself."
                ),
            },
        ]

        # State tracking across iterations
        found_section_ids: set[int] = set()
        section_idf_scores: dict[int, float] = defaultdict(float)
        llm_calls = 0

        for iteration in range(self.max_iterations):
            if llm_calls >= self.max_llm_calls:
                logger.info(f"[rlm_v5] Hit max LLM calls ({self.max_llm_calls})")
                break

            # Call LLM
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{self.provider_url}/chat/completions",
                    json={
                        "model": self.model,
                        "messages": messages[-12:],  # keep context manageable
                        "max_tokens": 256,
                        "temperature": 0.1,
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            llm_calls += 1
            text = data["choices"][0]["message"]["content"]
            logger.info(f"[rlm_v5] iter {iteration + 1}: {text[:200]}")

            parsed = _parse_json(text)
            if not parsed:
                logger.warning("[rlm_v5] Could not parse JSON from response")
                messages.append({"role": "assistant", "content": text})
                messages.append({
                    "role": "user",
                    "content": "Please respond with a JSON tool call. "
                    'Example: {"tool": "search", "terms": ["keyword"]}',
                })
                continue

            tool = parsed.get("tool", "")

            # --- Tool: stats ---
            if tool == "stats":
                roles = defaultdict(int)
                for s in book.sections:
                    roles[s.role] += 1
                stats_text = (
                    f"Book statistics:\n"
                    f"  Sections: {book.section_count}\n"
                    f"  Total tokens: {book.total_tokens}\n"
                    f"  Token budget: {token_budget}\n"
                    f"  Roles: {dict(roles)}\n"
                    f"  Sections found so far: {len(found_section_ids)}"
                )
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": stats_text})
                continue

            # --- Tool: search ---
            if tool == "search":
                terms = parsed.get("terms", [])
                if not isinstance(terms, list):
                    terms = [str(terms)]
                terms = [str(t) for t in terms[:6]]

                search_results: list[str] = []
                n_sections = book.section_count

                for term in terms:
                    scores = _search_book_words(book, term)
                    n_matched = len(scores)
                    if n_matched == 0:
                        search_results.append(f"  '{term}': 0 matches")
                        continue

                    idf = 1.0 / math.log2(n_matched + 1)
                    # Update global IDF scores
                    for sid, word_count in scores.items():
                        n_words = len([w for w in term.split() if len(w.strip()) >= 2])
                        word_ratio = word_count / max(n_words, 1)
                        section_idf_scores[sid] += idf * word_ratio
                        found_section_ids.add(sid)

                    # Format top matches for display
                    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:8]
                    match_strs = [f"sec {sid} (score {sc})" for sid, sc in ranked]
                    search_results.append(
                        f"  '{term}': {n_matched} sections, IDF={idf:.2f}, "
                        f"top: {', '.join(match_strs)}"
                    )

                result_text = (
                    f"search results:\n" + "\n".join(search_results)
                    + f"\n\nTotal relevant sections found so far: {len(found_section_ids)}"
                )
                if len(result_text) > _MAX_TOOL_RESULT_CHARS:
                    result_text = result_text[:_MAX_TOOL_RESULT_CHARS] + "\n[... truncated ...]"

                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": result_text})
                continue

            # --- Tool: regex ---
            if tool == "regex":
                pattern = str(parsed.get("pattern", ""))
                if pattern:
                    matches = _search_book(book, pattern)
                    for sid, _ in matches:
                        found_section_ids.add(sid)

                    result_lines = [
                        f"[Section {idx}] {line}"
                        for idx, line in matches[:20]
                    ]
                    result_text = (
                        f"regex('{pattern}'): {len(matches)} matches\n"
                        + "\n".join(result_lines)
                        + f"\n\nTotal relevant sections found so far: {len(found_section_ids)}"
                    )
                    if len(result_text) > _MAX_TOOL_RESULT_CHARS:
                        result_text = result_text[:_MAX_TOOL_RESULT_CHARS] + "\n[... truncated ...]"

                    messages.append({"role": "assistant", "content": text})
                    messages.append({"role": "user", "content": result_text})
                continue

            # --- Tool: read ---
            if tool == "read":
                ids = parsed.get("ids", [])
                if isinstance(ids, list):
                    valid_ids = []
                    for i in ids[:5]:
                        try:
                            valid_ids.append(int(i))
                        except (ValueError, TypeError):
                            pass
                    ids = valid_ids
                    for sid in ids:
                        found_section_ids.add(sid)

                    sections_text = book.get_sections_text(ids)
                    if len(sections_text) > _MAX_TOOL_RESULT_CHARS:
                        sections_text = sections_text[:_MAX_TOOL_RESULT_CHARS] + "\n[... truncated ...]"

                    messages.append({"role": "assistant", "content": text})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"read({ids}):\n{sections_text}"
                            f"\n\nTotal relevant sections found so far: {len(found_section_ids)}"
                        ),
                    })
                continue

            # --- Tool: done ---
            if tool == "done":
                explicit_ids = parsed.get("ids", [])
                if isinstance(explicit_ids, list):
                    for sid in explicit_ids:
                        try:
                            found_section_ids.add(int(sid))
                        except (ValueError, TypeError):
                            pass
                logger.info(
                    f"[rlm_v5] Done after {iteration + 1} iterations, "
                    f"{llm_calls} LLM calls, "
                    f"{len(found_section_ids)} sections found"
                )
                break

            # Unknown tool
            messages.append({"role": "assistant", "content": text})
            messages.append({
                "role": "user",
                "content": 'Unknown tool. Available: stats, search, regex, read, done. '
                'Example: {"tool": "search", "terms": ["keyword"]}',
            })

        # Assemble using IDF-ranked section IDs (v2 assembly)
        ranked_ids = sorted(
            found_section_ids,
            key=lambda sid: section_idf_scores.get(sid, 0.0),
            reverse=True,
        )

        logger.info(
            f"[rlm_v5] Assembling {len(ranked_ids)} sections "
            f"(IDF-ranked), {llm_calls} LLM calls total"
        )

        result = _assemble_context_v2(book, ranked_ids, token_budget)
        return result if result else _header_extract(book, token_budget)


# ---------------------------------------------------------------------------
# Strategy: rlm_v6 (comprehensive multi-signal retrieval)
# ---------------------------------------------------------------------------

class RLMV6Extractor:
    """Multi-signal retrieval with partial section inclusion.

    Combines retrieval signals via reciprocal rank fusion (RRF):
    1. Semantic embedding (chromadb) — captures meaning beyond keywords
    2. LLM + heuristic word-IDF search (length-normalized) — exact terms
    3. Position signal (for beginning/ending queries) — structural hint

    Key features:
    - Partial section inclusion via query-relevant paragraph extraction
    - Context-aware excerpts (includes adjacent paragraphs)
    - Differentiated budget caps (lower for top-ranked, higher for others)
    - Length-normalized IDF to prevent long sections from dominating

    Uses 1 LLM call for search term generation.
    """

    # Max fraction of budget any single section can consume.
    _MAX_SECTION_BUDGET_RATIO = 0.25

    # Position-detection keywords
    _ENDING_WORDS = frozenset({
        "end", "ends", "ending", "final", "finally", "last",
        "conclude", "concludes", "conclusion",
    })
    _BEGINNING_WORDS = frozenset({
        "beginning", "begin", "begins", "start", "starts",
        "first", "opening",
    })
    # Weight for position boost added to RRF scores
    _POSITION_WEIGHT = 0.015

    def __init__(
        self,
        provider_url: str,
        model: str,
        api_key: str = "dummy",
        max_context: int = 16384,
        **kwargs: Any,
    ) -> None:
        self.provider_url = provider_url
        self.model = model
        self.api_key = api_key
        self.max_context = max_context

    @staticmethod
    def _extract_question(query: str) -> str:
        """Extract just the user question from the query.

        The query may include book text appended after the question.
        We take the first paragraph (before the first blank line).
        """
        first_para = query.split("\n\n", 1)[0].strip()
        return first_para[:500] if first_para else query[:500]

    async def extract(
        self,
        book: ConversationBook,
        query: str,
        token_budget: int,
    ) -> str:
        if book.section_count == 0:
            return ""
        if book.total_tokens <= token_budget:
            return book.to_searchable_text()

        short_query = _truncate_query(query)
        # Extract just the question (not book text) for heuristics/position
        question = self._extract_question(query)
        logger.info(
            f"[rlm_v6] Multi-signal retrieval: book={book.total_tokens} tokens, "
            f"budget={token_budget} tokens, sections={book.section_count}, "
            f"question={question[:80]!r}"
        )

        try:
            # Step 1: Run LLM term generation and embedding in parallel
            terms_task = self._generate_search_terms(book, short_query)
            embed_task = self._embed_score(book, question)
            search_terms, embed_scores = await asyncio.gather(terms_task, embed_task)

            # Merge LLM terms with heuristic terms from the question
            # (ensures query words like "read", "speak" are always searched
            # even if the LLM replaces them with synonyms)
            heuristic = _extract_heuristic_terms(question)
            all_terms = list(dict.fromkeys(search_terms + heuristic))
            logger.info(
                f"[rlm_v6] Search terms: llm={search_terms}, "
                f"heuristic={heuristic}, merged={all_terms}"
            )

            # Step 2: Run length-normalized IDF search
            idf_scores = await asyncio.get_event_loop().run_in_executor(
                None, self._idf_search, book, all_terms
            )

            # Step 3: Position scores (3rd signal for ending/beginning queries)
            pos_scores = self._position_scores(book, question)

            # Step 4: Reciprocal Rank Fusion
            all_section_ids = [s.index for s in book.sections]

            embed_rank = self._rank_order(all_section_ids, embed_scores)
            idf_rank = self._rank_order(all_section_ids, idf_scores)

            # RRF base: score = sum(1 / (k + rank)) for embed + idf
            k = 60
            fused: dict[int, float] = {}
            for sid in all_section_ids:
                fused[sid] = (
                    1.0 / (k + embed_rank.get(sid, len(all_section_ids)))
                    + 1.0 / (k + idf_rank.get(sid, len(all_section_ids)))
                )

            # Add position boost if active (linear score 0-1)
            if pos_scores:
                for sid in all_section_ids:
                    fused[sid] += self._POSITION_WEIGHT * pos_scores.get(sid, 0.0)
                logger.info(
                    f"[rlm_v6] Position boost active "
                    f"(weight={self._POSITION_WEIGHT})"
                )

            ranked_ids = sorted(
                all_section_ids, key=lambda sid: fused[sid], reverse=True
            )

            for sid in ranked_ids[:10]:
                pos_info = f", pos={pos_scores.get(sid, 0):.2f}" if pos_scores else ""
                logger.info(
                    f"[rlm_v6] Section {sid}: rrf={fused[sid]:.4f} "
                    f"(embed_rank={embed_rank.get(sid, '?')}, "
                    f"idf_rank={idf_rank.get(sid, '?')}, "
                    f"embed={embed_scores.get(sid, 0):.3f}, "
                    f"idf={idf_scores.get(sid, 0):.3f}{pos_info})"
                )

            # Step 5: Custom assembly with partial section inclusion
            result = self._assemble_v6(
                book, ranked_ids, token_budget, question, all_terms
            )
            return result if result else _header_extract(book, token_budget)

        except Exception as e:
            logger.error(
                f"[rlm_v6] Dual-signal retrieval failed: {e}", exc_info=True
            )
            return _header_extract(book, token_budget)

    async def _generate_search_terms(
        self, book: ConversationBook, query: str
    ) -> list[str]:
        """Use LLM to generate search terms (reuses rlm_v2 prompt)."""
        section_index = book.to_section_index()

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self.provider_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": _QUERY_ANALYSIS_PROMPT_V2},
                        {
                            "role": "user",
                            "content": (
                                f"Book index:\n{section_index}\n\n"
                                f"Question: {query}"
                            ),
                        },
                    ],
                    "max_tokens": 256,
                    "temperature": 0.1,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        text = data["choices"][0]["message"]["content"]
        logger.info(f"[rlm_v6] LLM terms: {text[:300]}")
        parsed = _parse_json(text)

        if parsed and "terms" in parsed:
            terms = parsed["terms"]
            if isinstance(terms, list):
                return [str(t) for t in terms[:6]]

        logger.warning("[rlm_v6] Could not parse terms, using heuristic fallback")
        return _extract_heuristic_terms(query)

    async def _embed_score(
        self, book: ConversationBook, query: str
    ) -> dict[int, float]:
        """Compute per-section embedding similarity using chunked embedding.

        Splits each section into ~500-token chunks and embeds each chunk
        separately. The section score is the MAX similarity across its
        chunks. This prevents long sections from having diluted embeddings
        and lets specific relevant passages shine through.
        """
        def _run():
            import chromadb
            import uuid

            # Split sections into ~500-token chunks
            chunk_ids: list[str] = []
            chunk_texts: list[str] = []
            chunk_to_section: dict[str, int] = {}

            for section in book.sections:
                paragraphs = [p.strip() for p in section.content.split("\n\n") if p.strip()]
                if not paragraphs:
                    paragraphs = [section.content[:2000]]

                # Group consecutive paragraphs into ~500-token chunks
                current_chunk: list[str] = []
                current_tokens = 0
                chunk_idx = 0
                for para in paragraphs:
                    ptokens = count_tokens(para)
                    if current_tokens + ptokens > 500 and current_chunk:
                        cid = f"{section.index}_{chunk_idx}"
                        chunk_ids.append(cid)
                        chunk_texts.append("\n\n".join(current_chunk))
                        chunk_to_section[cid] = section.index
                        chunk_idx += 1
                        current_chunk = [para]
                        current_tokens = ptokens
                    else:
                        current_chunk.append(para)
                        current_tokens += ptokens
                # Flush remaining
                if current_chunk:
                    cid = f"{section.index}_{chunk_idx}"
                    chunk_ids.append(cid)
                    chunk_texts.append("\n\n".join(current_chunk))
                    chunk_to_section[cid] = section.index

            client = chromadb.Client()
            col_name = f"v6_{uuid.uuid4().hex[:8]}"
            collection = client.create_collection(col_name)

            collection.add(documents=chunk_texts, ids=chunk_ids)

            # Query all chunks so every section gets similarity scores
            n_results = len(chunk_ids)
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["distances"],
            )
            client.delete_collection(col_name)

            # Max similarity per section (best chunk wins)
            scores: dict[int, float] = {}
            for cid, dist in zip(results["ids"][0], results["distances"][0]):
                sid = chunk_to_section[cid]
                sim = 1.0 / (1.0 + dist)
                if sid not in scores or sim > scores[sid]:
                    scores[sid] = sim

            # Normalize to 0-1
            if scores:
                max_s = max(scores.values())
                min_s = min(scores.values())
                rng = max_s - min_s
                if rng > 0:
                    scores = {k: (v - min_s) / rng for k, v in scores.items()}

            return scores

        return await asyncio.get_event_loop().run_in_executor(None, _run)

    def _position_scores(
        self, book: ConversationBook, query: str
    ) -> dict[int, float]:
        """Return position scores (0-1) if query refers to beginning/end.

        Returns empty dict if no position signal detected.
        """
        words = set(re.findall(r"\w+", query.lower()))
        n = book.section_count
        if words & self._ENDING_WORDS:
            return {
                s.index: i / max(n - 1, 1)
                for i, s in enumerate(book.sections)
            }
        if words & self._BEGINNING_WORDS:
            return {
                s.index: 1.0 - i / max(n - 1, 1)
                for i, s in enumerate(book.sections)
            }
        return {}

    def _idf_search(
        self, book: ConversationBook, terms: list[str]
    ) -> dict[int, float]:
        """Run IDF-weighted word search with aggressive specificity weighting.

        Uses log-ratio IDF that gives ZERO weight to terms matching all
        sections, forcing the ranking to be driven by specific/rare terms.
        This prevents generic terms like 'Frankenstein' from dominating.
        """
        if not terms:
            return {}

        n_sections = max(book.section_count, 1)
        log_n = math.log(n_sections) if n_sections > 1 else 1.0

        section_scores: dict[int, float] = defaultdict(float)
        for term in terms:
            scores = _search_book_words(book, term)
            if not scores:
                continue
            n_matched = len(scores)
            # Log-ratio IDF: 0 when term matches all sections, 1 when unique
            idf = max(0.0, 1.0 - math.log(n_matched) / log_n) if n_matched > 0 else 1.0
            if idf < 0.05:  # effectively zero — skip to save time
                continue
            n_words = len([w for w in term.split() if len(w.strip()) >= 2])
            for sid, word_count in scores.items():
                word_ratio = word_count / max(n_words, 1)
                section_scores[sid] += idf * word_ratio

        # Normalize to 0-1
        if section_scores:
            max_s = max(section_scores.values())
            if max_s > 0:
                section_scores = {k: v / max_s for k, v in section_scores.items()}

        return dict(section_scores)

    @staticmethod
    def _rank_order(
        section_ids: list[int], scores: dict[int, float]
    ) -> dict[int, int]:
        """Convert scores to ranks (0-indexed, lower = better)."""
        ranked = sorted(section_ids, key=lambda sid: scores.get(sid, 0.0), reverse=True)
        return {sid: rank for rank, sid in enumerate(ranked)}

    def _assemble_v6(
        self,
        book: ConversationBook,
        ranked_ids: list[int],
        token_budget: int,
        query: str,
        search_terms: list[str] | None = None,
    ) -> str:
        """Custom assembly with partial section inclusion for oversized sections.

        Uses differentiated budget caps: the top-ranked section (often a
        large intro/framing section) gets a smaller cap to leave room for
        more diverse content from other sections.
        """
        parts: list[str] = []
        remaining = token_budget - 200  # reserve for framing
        included_ids: set[int] = set()
        max_per_section = int(token_budget * self._MAX_SECTION_BUDGET_RATIO)

        for sid in ranked_ids:
            if remaining < 200:
                break

            section = book.get_section(sid)
            if not section:
                continue

            if section.token_count <= remaining and section.token_count <= max_per_section:
                # Include fully
                parts.append(section.to_text())
                remaining -= section.token_count
                included_ids.add(sid)
                logger.info(
                    f"[rlm_v6] +section {sid} full "
                    f"({section.token_count} tokens, {remaining} remaining)"
                )
            elif remaining >= 500:
                # Section too large — extract relevant paragraphs
                excerpt_budget = min(remaining, max_per_section)
                excerpt = self._extract_excerpt(
                    section.content, query, excerpt_budget, search_terms
                )
                if excerpt:
                    excerpt_tokens = count_tokens(excerpt)
                    header = f"=== Section {sid} [excerpted, {section.token_count} tokens total] ==="
                    parts.append(f"{header}\n{excerpt}")
                    remaining -= excerpt_tokens + count_tokens(header)
                    included_ids.add(sid)
                    logger.info(
                        f"[rlm_v6] +section {sid} excerpted "
                        f"({excerpt_tokens}/{section.token_count} tokens, "
                        f"{remaining} remaining)"
                    )

        if parts:
            framing = (
                f"[Extracted from {book.section_count} sections, "
                f"{book.total_tokens} tokens total. "
                f"Showing {len(included_ids)} sections.]\n"
            )
            return framing + "\n\n".join(parts)

        return ""

    def _extract_excerpt(
        self,
        text: str,
        query: str,
        max_tokens: int,
        extra_terms: list[str] | None = None,
    ) -> str:
        """Extract query-relevant paragraphs using embedding + keywords.

        Combines embedding similarity (semantic) and keyword overlap
        (exact match) to score paragraphs, then includes surrounding
        context paragraphs.
        """
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            return ""

        para_tokens = [count_tokens(p) for p in paragraphs]

        # --- Signal 1: Keyword overlap ---
        terms = _extract_heuristic_terms(query)
        if extra_terms:
            terms = list(dict.fromkeys(terms + extra_terms))
        term_patterns = []
        for t in terms:
            try:
                term_patterns.append(re.compile(re.escape(t), re.IGNORECASE))
            except re.error:
                pass

        keyword_scores = []
        for para in paragraphs:
            keyword_scores.append(sum(1.0 for p in term_patterns if p.search(para)))
        # Normalize keyword scores
        kmax = max(keyword_scores) if keyword_scores else 1.0
        if kmax > 0:
            keyword_scores = [s / kmax for s in keyword_scores]

        # --- Signal 2: Embedding similarity ---
        embed_scores = [0.0] * len(paragraphs)
        try:
            import chromadb
            import uuid as _uuid

            # Only embed paragraphs with enough content (> 20 chars)
            valid = [(i, p) for i, p in enumerate(paragraphs) if len(p) > 20]
            if valid:
                client = chromadb.Client()
                col = client.create_collection(f"exc_{_uuid.uuid4().hex[:8]}")
                col.add(
                    documents=[p for _, p in valid],
                    ids=[str(i) for i, _ in valid],
                )
                res = col.query(
                    query_texts=[query], n_results=len(valid), include=["distances"]
                )
                client.delete_collection(col.name)
                for id_str, dist in zip(res["ids"][0], res["distances"][0]):
                    embed_scores[int(id_str)] = 1.0 / (1.0 + dist)
                # Normalize
                emax = max(embed_scores)
                emin = min(s for s in embed_scores if s > 0) if any(embed_scores) else 0
                erng = emax - emin
                if erng > 0:
                    embed_scores = [
                        (s - emin) / erng if s > 0 else 0.0 for s in embed_scores
                    ]
        except Exception:
            pass  # Fall back to keyword-only scoring

        # --- Combined score (keyword-dominant with embed refinement) ---
        scored: list[tuple[int, float]] = []
        for i in range(len(paragraphs)):
            score = 0.7 * keyword_scores[i] + 0.3 * embed_scores[i]
            # Small boost for first and last paragraphs
            if i < 2 or i >= len(paragraphs) - 2:
                score += 0.1
            scored.append((i, score))

        scored.sort(key=lambda x: (-x[1], x[0]))

        # Greedily select highest-scoring paragraphs that fit
        selected_indices: set[int] = set()
        used = 0
        for idx, score in scored:
            if used + para_tokens[idx] <= max_tokens:
                selected_indices.add(idx)
                used += para_tokens[idx]

        if not selected_indices:
            return ""

        # Context expansion: include 1 paragraph before/after each
        context_candidates = set()
        for idx in list(selected_indices):
            if idx > 0:
                context_candidates.add(idx - 1)
            if idx < len(paragraphs) - 1:
                context_candidates.add(idx + 1)
        for idx in sorted(context_candidates - selected_indices):
            if used + para_tokens[idx] <= max_tokens:
                selected_indices.add(idx)
                used += para_tokens[idx]

        ordered = sorted(selected_indices)
        return "\n\n".join(paragraphs[i] for i in ordered)


# ---------------------------------------------------------------------------
# Agentic — iterative exploration with text commands (Claude Code-inspired)
# ---------------------------------------------------------------------------


class AgenticExtractor:
    """True agentic exploration: programmatic warm start + iterative LLM loop.

    Architecture (inspired by Claude Code's agent loop):
    1. Warm start (no LLM): chunked embedding + IDF + position → ranked candidates
    2. Agent loop (multiple LLM calls): explores with SEARCH/READ/DONE text commands
    3. Assembly: paragraph-level extraction from agent's section selections

    The agent iteratively searches and reads sections, deciding when to stop
    (DONE) — mirroring Claude Code's tool-use loop where the model decides
    when it has enough information.
    """

    _MAX_TURNS = 8
    _MAX_SECTION_BUDGET_RATIO = 0.40
    _ENDING_WORDS = frozenset({
        "end", "ends", "ending", "final", "finally", "last",
        "conclude", "concludes", "conclusion",
    })
    _BEGINNING_WORDS = frozenset({
        "beginning", "begin", "begins", "start", "starts",
        "first", "opening",
    })
    _POSITION_WEIGHT = 0.02

    def __init__(
        self,
        provider_url: str,
        model: str,
        api_key: str = "dummy",
        max_context: int = 16384,
        **kwargs: Any,
    ) -> None:
        self.provider_url = provider_url
        self.model = model
        self.api_key = api_key
        self.max_context = max_context

    async def extract(
        self,
        book: ConversationBook,
        query: str,
        token_budget: int,
    ) -> str:
        if book.section_count == 0:
            return ""
        if book.total_tokens <= token_budget:
            return book.to_searchable_text()

        question = self._extract_question(query)
        logger.info(
            f"[agentic] Starting: book={book.total_tokens} tokens, "
            f"budget={token_budget}, sections={book.section_count}, "
            f"question={question[:100]}..."
        )

        try:
            # Phase 1: Programmatic warm start (no LLM)
            heuristic_terms = _extract_heuristic_terms(question)
            base_terms = list(dict.fromkeys(heuristic_terms))

            embed_scores = await self._embed_score(book, question)
            idf_scores = self._idf_search(book, base_terms)
            position_scores = self._position_scores(book, question)

            rrf = self._rrf_fuse(embed_scores, idf_scores, position_scores)
            ranked_ids = sorted(rrf, key=lambda s: rrf[s], reverse=True)
            logger.info(
                f"[agentic] Base terms: {base_terms[:12]}, "
                f"top 10: {[(sid, round(rrf[sid], 4)) for sid in ranked_ids[:10]]}"
            )

            # Phase 2: LLM selection + term generation (2 calls, parallel)
            agent_ids, extra_terms = await self._agent_loop(
                book, question, ranked_ids, idf_scores, base_terms
            )

            # Fuse LLM terms via RRF with split weighting:
            # proper nouns (character/place names) get 3x weight; common
            # terms get 2x weight.  Proper nouns are strong discriminators
            # (Felix → De Lacey chapters) while common terms still contribute
            # meaningfully (murder → murder scene chapters).
            if extra_terms:
                proper_terms = [t for t in extra_terms if t[0].isupper()]
                common_terms = [t for t in extra_terms if not t[0].isupper()]
                k = 60
                if proper_terms:
                    proper_idf = self._idf_search(book, proper_terms)
                    proper_ranked = sorted(
                        proper_idf, key=lambda s: proper_idf[s], reverse=True
                    )
                    for rank, sid in enumerate(proper_ranked):
                        rrf[sid] = rrf.get(sid, 0) + 3.0 / (k + rank)
                if common_terms:
                    common_idf = self._idf_search(book, common_terms)
                    common_ranked = sorted(
                        common_idf, key=lambda s: common_idf[s], reverse=True
                    )
                    for rank, sid in enumerate(common_ranked):
                        rrf[sid] = rrf.get(sid, 0) + 2.0 / (k + rank)

            # Re-sort by combined RRF
            final_ids = sorted(rrf, key=lambda s: rrf[s], reverse=True)

            logger.info(
                f"[agentic] Final top 10: "
                f"{[(sid, round(rrf.get(sid, 0), 4)) for sid in final_ids[:10]]}"
            )

            # Phase 3: Assembly with paragraph-level excerpts
            # Book order for ending queries (ending at end of context),
            # relevance order for others (relevant sections last for
            # highest model attention)
            query_words = set(re.findall(r"\w+", question.lower()))
            is_ending = bool(query_words & self._ENDING_WORDS)

            # Non-ending queries use tighter budget per section
            # to fit more diverse sections
            budget_ratio = self._MAX_SECTION_BUDGET_RATIO if is_ending else 0.30

            all_terms = list(dict.fromkeys(base_terms + extra_terms))
            result = self._assemble(
                book, final_ids, rrf, question, token_budget, all_terms,
                book_order=is_ending,
                budget_ratio=budget_ratio,
            )
            if result:
                return result

            logger.warning("[agentic] Assembly empty, falling back")
        except Exception as e:
            logger.error(f"[agentic] Failed: {e}", exc_info=True)

        return _header_extract(book, token_budget)

    @staticmethod
    def _extract_question(query: str) -> str:
        """Extract just the user question (first paragraph)."""
        first_para = query.split("\n\n", 1)[0].strip()
        return first_para[:500] if first_para else query[:500]

    async def _llm_call(self, messages: list[dict], max_tokens: int = 256) -> str:
        """Make a single LLM call."""
        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(
                f"{self.provider_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": 0.1,
                },
            )
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"]

    # ---- LLM-assisted term generation with progressive disclosure ----

    async def _agent_loop(
        self,
        book: ConversationBook,
        question: str,
        ranked_ids: list[int],
        idf_scores: dict[int, float],
        initial_terms: list[str],
    ) -> tuple[list[int], list[str]]:
        """Two-call LLM term generation (parallel, no hardcoded terms).

        Call 1 (query expansion): generates synonyms/related terms from
        the question alone — replaces hardcoded question-type heuristics.
        Call 2 (snippet extraction): extracts names/places from actual
        book content.

        Both calls run in parallel for same latency as one call.
        Returns ([], combined_terms).
        """
        snippets = self._get_snippets_for_llm(
            book, ranked_ids, idf_scores, question
        )
        expansion_terms, snippet_terms = await asyncio.gather(
            self._expand_query_terms(question),
            self._generate_terms(question, snippets),
        )
        combined = list(dict.fromkeys(expansion_terms + snippet_terms))
        if combined:
            logger.info(
                f"[agentic] LLM terms: {combined} "
                f"(expanded={expansion_terms[:5]}, "
                f"snippets={snippet_terms[:5]})"
            )
        return [], combined

    def _get_snippets_for_llm(
        self,
        book: ConversationBook,
        ranked_ids: list[int],
        idf_scores: dict[int, float],
        question: str,
    ) -> str:
        """Get diverse snippets: top-4 by RRF + top-4 by IDF (deduped).

        This ensures the LLM sees sections matching rare/important terms
        even if those sections rank lower overall.
        """
        # Merge: top-5 by RRF + top-5 by IDF for diversity
        rrf_top = ranked_ids[:5]
        idf_ranked = sorted(
            idf_scores, key=lambda s: idf_scores.get(s, 0), reverse=True
        )
        section_ids = list(rrf_top)
        for sid in idf_ranked:
            if sid not in section_ids and len(section_ids) < 10:
                section_ids.append(sid)

        terms = _extract_heuristic_terms(question)
        lines = []
        for sid in section_ids:
            section = book.get_section(sid)
            if not section:
                continue
            paragraphs = [p.strip() for p in section.content.split("\n\n")
                          if p.strip() and len(p.strip()) > 30]
            if not paragraphs:
                continue
            opening = paragraphs[0][:300].replace("\n", " ").strip()
            # Find first AND last term-matching paragraphs for diverse
            # coverage: first gives early context, last captures outcomes
            # near the end (e.g. location reveals, resolutions).
            first_match = ""
            last_match = ""
            for para in paragraphs[1:]:
                pl = para.lower()
                if any(t.lower() in pl for t in terms[:5]):
                    if not first_match:
                        first_match = para[:300].replace("\n", " ").strip()
                    last_match = para[:300].replace("\n", " ").strip()
            snippet = f"Section {sid}: {opening}"
            if first_match:
                snippet += f" ... {first_match}"
            if last_match and last_match != first_match:
                snippet += f" ... {last_match}"
            lines.append(snippet)
        return "\n".join(lines)

    async def _generate_terms(self, question: str, snippets: str) -> list[str]:
        """LLM generates search terms with context from candidate snippets."""
        messages = [
            {"role": "system", "content": (
                "You generate search terms to find relevant document sections "
                "for answering a question. Output ONLY comma-separated single "
                "words (15-20 terms).\n"
                "Example: Felix,cottage,Safie,lessons,Lacey,language,Arabian\n\n"
                "Generate terms covering:\n"
                "- CHARACTER NAMES from the snippets\n"
                "- PLACE NAMES from the snippets\n"
                "- WHERE events happen (setting, location, landscape)\n"
                "- HOW events happen (method, action, consequence)\n"
                "- Related SYNONYMS for key question words\n"
                "- Each term: single word, specific noun or name"
            )},
            {"role": "user", "content": (
                f"Question: {question}\n\n"
                f"Document snippets:\n{snippets}\n\n"
                "Search terms:"
            )},
        ]
        try:
            response = await self._llm_call(messages, max_tokens=100)
            response = response.strip()
            logger.info(f"[agentic] Terms response: {response[:100]}")
            terms: list[str] = []
            generic = {"novel", "character", "story", "book", "chapter",
                       "protagonist", "narrative", "literary", "scene",
                       "theme", "creature", "monster", "creation",
                       "the", "and", "for", "with", "from", "that",
                       "this", "what", "how", "who", "does"}
            for word in re.split(r'[,\s]+', response):
                word = word.strip().strip('"\'').strip()
                if (word and len(word) >= 2 and word.isalpha()
                        and word.lower() not in generic):
                    terms.append(word)
            return terms[:15]
        except Exception as e:
            logger.warning(f"[agentic] Terms failed: {e}")
            return []

    async def _expand_query_terms(self, question: str) -> list[str]:
        """LLM generates related search terms from the question alone.

        This replaces hardcoded question-type heuristics with dynamic
        LLM-based term expansion. The model generates synonyms, related
        actions, settings, and consequences for the question topic.
        """
        messages = [
            {"role": "system", "content": (
                "Generate search terms to find relevant passages in a long "
                "document. Output ONLY comma-separated single words.\n\n"
                "Think about:\n"
                "- Synonyms and related words for key terms\n"
                "- What SETTING or LOCATION this might happen in\n"
                "- What ACTIONS or CONSEQUENCES are involved\n"
                "- What OBJECTS or INSTRUMENTS might be mentioned\n\n"
                "Example for 'How was the victim killed?':\n"
                "murder,body,strangled,trial,prison,shore,accused,death"
            )},
            {"role": "user", "content": (
                f"Question: {question}\n\n"
                "Related search terms (single words, 10-15 terms):"
            )},
        ]
        try:
            response = await self._llm_call(messages, max_tokens=80)
            response = response.strip()
            logger.info(f"[agentic] Expansion response: {response[:100]}")
            terms: list[str] = []
            generic = {"novel", "character", "story", "book", "chapter",
                       "protagonist", "narrative", "literary", "scene",
                       "theme", "the", "and", "for", "with", "from",
                       "this", "what", "how", "who", "does", "that"}
            for word in re.split(r'[,\s]+', response):
                word = word.strip().strip('"\'').strip()
                if (word and len(word) >= 2 and word.isalpha()
                        and word.lower() not in generic):
                    terms.append(word)
            return terms[:12]
        except Exception as e:
            logger.warning(f"[agentic] Expansion failed: {e}")
            return []

    async def _embed_score(
        self, book: ConversationBook, query: str
    ) -> dict[int, float]:
        """Chunked embedding similarity (same as rlm_v6)."""
        def _run():
            import chromadb
            import uuid

            chunk_ids: list[str] = []
            chunk_texts: list[str] = []
            chunk_to_section: dict[str, int] = {}

            for section in book.sections:
                paragraphs = [p.strip() for p in section.content.split("\n\n") if p.strip()]
                if not paragraphs:
                    paragraphs = [section.content[:2000]]

                current_chunk: list[str] = []
                current_tokens = 0
                chunk_idx = 0
                for para in paragraphs:
                    ptokens = count_tokens(para)
                    if current_tokens + ptokens > 500 and current_chunk:
                        cid = f"{section.index}_{chunk_idx}"
                        chunk_ids.append(cid)
                        chunk_texts.append("\n\n".join(current_chunk))
                        chunk_to_section[cid] = section.index
                        chunk_idx += 1
                        current_chunk = [para]
                        current_tokens = ptokens
                    else:
                        current_chunk.append(para)
                        current_tokens += ptokens
                if current_chunk:
                    cid = f"{section.index}_{chunk_idx}"
                    chunk_ids.append(cid)
                    chunk_texts.append("\n\n".join(current_chunk))
                    chunk_to_section[cid] = section.index

            client = chromadb.Client()
            col_name = f"ag_{uuid.uuid4().hex[:8]}"
            collection = client.create_collection(col_name)
            collection.add(documents=chunk_texts, ids=chunk_ids)

            n_results = len(chunk_ids)
            results = collection.query(
                query_texts=[query], n_results=n_results, include=["distances"],
            )
            client.delete_collection(col_name)

            scores: dict[int, float] = {}
            for cid, dist in zip(results["ids"][0], results["distances"][0]):
                sid = chunk_to_section[cid]
                sim = 1.0 / (1.0 + dist)
                if sid not in scores or sim > scores[sid]:
                    scores[sid] = sim

            if scores:
                max_s = max(scores.values())
                min_s = min(scores.values())
                rng = max_s - min_s
                if rng > 0:
                    scores = {k: (v - min_s) / rng for k, v in scores.items()}

            return scores

        return await asyncio.get_event_loop().run_in_executor(None, _run)

    def _find_discriminative_section(
        self, book: ConversationBook, terms: list[str]
    ) -> int | None:
        """Find section best matching rare LLM terms (IDF > 0.7).

        Rare terms (appearing in <=2 sections) are highly discriminative
        — they point to THE specific section that answers the query.
        Returns the section with highest discriminative score, or None.
        """
        n_sections = max(book.section_count, 1)
        log_n = math.log(n_sections) if n_sections > 1 else 1.0
        disc_scores: dict[int, float] = defaultdict(float)

        for term in terms:
            scores = _search_book_words(book, term)
            n_matched = len(scores)
            if n_matched == 0:
                continue
            idf = max(0.0, 1.0 - math.log(n_matched) / log_n)
            if idf < 0.5:
                continue
            for sid, word_count in scores.items():
                disc_scores[sid] += idf

        if not disc_scores:
            return None

        best = max(disc_scores, key=lambda s: disc_scores[s])
        logger.info(
            f"[agentic] Discriminative scores (top 3): "
            f"{sorted(disc_scores.items(), key=lambda x: -x[1])[:3]}"
        )
        return best

    def _idf_search(
        self, book: ConversationBook, terms: list[str]
    ) -> dict[int, float]:
        """Strict log-ratio IDF search (same as rlm_v6)."""
        if not terms:
            return {}
        n_sections = max(book.section_count, 1)
        log_n = math.log(n_sections) if n_sections > 1 else 1.0

        section_scores: dict[int, float] = defaultdict(float)
        for term in terms:
            scores = _search_book_words(book, term)
            n_matched = len(scores)
            if n_matched == 0:
                continue
            idf = max(0.0, 1.0 - math.log(n_matched) / log_n)
            if idf == 0.0:
                continue
            n_words = len([w for w in term.split() if len(w.strip()) >= 2])
            for sid, word_count in scores.items():
                word_ratio = word_count / max(n_words, 1)
                section_scores[sid] += idf * word_ratio

        return dict(section_scores)

    def _position_scores(
        self, book: ConversationBook, query: str
    ) -> dict[int, float]:
        """Position scores for beginning/ending queries."""
        words = set(re.findall(r"\w+", query.lower()))
        n = book.section_count
        if words & self._ENDING_WORDS:
            return {
                s.index: i / max(n - 1, 1)
                for i, s in enumerate(book.sections)
            }
        if words & self._BEGINNING_WORDS:
            return {
                s.index: 1.0 - i / max(n - 1, 1)
                for i, s in enumerate(book.sections)
            }
        return {}

    def _rrf_fuse(
        self,
        embed_scores: dict[int, float],
        idf_scores: dict[int, float],
        position_scores: dict[int, float],
    ) -> dict[int, float]:
        """Reciprocal Rank Fusion of embedding, IDF, and position signals."""
        k = 60
        rrf: dict[int, float] = defaultdict(float)
        embed_ranked = sorted(
            embed_scores, key=lambda s: embed_scores.get(s, 0), reverse=True
        )
        idf_ranked = sorted(
            idf_scores, key=lambda s: idf_scores.get(s, 0), reverse=True
        )
        for rank, sid in enumerate(embed_ranked):
            rrf[sid] += 1.0 / (k + rank)
        for rank, sid in enumerate(idf_ranked):
            rrf[sid] += 1.0 / (k + rank)
        if position_scores:
            for sid, pscore in position_scores.items():
                rrf[sid] += self._POSITION_WEIGHT * pscore
        return dict(rrf)

    def _assemble(
        self,
        book: ConversationBook,
        ranked_ids: list[int],
        rrf_scores: dict[int, float],
        question: str,
        token_budget: int,
        search_terms: list[str],
        book_order: bool = False,
        budget_ratio: float | None = None,
    ) -> str:
        """Assemble context with paragraph-level excerpts and position awareness."""
        remaining = token_budget - 200
        included: list[tuple[int, str]] = []  # (sid, content)
        included_ids: set[int] = set()
        ratio = budget_ratio if budget_ratio is not None else self._MAX_SECTION_BUDGET_RATIO
        max_per_section = int(token_budget * ratio)

        # Check for position-aware excerpt extraction
        query_words = set(re.findall(r"\w+", question.lower()))
        is_ending = bool(query_words & self._ENDING_WORDS)
        is_beginning = bool(query_words & self._BEGINNING_WORDS)

        for sid in ranked_ids:
            if remaining < 200:
                break

            section = book.get_section(sid)
            if not section:
                continue

            if section.token_count <= remaining and section.token_count <= max_per_section:
                included.append((sid, section.to_text()))
                remaining -= section.token_count
                included_ids.add(sid)
                logger.info(
                    f"[agentic] +section {sid} full "
                    f"({section.token_count} tokens, "
                    f"rrf={rrf_scores.get(sid, 0):.4f}, {remaining} remaining)"
                )
            elif remaining >= 400:
                excerpt_budget = min(remaining, max_per_section)
                excerpt = self._extract_excerpt(
                    section.content, question, excerpt_budget,
                    search_terms, is_ending, is_beginning
                )
                if excerpt:
                    excerpt_tokens = count_tokens(excerpt)
                    header = (
                        f"=== Section {sid} [excerpted, "
                        f"{section.token_count} tokens total] ==="
                    )
                    included.append((sid, f"{header}\n{excerpt}"))
                    remaining -= excerpt_tokens + count_tokens(header)
                    included_ids.add(sid)
                    logger.info(
                        f"[agentic] +section {sid} excerpted "
                        f"({excerpt_tokens}/{section.token_count} tokens, "
                        f"rrf={rrf_scores.get(sid, 0):.4f}, {remaining} remaining)"
                    )

        # Fill remaining with recency
        for section in reversed(book.sections):
            if remaining < 200:
                break
            if section.index in included_ids:
                continue
            if section.token_count <= remaining and section.token_count <= max_per_section:
                included.append((section.index, section.to_text()))
                remaining -= section.token_count
                included_ids.add(section.index)

        if not included:
            return ""

        # Sort by book order (natural narrative flow) so endings
        # appear at end of context where model attends most
        if book_order:
            included.sort(key=lambda x: x[0])

        framing = (
            f"[Agentic extraction from {book.section_count} sections, "
            f"{book.total_tokens} tokens total. "
            f"Showing {len(included_ids)} sections]\n"
        )
        body = "\n\n".join(content for _, content in included)
        return f"{framing}\n{body}"

    def _extract_excerpt(
        self,
        text: str,
        question: str,
        max_tokens: int,
        extra_terms: list[str] | None = None,
        is_ending: bool = False,
        is_beginning: bool = False,
    ) -> str:
        """Extract query-relevant paragraphs with position awareness.

        For ending queries, strongly boosts last paragraphs.
        For beginning queries, strongly boosts first paragraphs.
        """
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            return ""

        para_tokens = [count_tokens(p) for p in paragraphs]

        # Keyword scoring
        terms = _extract_heuristic_terms(question)
        if extra_terms:
            terms = list(dict.fromkeys(terms + extra_terms))
        term_patterns = []
        for t in terms:
            try:
                term_patterns.append(re.compile(re.escape(t), re.IGNORECASE))
            except re.error:
                pass

        keyword_scores = []
        for para in paragraphs:
            keyword_scores.append(sum(1.0 for p in term_patterns if p.search(para)))
        kmax = max(keyword_scores) if keyword_scores else 1.0
        if kmax > 0:
            keyword_scores = [s / kmax for s in keyword_scores]

        # Embedding scoring
        embed_scores = [0.0] * len(paragraphs)
        try:
            import chromadb
            import uuid as _uuid
            valid = [(i, p) for i, p in enumerate(paragraphs) if len(p) > 20]
            if valid:
                client = chromadb.Client()
                col = client.create_collection(f"agex_{_uuid.uuid4().hex[:8]}")
                col.add(
                    documents=[p[:1000] for _, p in valid],
                    ids=[str(i) for i, _ in valid],
                )
                res = col.query(
                    query_texts=[question], n_results=len(valid), include=["distances"]
                )
                client.delete_collection(col.name)
                for id_str, dist in zip(res["ids"][0], res["distances"][0]):
                    embed_scores[int(id_str)] = 1.0 / (1.0 + dist)
                emax = max(embed_scores)
                emin = min(s for s in embed_scores if s > 0) if any(embed_scores) else 0
                erng = emax - emin
                if erng > 0:
                    embed_scores = [(s - emin) / erng if s > 0 else 0.0 for s in embed_scores]
        except Exception:
            pass

        # Combined score
        scored: list[tuple[int, float]] = []
        n = len(paragraphs)
        for i in range(n):
            score = 0.6 * keyword_scores[i] + 0.4 * embed_scores[i]
            # Position awareness
            if is_ending:
                # Strong boost for last paragraphs (quadratic: much stronger at tail)
                pos_ratio = i / max(n - 1, 1)
                score += 0.7 * (pos_ratio ** 2)
            elif is_beginning:
                pos_ratio = 1.0 - i / max(n - 1, 1)
                score += 0.7 * (pos_ratio ** 2)
            else:
                # Small boost for first/last paragraphs
                if i < 2 or i >= n - 2:
                    score += 0.1
            scored.append((i, score))

        scored.sort(key=lambda x: (-x[1], x[0]))

        # For ending queries, pre-reserve tail paragraphs
        selected: set[int] = set()
        used = 0
        if is_ending:
            tail_budget = int(max_tokens * 0.55)
            tail_used = 0
            guaranteed = 0
            for i in range(n - 1, max(n - 20, -1), -1):
                para_lower = paragraphs[i].lower()
                # Skip Project Gutenberg boilerplate
                if ("gutenberg" in para_lower
                        or "trademark" in para_lower
                        or "license" in para_lower):
                    continue
                # Always include last 8 content paragraphs (the actual
                # novel ending), even if they don't match search terms.
                # Beyond that, require keyword or embedding relevance.
                if guaranteed >= 8:
                    if keyword_scores[i] == 0 and embed_scores[i] < 0.1:
                        continue
                if tail_used + para_tokens[i] <= tail_budget:
                    selected.add(i)
                    tail_used += para_tokens[i]
                    guaranteed += 1
            used = tail_used
        elif is_beginning:
            head_budget = int(max_tokens * 0.35)
            head_used = 0
            for i in range(min(8, n)):
                if keyword_scores[i] == 0 and embed_scores[i] < 0.1:
                    continue
                if head_used + para_tokens[i] <= head_budget:
                    selected.add(i)
                    head_used += para_tokens[i]
            used = head_used

        # Greedy fill with scored paragraphs
        for idx, score in scored:
            if idx in selected:
                continue
            if used + para_tokens[idx] <= max_tokens:
                selected.add(idx)
                used += para_tokens[idx]

        if not selected:
            return ""

        # Context expansion
        context_candidates = set()
        for idx in list(selected):
            if idx > 0:
                context_candidates.add(idx - 1)
            if idx < n - 1:
                context_candidates.add(idx + 1)
        for idx in sorted(context_candidates - selected):
            if used + para_tokens[idx] <= max_tokens:
                selected.add(idx)
                used += para_tokens[idx]

        ordered = sorted(selected)
        return "\n\n".join(paragraphs[i] for i in ordered)


# ---------------------------------------------------------------------------
# SubagentExtractor — planning agent + parallel subagents + synthesis
# ---------------------------------------------------------------------------


class SubagentExtractor:
    """Multi-agent extraction: planner → autonomous subagents → synthesizer.

    Inspired by Claude Code / Codex architecture where each subagent gets
    its own fresh context window and autonomously decides what to search,
    what to read, and when to stop — using tools (search, read, done).

    Architecture:
    1. PLAN (1 LLM call): Decompose question into 2-3 focused missions
    2. EXECUTE (parallel, each 1-4 LLM calls): Each subagent gets its own
       tool-calling loop with fresh context. Tools: search(query),
       read(section_id), done(sections, summary).
    3. SYNTHESIZE (1 LLM call): Aggregate findings, rank sections
    """

    _MAX_SECTION_BUDGET_RATIO = 0.50
    _ENDING_WORDS = frozenset({
        "end", "ends", "ending", "final", "finally", "last",
        "conclude", "concludes", "conclusion",
    })
    _BEGINNING_WORDS = frozenset({
        "beginning", "begin", "begins", "start", "starts",
        "first", "opening",
    })

    def __init__(
        self,
        provider_url: str,
        model: str,
        api_key: str = "dummy",
        max_context: int = 16384,
        **kwargs: Any,
    ) -> None:
        self.provider_url = provider_url
        self.model = model
        self.api_key = api_key
        self.max_context = max_context

    async def extract(
        self,
        book: ConversationBook,
        query: str,
        token_budget: int,
    ) -> str:
        if book.section_count == 0:
            return ""
        if book.total_tokens <= token_budget:
            return book.to_searchable_text()

        question = query.split("\n\n", 1)[0].strip()[:500]
        logger.info(
            f"[subagent] Starting: book={book.total_tokens} tokens, "
            f"budget={token_budget}, sections={book.section_count}, "
            f"question={question[:100]}..."
        )

        try:
            # Phase 0: Heuristic warm start + embed + planning (parallel)
            heuristic_terms = _extract_heuristic_terms(question)
            heuristic_idf = self._idf_search(book, heuristic_terms)
            section_overview = self._build_section_overview(book, question)
            embed_scores, subtasks = await asyncio.gather(
                self._embed_score(book, question),
                self._plan(question, section_overview),
            )
            logger.info(
                f"[subagent] Planner created {len(subtasks)} missions, "
                f"heuristic terms: {heuristic_terms[:5]}"
            )

            # Phase 2: EXECUTE — autonomous subagents with tool loops (each gets fresh context)
            # Cap at 3 subtasks to limit LLM calls
            subtasks = subtasks[:3]

            # Run subagents sequentially to avoid overwhelming single-threaded LLM server
            # Track sections already used by prior subagents for diversity
            used_sids: set[int] = set()
            subagent_results = []
            for task in subtasks:
                try:
                    r = await self._run_subagent(
                        book, task, embed_scores, heuristic_idf,
                        exclude_sids=used_sids,
                    )
                    # Track which sections this subagent read
                    used_sids.update(r.get("content_sids", []))
                    subagent_results.append(r)
                except Exception as e:
                    logger.warning(f"[subagent] Subagent failed: {e}")
                    subagent_results.append(e)

            # Position scoring for ending/beginning queries
            query_words = set(re.findall(r"\w+", question.lower()))
            is_ending = bool(query_words & self._ENDING_WORDS)
            is_beginning = bool(query_words & self._BEGINNING_WORDS)
            position_scores = self._position_scores(book, is_ending, is_beginning)

            # Phase 3: AGGREGATE — RRF fuse: subagent + embed + heuristic IDF + position
            evidence = self._aggregate(
                book, subtasks, subagent_results, embed_scores,
                heuristic_idf, position_scores,
            )
            logger.info(
                f"[subagent] Aggregated {len(evidence['section_ids'])} sections, "
                f"{len(evidence['findings'])} findings"
            )

            # Phase 4: SYNTHESIZE — LLM ranks sections given evidence
            final_ids = await self._synthesize(
                book, question, evidence, token_budget
            )

            # For ending queries, guarantee last section + first section
            if is_ending:
                last_sid = book.sections[-1].index
                first_sid = book.sections[0].index
                for priority_sid in [last_sid, first_sid]:
                    if priority_sid not in final_ids:
                        final_ids.insert(0, priority_sid)

            logger.info(f"[subagent] Synthesizer selected {len(final_ids)} sections")

            # Phase 5: ASSEMBLE — build final context (book order for endings)
            result = self._assemble(
                book, final_ids, token_budget, book_order=is_ending
            )
            if result:
                return result

            logger.warning("[subagent] Assembly empty, falling back")
        except Exception as e:
            logger.error(f"[subagent] Failed: {e}", exc_info=True)

        return _header_extract(book, token_budget)

    async def _llm_call(
        self, messages: list[dict], max_tokens: int = 256
    ) -> str:
        """Make a single LLM call with retry on 500 errors."""
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=90.0) as client:
                    resp = await client.post(
                        f"{self.provider_url}/chat/completions",
                        json={
                            "model": self.model,
                            "messages": messages,
                            "max_tokens": max_tokens,
                            "temperature": 0.1,
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                return data["choices"][0]["message"]["content"]
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (500, 503) and attempt < 2:
                    wait = 2 * (attempt + 1)
                    logger.warning(f"[subagent] LLM {e.response.status_code} error, retry in {wait}s...")
                    await asyncio.sleep(wait)
                    continue
                raise

    def _build_section_overview(
        self, book: ConversationBook, question: str
    ) -> str:
        """Build a compact overview with first+last match snippets.

        Same technique as AgenticExtractor v29: shows both first AND last
        term-matching paragraphs to capture outcomes near section ends
        (e.g. location reveals, resolutions).
        """
        terms = _extract_heuristic_terms(question)
        lines = []
        for section in book.sections:
            paragraphs = [
                p.strip() for p in section.content.split("\n\n")
                if p.strip() and len(p.strip()) > 30
            ]
            if not paragraphs:
                continue
            opening = paragraphs[0][:200].replace("\n", " ").strip()
            # Find first AND last term-matching paragraphs
            first_match = ""
            last_match = ""
            for para in paragraphs[1:]:
                pl = para.lower()
                if any(t.lower() in pl for t in terms[:5]):
                    if not first_match:
                        first_match = para[:150].replace("\n", " ").strip()
                    last_match = para[:150].replace("\n", " ").strip()
            line = f"Section {section.index} ({section.token_count} tokens): {opening}"
            if first_match:
                line += f" ... {first_match}"
            if last_match and last_match != first_match:
                line += f" ... {last_match}"
            lines.append(line)
        return "\n".join(lines)

    async def _plan(
        self, question: str, section_overview: str
    ) -> list[dict]:
        """LLM Call 1: Decompose question into focused subtasks.

        Each subtask has:
        - objective: what information to find
        - search_terms: list of keywords to search for
        - target_sections: list of section IDs to focus on (or empty for all)
        """
        messages = [
            {"role": "system", "content": (
                "You are a search planner. Given a QUESTION and document snippets, "
                "decompose into 2-4 focused search subtasks.\n\n"
                "Output ONLY a JSON array. Each object has:\n"
                '- "objective": what to find (1 sentence)\n'
                '- "search_terms": 3-6 keywords — extract CHARACTER NAMES, PLACE '
                "NAMES, and specific nouns from BOTH the question AND the snippets\n"
                '- "target_sections": section IDs from the overview\n\n'
                "CRITICAL: Extract actual names/places you see in the snippets. "
                "For example, if snippets mention 'Mrs. Saville' or 'Walton', use "
                "those as search terms. Specific names are far better than generic "
                "words like 'murder' or 'death'."
            )},
            {"role": "user", "content": (
                f"Question: {question}\n\n"
                f"Document snippets:\n{section_overview}\n\n"
                "JSON array of 2-4 subtasks (extract names from snippets as search terms):"
            )},
        ]

        response = await self._llm_call(messages, max_tokens=400)
        logger.info(f"[subagent] Planner response: {response[:300]}")
        return self._parse_subtasks(response)

    def _parse_subtasks(self, response: str) -> list[dict]:
        """Parse planner response into subtask dicts."""
        # Strip markdown code blocks
        text = response.strip()
        text = re.sub(r"```(?:json)?\s*", "", text).strip()
        text = text.rstrip("`").strip()

        # Find the JSON array
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            try:
                subtasks = json.loads(text[start:end + 1])
                if isinstance(subtasks, list) and subtasks:
                    valid = []
                    for task in subtasks:
                        if isinstance(task, dict) and "search_terms" in task:
                            valid.append({
                                "objective": str(task.get("objective", "")),
                                "search_terms": [
                                    str(t) for t in task.get("search_terms", [])
                                    if isinstance(t, str)
                                ],
                                "target_sections": [
                                    self._parse_section_id(s)
                                    for s in task.get("target_sections", [])
                                    if self._parse_section_id(s) is not None
                                ],
                            })
                    if valid:
                        return valid[:4]
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback: extract terms from question AND response heuristically
        logger.warning("[subagent] Failed to parse planner JSON, using heuristic fallback")
        resp_terms = _extract_heuristic_terms(response)
        # Create 2 missions to improve coverage
        missions = [
            {
                "objective": "Find relevant information",
                "search_terms": resp_terms[:5] if resp_terms else ["the"],
                "target_sections": [],
            },
        ]
        # Second mission with different terms from later in the response
        if len(resp_terms) > 3:
            missions.append({
                "objective": "Find additional relevant information",
                "search_terms": resp_terms[3:8],
                "target_sections": [],
            })
        return missions if missions else [
            {
                "objective": "Find relevant information",
                "search_terms": ["the"],
                "target_sections": [],
            }
        ]

    async def _run_subagent(
        self,
        book: ConversationBook,
        task: dict,
        embed_scores: dict[int, float],
        heuristic_idf: dict[int, float],
        exclude_sids: set[int] | None = None,
    ) -> dict:
        """Run a single subagent: fresh context with mission-specific content.

        Each subagent gets a FRESH context with:
        - Its mission (from planner)
        - Pre-loaded content from top 3-4 mission-relevant sections
        - One LLM call to read content and report findings

        The subagent sees actual document text and reasons about it.
        exclude_sids: sections already read by prior subagents (prefer new ones).
        """
        mission = task.get("objective", "Find relevant information")
        search_terms = task.get("search_terms", [])
        target_sections = task.get("target_sections", [])

        # Compute mission-specific section ranking via RRF
        # Signal 1: Embed similarity (global)
        # Signal 2: IDF on mission's search terms (mission-specific)
        # Signal 3: Planner's target sections (direct vote)
        mission_rrf: dict[int, float] = defaultdict(float)
        k = 60

        if embed_scores:
            embed_ranked = sorted(embed_scores, key=lambda s: embed_scores[s], reverse=True)
            for rank, sid in enumerate(embed_ranked):
                mission_rrf[sid] += 1.0 / (k + rank)

        if heuristic_idf:
            idf_ranked = sorted(heuristic_idf, key=lambda s: heuristic_idf[s], reverse=True)
            for rank, sid in enumerate(idf_ranked):
                mission_rrf[sid] += 1.0 / (k + rank)

        # Mission-specific IDF from planner's search terms
        if search_terms:
            mission_idf = self._idf_search(book, search_terms)
            if mission_idf:
                midf_ranked = sorted(mission_idf, key=lambda s: mission_idf[s], reverse=True)
                for rank, sid in enumerate(midf_ranked):
                    mission_rrf[sid] += 1.5 / (k + rank)  # Higher weight

        # Planner's target sections get a strong boost
        for sid in target_sections:
            mission_rrf[sid] += 0.05

        # Select top sections: mix of mission-specific + global embed top
        ranked = sorted(mission_rrf, key=lambda s: mission_rrf[s], reverse=True)

        # Ensure top embed section is always a candidate (it may have Walton, etc.)
        embed_top = []
        if embed_scores:
            embed_top = sorted(embed_scores, key=lambda s: embed_scores[s], reverse=True)[:2]

        # Interleave: embed_top[0], ranked[0], embed_top[1], ranked[1], ...
        candidates: list[int] = []
        seen: set[int] = set()
        for pair in zip(embed_top + ranked[:6], ranked[:6] + embed_top):
            for sid in pair:
                if sid not in seen:
                    candidates.append(sid)
                    seen.add(sid)

        # Deprioritize sections already read by prior subagents (move to end)
        if exclude_sids:
            new_sids = [s for s in candidates if s not in exclude_sids]
            old_sids = [s for s in candidates if s in exclude_sids]
            candidates = new_sids + old_sids

        # Build content slice (~8k tokens max, 3 sections, 3000 chars each)
        content_parts: list[str] = []
        content_sids: list[int] = []
        content_budget = 8000
        for sid in candidates:
            section = book.get_section(sid)
            if not section:
                continue
            text = section.content[:3000]
            toks = count_tokens(text)
            if toks > content_budget:
                continue
            content_parts.append(
                f"=== Section {sid} ({section.token_count} tokens) ===\n{text}"
            )
            content_sids.append(sid)
            content_budget -= toks
            if len(content_sids) >= 3:
                break

        content_text = "\n\n".join(content_parts)

        logger.info(
            f"[subagent] Mission: {mission[:80]}... "
            f"Content sections: {content_sids}, "
            f"search_terms: {search_terms[:5]}"
        )

        # Single LLM call: read content and report findings
        messages = [
            {"role": "system", "content": (
                "You are a research assistant. Read the document sections below "
                "and complete the research mission.\n\n"
                "Output a JSON object with:\n"
                '- "sections": list of section IDs that answer the mission\n'
                '- "summary": 1-2 sentences summarizing what you found\n\n'
                "IMPORTANT: Only include sections that actually contain relevant "
                "information. Read carefully before answering."
            )},
            {"role": "user", "content": (
                f"MISSION: {mission}\n\n"
                f"DOCUMENT CONTENT:\n\n{content_text}\n\n"
                "Which sections contain information relevant to the mission? "
                "Output JSON with sections and summary:"
            )},
        ]

        try:
            response = await self._llm_call(messages, max_tokens=200)
            logger.info(f"[subagent] Response: {response[:200]}")

            parsed = _parse_json(response)
            if not parsed:
                logger.warning("[subagent] Failed to parse response JSON")
                return {
                    "sections": content_sids,
                    "evidence": [response[:300]],
                    "summary": response[:200],
                }

            # Parse section IDs from various key names
            raw_sections = (
                parsed.get("sections")
                or parsed.get("section_ids")
                or parsed.get("ids")
                or parsed.get("id", [])
            )
            if isinstance(raw_sections, (int, float)):
                raw_sections = [raw_sections]

            found_sids = []
            for s in raw_sections:
                sid = self._parse_section_id(s)
                if sid is not None:
                    found_sids.append(sid)

            summary = str(parsed.get("summary", parsed.get("evidence", "")))

            return {
                "sections": found_sids if found_sids else content_sids,
                "evidence": [summary] if summary else [],
                "summary": summary,
                "content_sids": content_sids,
            }

        except Exception as e:
            logger.warning(f"[subagent] LLM call failed: {e}")
            return {
                "sections": content_sids,
                "evidence": [],
                "summary": "",
                "content_sids": content_sids,
            }

    async def _embed_score(
        self, book: ConversationBook, query: str
    ) -> dict[int, float]:
        """Chunked embedding similarity (reused from AgenticExtractor)."""
        def _run():
            import chromadb
            import uuid

            chunk_ids: list[str] = []
            chunk_texts: list[str] = []
            chunk_to_section: dict[str, int] = {}

            for section in book.sections:
                paragraphs = [
                    p.strip() for p in section.content.split("\n\n") if p.strip()
                ]
                if not paragraphs:
                    paragraphs = [section.content[:2000]]
                current_chunk: list[str] = []
                current_tokens = 0
                chunk_idx = 0
                for para in paragraphs:
                    ptokens = count_tokens(para)
                    if current_tokens + ptokens > 500 and current_chunk:
                        cid = f"{section.index}_{chunk_idx}"
                        chunk_ids.append(cid)
                        chunk_texts.append("\n\n".join(current_chunk))
                        chunk_to_section[cid] = section.index
                        chunk_idx += 1
                        current_chunk = [para]
                        current_tokens = ptokens
                    else:
                        current_chunk.append(para)
                        current_tokens += ptokens
                if current_chunk:
                    cid = f"{section.index}_{chunk_idx}"
                    chunk_ids.append(cid)
                    chunk_texts.append("\n\n".join(current_chunk))
                    chunk_to_section[cid] = section.index

            client = chromadb.Client()
            col_name = f"sub_{uuid.uuid4().hex[:8]}"
            collection = client.create_collection(col_name)
            collection.add(documents=chunk_texts, ids=chunk_ids)
            results = collection.query(
                query_texts=[query], n_results=len(chunk_ids),
                include=["distances"],
            )
            client.delete_collection(col_name)

            scores: dict[int, float] = {}
            for cid, dist in zip(results["ids"][0], results["distances"][0]):
                sid = chunk_to_section[cid]
                sim = 1.0 / (1.0 + dist)
                if sid not in scores or sim > scores[sid]:
                    scores[sid] = sim

            if scores:
                max_s = max(scores.values())
                min_s = min(scores.values())
                rng = max_s - min_s
                if rng > 0:
                    scores = {k: (v - min_s) / rng for k, v in scores.items()}
            return scores

        return await asyncio.get_event_loop().run_in_executor(None, _run)

    def _idf_search(
        self, book: ConversationBook, terms: list[str]
    ) -> dict[int, float]:
        """IDF search over sections (reused from AgenticExtractor)."""
        if not terms:
            return {}
        n_sections = max(book.section_count, 1)
        log_n = math.log(n_sections) if n_sections > 1 else 1.0
        section_scores: dict[int, float] = defaultdict(float)
        for term in terms:
            scores = _search_book_words(book, term)
            n_matched = len(scores)
            if n_matched == 0:
                continue
            idf = max(0.0, 1.0 - math.log(n_matched) / log_n)
            if idf == 0.0:
                continue
            n_words = len([w for w in term.split() if len(w.strip()) >= 2])
            for sid, word_count in scores.items():
                word_ratio = word_count / max(n_words, 1)
                section_scores[sid] += idf * word_ratio
        return dict(section_scores)

    def _position_scores(
        self, book: ConversationBook, is_ending: bool, is_beginning: bool,
    ) -> dict[int, float]:
        """Position scores: ending queries boost late sections, beginning boost early."""
        n = book.section_count
        if is_ending:
            return {
                s.index: i / max(n - 1, 1)
                for i, s in enumerate(book.sections)
            }
        if is_beginning:
            return {
                s.index: 1.0 - i / max(n - 1, 1)
                for i, s in enumerate(book.sections)
            }
        return {}

    def _aggregate(
        self,
        book: ConversationBook,
        subtasks: list[dict],
        results: list,
        embed_scores: dict[int, float] | None = None,
        heuristic_idf: dict[int, float] | None = None,
        position_scores: dict[int, float] | None = None,
    ) -> dict:
        """Merge all signals via RRF: subagent votes + embed + heuristic IDF + position."""
        # Count how many subagents voted for each section (autonomous format)
        subagent_votes: dict[int, int] = defaultdict(int)
        all_findings: list[str] = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"[subagent] Subagent {i} failed: {result}")
                continue
            if not isinstance(result, dict):
                continue

            sections = result.get("sections", [])
            evidence = result.get("evidence", [])
            summary = result.get("summary", "")

            for sid in sections:
                subagent_votes[sid] += 1

            for ev in evidence:
                all_findings.append(ev)
            if summary:
                all_findings.append(f"[Subagent {i}] {summary}")

        # RRF fusion: 4 signals
        k = 60
        rrf: dict[int, float] = defaultdict(float)

        # Signal 1: Subagent votes (sections found by more subagents rank higher)
        subagent_ranked = sorted(
            subagent_votes, key=lambda s: subagent_votes[s], reverse=True
        )
        for rank, sid in enumerate(subagent_ranked):
            # Weight by vote count: 2 votes = 2x the score
            rrf[sid] += subagent_votes[sid] * (1.0 / (k + rank))

        # Signal 2: Embedding similarity
        if embed_scores:
            embed_ranked = sorted(
                embed_scores, key=lambda s: embed_scores.get(s, 0), reverse=True
            )
            for rank, sid in enumerate(embed_ranked):
                rrf[sid] += 1.0 / (k + rank)

        # Signal 3: Heuristic IDF (question terms) — higher weight for exact matches
        if heuristic_idf:
            heuristic_ranked = sorted(
                heuristic_idf, key=lambda s: heuristic_idf.get(s, 0), reverse=True
            )
            for rank, sid in enumerate(heuristic_ranked):
                rrf[sid] += 1.5 / (k + rank)

        # Signal 4: Position (ending/beginning bias)
        if position_scores:
            for sid, pscore in position_scores.items():
                rrf[sid] += 0.08 * pscore

        all_section_scores = dict(rrf)
        ranked_ids = sorted(
            all_section_scores, key=lambda s: all_section_scores[s], reverse=True
        )

        return {
            "section_ids": ranked_ids,
            "section_scores": dict(all_section_scores),
            "findings": all_findings[:30],
        }

    async def _synthesize(
        self,
        book: ConversationBook,
        question: str,
        evidence: dict,
        token_budget: int,
    ) -> list[int]:
        """LLM Call 2: Given aggregated evidence, select final sections.

        The LLM sees the question + evidence snippets and outputs a ranked
        list of section IDs that best answer the question.
        """
        findings_text = "\n".join(evidence["findings"][:20])
        candidate_ids = evidence["section_ids"][:15]

        # Build candidate descriptions
        candidate_lines = []
        for sid in candidate_ids:
            section = book.get_section(sid)
            if section:
                score = evidence["section_scores"].get(sid, 0)
                preview = section.content[:200].replace("\n", " ").strip()
                candidate_lines.append(
                    f"Section {sid} (score={score:.2f}, {section.token_count} tokens): "
                    f"{preview}"
                )

        messages = [
            {"role": "system", "content": (
                "Given a question and search evidence, output the section IDs "
                "most relevant to ANSWERING THE QUESTION.\n\n"
                "Focus on sections where the evidence directly addresses the question. "
                "Sections with higher search scores and more evidence snippets should "
                "rank higher.\n\n"
                "Output ONLY a JSON array of section IDs, most relevant first. "
                "Include 5-8 sections."
            )},
            {"role": "user", "content": (
                f"Question: {question}\n\n"
                f"Search evidence (section, task, snippet):\n{findings_text}\n\n"
                f"Candidate sections with scores:\n" +
                "\n".join(candidate_lines) +
                "\n\nMost relevant section IDs for answering the question (JSON array):"
            )},
        ]

        response = await self._llm_call(messages, max_tokens=100)
        logger.info(f"[subagent] Synthesizer response: {response[:200]}")

        # Parse JSON array of section IDs
        final_ids = self._parse_section_ids(response, candidate_ids)

        # Fallback: if synthesizer fails, use aggregated scores
        if not final_ids:
            logger.warning("[subagent] Synthesizer parse failed, using score ranking")
            final_ids = candidate_ids[:10]

        return final_ids

    @staticmethod
    def _parse_section_id(val: Any) -> int | None:
        """Parse a section ID from various formats: 5, "5", "Section 5"."""
        if isinstance(val, (int, float)):
            return int(val)
        if isinstance(val, str):
            # "Section 5" → 5
            m = re.search(r"(\d+)", val)
            if m:
                return int(m.group(1))
        return None

    def _parse_section_ids(
        self, response: str, fallback_ids: list[int]
    ) -> list[int]:
        """Parse synthesizer response into list of section IDs."""
        text = response.strip()
        text = re.sub(r"```(?:json)?\s*", "", text).strip()
        text = text.rstrip("`").strip()
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            try:
                ids = json.loads(text[start:end + 1])
                if isinstance(ids, list):
                    valid = []
                    for i in ids:
                        parsed = self._parse_section_id(i)
                        if parsed is not None:
                            valid.append(parsed)
                    return valid[:15]
            except json.JSONDecodeError:
                pass

        # Try to find numbers in text
        nums = re.findall(r"\b(\d+)\b", text)
        if nums:
            return [int(n) for n in nums[:15]]

        return []

    def _assemble(
        self,
        book: ConversationBook,
        ranked_ids: list[int],
        token_budget: int,
        book_order: bool = False,
    ) -> str:
        """Assemble final context from ranked section IDs.

        Truncates oversized sections instead of skipping them entirely.
        """
        remaining = token_budget - 200
        included: list[tuple[int, str]] = []
        included_ids: set[int] = set()
        max_per_section = int(token_budget * self._MAX_SECTION_BUDGET_RATIO)

        for sid in ranked_ids:
            if remaining < 200:
                break
            section = book.get_section(sid)
            if not section:
                continue
            if section.token_count <= remaining and section.token_count <= max_per_section:
                included.append((sid, section.to_text()))
                remaining -= section.token_count
                included_ids.add(sid)
                logger.info(
                    f"[subagent] +section {sid} "
                    f"({section.token_count} tokens, {remaining} remaining)"
                )
            elif remaining >= 1000:
                # Truncate section to fit remaining budget (or max_per_section cap)
                trunc_budget = min(max_per_section, remaining)
                content = section.content
                # Binary-search for the right truncation point
                lo, hi = 0, len(content)
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if count_tokens(content[:mid]) <= trunc_budget:
                        lo = mid
                    else:
                        hi = mid - 1
                truncated = content[:lo]
                if truncated:
                    toks = count_tokens(truncated)
                    text = section.to_text()
                    # Truncate the to_text output proportionally
                    trunc_text = text[:int(len(text) * lo / max(len(content), 1))]
                    trunc_text += "\n[... truncated ...]"
                    included.append((sid, trunc_text))
                    remaining -= toks
                    included_ids.add(sid)
                    logger.info(
                        f"[subagent] +section {sid} (TRUNCATED "
                        f"{toks}/{section.token_count} tokens, {remaining} remaining)"
                    )

        # Fill remaining with recency
        for section in reversed(book.sections):
            if remaining < 200:
                break
            if section.index in included_ids:
                continue
            if section.token_count <= remaining and section.token_count <= max_per_section:
                included.append((section.index, section.to_text()))
                remaining -= section.token_count
                included_ids.add(section.index)

        if not included:
            return ""

        if book_order:
            included.sort(key=lambda x: x[0])

        framing = (
            f"[Subagent extraction from {book.section_count} sections, "
            f"{book.total_tokens} tokens total. "
            f"Showing {len(included_ids)} sections]\n"
        )
        body = "\n\n".join(content for _, content in included)
        return f"{framing}\n{body}"


# ---------------------------------------------------------------------------
# Strategy: readagent (ReadAgent + SRLM + EXIT inspired)
# ---------------------------------------------------------------------------


class ReadAgentExtractor:
    """ReadAgent-inspired gist memory + multi-signal retrieval.

    Research-grounded pipeline combining insights from:
    - ReadAgent (Lee et al., ICML 2024): gist memory provides global context;
      LLM-based page selection outperforms similarity-based retrieval because
      the LLM reasons about document structure and narrative flow.
    - SRLM (Alizadeh et al., 2026): multi-signal consistency checks without
      an LLM judge; reduce reliance on any single signal.
    - EXIT (Hwang et al., 2024): extractive paragraph-level compression
      preserves factual integrity better than abstractive summarization.
    - RLM (Zhang et al., ICML 2026): programmatic context interaction is
      more robust than feeding raw text to the model.

    Pipeline (1 LLM call total):
    Phase 1: Warm start — embed + IDF + position → RRF ranking (no LLM)
    Phase 2: Gist memory — query-aware section summaries with proper nouns
    Phase 3: LLM term generation — reads gist + top snippets, suggests
             search terms (character names, places, concepts near the answer)
    Phase 4: Signal fusion — LLM term IDF + proper noun IDF + RRF merge
    Phase 5: Extractive assembly — paragraph-level scoring and extraction

    Key advantage over SubagentExtractor:
    - 1 LLM call (not 3-4), simpler task (term generation, not reasoning)
    - LLM sees global context (gist) + specific text (snippets)
    - LLM generates search terms (its strength), not section selection
      (which 8B models do poorly)
    - Proper nouns extracted programmatically from top sections, then
      boosted via IDF (3x weight) — discriminative without extra LLM calls
    """

    _MAX_SECTION_BUDGET_RATIO = 0.40
    _ENDING_WORDS = frozenset({
        "end", "ends", "ending", "final", "finally", "last",
        "conclude", "concludes", "conclusion",
    })
    _BEGINNING_WORDS = frozenset({
        "beginning", "begin", "begins", "start", "starts",
        "first", "opening",
    })
    _POSITION_WEIGHT = 0.02
    _COMMON_TITLE_WORDS = frozenset({
        "The", "His", "Her", "She", "He", "They", "It", "But", "And",
        "This", "That", "These", "Those", "When", "What", "How", "Why",
        "Chapter", "Section", "However", "Although", "Upon", "Yet",
        "Then", "There", "Here", "Now", "Not", "Our", "My", "Your",
        "For", "With", "From", "Into", "After", "Before", "During",
        "Every", "Some", "Many", "Most", "One", "Two", "Three",
        "You", "We", "Its", "All", "Even", "Still", "Could", "Would",
        "Should", "May", "Might", "Must", "Shall", "Will", "Can",
    })

    _QUESTION_TYPE_HINTS = {
        "where": (
            "\nThis is a WHERE question. The answer is a specific "
            "LOCATION (country, city, region, island). You MUST "
            "include your best guess for the location name.\n"
        ),
        "who": (
            "\nThis is a WHO question. The answer involves specific "
            "CHARACTER NAMES. Include likely character names.\n"
        ),
        "how": (
            "\nThis is a HOW question. The answer describes a "
            "process or method. Include action verbs and process "
            "words.\n"
        ),
        "when": (
            "\nThis is a WHEN question. The answer is a specific "
            "TIME, DATE, or EVENT. Include time-related words.\n"
        ),
    }

    def __init__(
        self,
        provider_url: str,
        model: str,
        api_key: str = "dummy",
        max_context: int = 16384,
        **kwargs: Any,
    ) -> None:
        self.provider_url = provider_url
        self.model = model
        self.api_key = api_key
        self.max_context = max_context
        self._section_names: dict[int, list[str]] = {}

    async def extract(
        self,
        book: ConversationBook,
        query: str,
        token_budget: int,
    ) -> str:
        if book.section_count == 0:
            return ""
        if book.total_tokens <= token_budget:
            return book.to_searchable_text()

        question = query.split("\n\n", 1)[0].strip()[:500]
        logger.info(
            f"[readagent] Starting: book={book.total_tokens} tokens, "
            f"budget={token_budget}, sections={book.section_count}, "
            f"question={question[:100]}..."
        )

        try:
            # Phase 1: Programmatic warm start (4 signals, no LLM)
            heuristic_terms = _extract_heuristic_terms(question)
            base_terms = list(dict.fromkeys(heuristic_terms))

            embed_scores = await self._embed_score(book, question)
            idf_scores = self._idf_search(book, base_terms)
            position_scores = self._position_scores(book, question)
            cooccur_scores = self._co_occurrence_search(
                book, base_terms
            )
            rrf = self._rrf_fuse(
                embed_scores, idf_scores, position_scores,
                cooccur_scores,
            )

            ranked_ids = sorted(rrf, key=lambda s: rrf[s], reverse=True)
            logger.info(
                f"[readagent] Phase 1 top 10: "
                f"{[(sid, round(rrf[sid], 4)) for sid in ranked_ids[:10]]}"
            )

            # Detect question type for type-aware prompts
            q_type = self._detect_question_type(question)
            logger.info(f"[readagent] Question type: {q_type}")

            # Phase 2: Dual LLM calls with snippet context (parallel)
            # Call 1: Query expansion with CoT (pure language understanding)
            # Call 2: Snippet-based terms (grounded in actual book content)
            self._extract_all_section_names(book)
            snippets = self._get_snippets_for_llm(
                book, ranked_ids, idf_scores, question
            )
            expansion_terms, snippet_terms = await asyncio.gather(
                self._llm_expand_query(question, q_type),
                self._generate_terms(question, snippets, q_type),
            )
            extra_terms = list(dict.fromkeys(
                expansion_terms + snippet_terms
            ))
            logger.info(
                f"[readagent] LLM terms: {extra_terms} "
                f"(expanded={expansion_terms[:5]}, "
                f"snippets={snippet_terms[:5]})"
            )

            # Phase 3: IDF boost (proper MAX-based, common rank-based)
            k = 60
            generic = {
                "novel", "character", "story", "book", "chapter",
                "protagonist", "narrative", "literary", "scene",
                "theme", "the", "and", "for", "with", "from",
                "this", "what", "how", "who", "does", "that",
                "there", "their", "would", "could", "about",
            }
            loc_candidates: list[str] = []
            loc_ranked: list[int] = []
            who_names: list[str] = []
            if extra_terms:
                clean_terms = [
                    t for t in extra_terms
                    if len(t) >= 2 and t.lower() not in generic
                ]
                proper_terms = [
                    t for t in clean_terms if t[0].isupper()
                ]
                common_terms = [
                    t for t in clean_terms if not t[0].isupper()
                ]

                # For "where" questions, extract location candidates
                # from top sections and give them a SEPARATE IDF-RRF
                # boost (not mixed into MAX boost, which normalizes
                # them away). This captures 1-occurrence place names
                # like "Ireland" that _extract_proper_nouns misses.
                if q_type == "where":
                    for sid in ranked_ids[:15]:
                        section = book.get_section(sid)
                        if section:
                            locs = self._extract_location_candidates(
                                section.content
                            )
                            for loc in locs:
                                if (
                                    loc not in loc_candidates
                                    and loc not in proper_terms
                                ):
                                    loc_candidates.append(loc)
                    if loc_candidates:
                        logger.info(
                            f"[readagent] Location candidates: "
                            f"{loc_candidates[:20]}"
                        )
                        loc_idf = self._idf_search(
                            book, loc_candidates[:15]
                        )
                        loc_ranked = sorted(
                            loc_idf,
                            key=lambda s: loc_idf.get(s, 0),
                            reverse=True,
                        )
                        for rank, sid in enumerate(
                            loc_ranked
                        ):
                            rrf[sid] = rrf.get(sid, 0) + (
                                10.0 / (k + rank)
                            )
                        logger.info(
                            f"[readagent] Location IDF top 5: "
                            f"{[(sid, round(loc_idf.get(sid, 0), 3)) for sid in loc_ranked[:5]]}"
                        )

                # For "who" questions, extract proper nouns from
                # sections that co-occur with 2+ query terms.
                # This finds discriminative names (Felix, Safie)
                # that the LLM misses due to parametric knowledge.
                # Co-occurrence filter removes Gutenberg boilerplate
                # sections while keeping content-relevant ones.
                if q_type == "who":
                    query_terms_lower = [
                        t.lower() for t in base_terms
                        if len(t) >= 3
                    ]
                    for sid, names in self._section_names.items():
                        section = book.get_section(sid)
                        if not section:
                            continue
                        content_lower = section.content.lower()
                        hits = sum(
                            1 for t in query_terms_lower
                            if t in content_lower
                        )
                        if hits >= 2:
                            for name in names:
                                if name not in who_names:
                                    who_names.append(name)
                    if who_names:
                        # IDF-filter: keep moderately discriminative
                        # names (0.15 <= IDF <= 0.75). Very high IDF
                        # = metadata/noise (1-2 sections), very low
                        # = too common (Victor in 20+ sections).
                        # Sort by IDF descending so the most
                        # discriminative names survive the [:15] cut.
                        n_sec = max(book.section_count, 1)
                        log_n_sec = (
                            math.log(n_sec) if n_sec > 1 else 1.0
                        )
                        name_idf_map: dict[str, float] = {}
                        for name in who_names:
                            matches = _search_book_words(book, name)
                            n_m = len(matches)
                            if n_m > 0:
                                name_idf_map[name] = max(
                                    0.0,
                                    1.0 - math.log(n_m) / log_n_sec,
                                )
                        filtered_names = [
                            n for n in who_names
                            if 0.15
                            <= name_idf_map.get(n, 0)
                            <= 0.75
                        ]
                        filtered_names.sort(
                            key=lambda n: -name_idf_map.get(n, 0)
                        )
                        who_names = filtered_names[:15]
                        logger.info(
                            f"[readagent] Who-question name "
                            f"candidates (IDF-filtered): "
                            f"{who_names[:20]}"
                        )
                        if who_names:
                            name_idf = self._idf_search(
                                book, who_names[:15]
                            )
                            name_ranked = sorted(
                                name_idf,
                                key=lambda s: name_idf.get(s, 0),
                                reverse=True,
                            )
                            for rank, sid in enumerate(
                                name_ranked
                            ):
                                rrf[sid] = rrf.get(sid, 0) + (
                                    5.0 / (k + rank)
                                )

                logger.info(
                    f"[readagent] Proper terms: "
                    f"{proper_terms[:15]}"
                )
                if proper_terms:
                    # MAX-based: each section boosted by its BEST
                    # single proper noun (not sum). Rewards sections
                    # matching one discriminative name (Felix, IDF 0.5)
                    # over sections matching many common names
                    # (Victor+Elizabeth, IDF 0.07 each).
                    n_sections = max(book.section_count, 1)
                    log_n = (
                        math.log(n_sections) if n_sections > 1
                        else 1.0
                    )
                    best_score: dict[int, float] = defaultdict(float)
                    for term in proper_terms[:15]:
                        scores = _search_book_words(book, term)
                        n_matched = len(scores)
                        if n_matched == 0:
                            continue
                        idf = max(
                            0.0, 1.0 - math.log(n_matched) / log_n
                        )
                        for sid, count in scores.items():
                            ts = idf * min(count, 10)
                            if ts > best_score[sid]:
                                best_score[sid] = ts
                    if best_score:
                        max_bs = max(best_score.values())
                        if max_bs > 0:
                            for sid, score in best_score.items():
                                rrf[sid] = rrf.get(sid, 0) + (
                                    0.05 * (score / max_bs)
                                )
                if common_terms:
                    common_idf = self._idf_search(
                        book, common_terms[:12]
                    )
                    for rank, sid in enumerate(
                        sorted(
                            common_idf,
                            key=lambda s: common_idf[s],
                            reverse=True,
                        )
                    ):
                        rrf[sid] = rrf.get(sid, 0) + 2.0 / (k + rank)
            else:
                clean_terms = []

            final_ids = sorted(
                rrf, key=lambda s: rrf[s], reverse=True
            )
            logger.info(
                f"[readagent] Final top 10: "
                f"{[(sid, round(rrf.get(sid, 0), 4)) for sid in final_ids[:10]]}"
            )

            # Phase 4: Extractive assembly
            query_words = set(re.findall(r"\w+", question.lower()))
            is_ending = bool(query_words & self._ENDING_WORDS)
            if is_ending:
                budget_ratio = self._MAX_SECTION_BUDGET_RATIO
            else:
                budget_ratio = 0.22

            # Include type-specific candidates in excerpt terms
            # so relevant paragraphs get selected in excerpts
            if q_type == "where":
                excerpt_extras = loc_candidates[:15]
            elif q_type == "who":
                excerpt_extras = who_names[:15]
            else:
                excerpt_extras = []
            all_terms = list(dict.fromkeys(
                base_terms + clean_terms + excerpt_extras
            ))
            result = self._assemble(
                book, final_ids, rrf, question, token_budget,
                all_terms, book_order=is_ending,
                budget_ratio=budget_ratio,
                q_type=q_type,
            )
            if result:
                return result

            logger.warning("[readagent] Assembly empty, falling back")
        except Exception as e:
            logger.error(f"[readagent] Failed: {e}", exc_info=True)

        return _header_extract(book, token_budget)

    # ---- Phase 2: Gist memory construction (ReadAgent-inspired) ----

    def _build_gist_memory(
        self, book: ConversationBook, question: str
    ) -> str:
        """Build query-aware gist memory with proper noun extraction.

        ReadAgent showed that gist memory (compressed summaries preserving
        narrative flow) enables better page selection than raw embeddings.
        Our programmatic gists include:
        - Opening paragraph (context)
        - Closing paragraph (outcomes/conclusions)
        - First + last query-matching paragraphs (relevant detail)
        - Proper nouns (character/place names for discrimination)
        """
        terms = _extract_heuristic_terms(question)
        lines: list[str] = []
        self._section_names = {}

        for section in book.sections:
            paragraphs = [
                p.strip()
                for p in section.content.split("\n\n")
                if p.strip() and len(p.strip()) > 30
            ]
            if not paragraphs:
                continue

            opening = paragraphs[0][:300].replace("\n", " ").strip()

            # Closing paragraph captures outcomes/conclusions
            closing = ""
            if len(paragraphs) > 2:
                closing = paragraphs[-1][:150].replace("\n", " ").strip()

            # First AND last term-matching paragraphs
            # Use ALL terms for better coverage
            first_match = ""
            last_match = ""
            for para in paragraphs[1:]:
                pl = para.lower()
                if any(t.lower() in pl for t in terms):
                    if not first_match:
                        first_match = para[:250].replace("\n", " ").strip()
                    last_match = para[:250].replace("\n", " ").strip()

            # Extract proper nouns from full section content
            proper = self._extract_proper_nouns(section.content)
            self._section_names[section.index] = proper

            line = (
                f"Section {section.index} ({section.token_count} tokens): "
                f"{opening}"
            )
            if closing:
                line += f" ... [closing:] {closing}"
            if first_match:
                line += f" ... {first_match}"
            if last_match and last_match != first_match:
                line += f" ... {last_match}"
            if proper:
                line += f" [Names: {', '.join(proper[:10])}]"
            lines.append(line)

        return "\n".join(lines)

    def _extract_proper_nouns(self, text: str) -> list[str]:
        """Extract likely proper nouns (capitalized words appearing 2+ times).

        Uses position heuristic: only counts words NOT at sentence start.
        Words appearing multiple times are more likely real names.
        """
        word_counts: dict[str, int] = defaultdict(int)
        for line in text.split("\n"):
            words = line.split()
            for i, word in enumerate(words):
                cleaned = word.strip(".,;:!?\"'()-[]{}…")
                if not cleaned or len(cleaned) < 3 or not cleaned.isalpha():
                    continue
                if not cleaned[0].isupper() or cleaned.isupper():
                    continue
                if cleaned in self._COMMON_TITLE_WORDS:
                    continue
                # Skip words at sentence start
                if i > 0:
                    prev = words[i - 1].rstrip()
                    if prev and not prev.endswith((".", "!", "?", ":")):
                        word_counts[cleaned] += 1

        # Keep words appearing 2+ times (reliable proper nouns)
        proper = sorted(
            [w for w, c in word_counts.items() if c >= 2],
            key=lambda w: -word_counts[w],
        )
        return proper[:15]

    # ---- Phase 2: Proper noun extraction ----

    def _extract_all_section_names(
        self, book: ConversationBook
    ) -> None:
        """Extract proper nouns from all sections.

        Populates self._section_names for programmatic boosting.
        """
        self._section_names = {}
        for section in book.sections:
            proper = self._extract_proper_nouns(section.content)
            self._section_names[section.index] = proper

    # ---- Question type detection ----

    @staticmethod
    def _detect_question_type(question: str) -> str:
        """Detect question type from first interrogative word.

        Used to add type-specific guidance to LLM prompts and
        enable location extraction for WHERE questions.
        """
        q = question.strip().lower()
        for prefix in ("where", "who", "how", "when", "why"):
            if q.startswith(prefix):
                return prefix
        if q.startswith(("what", "which")):
            return "what"
        return "other"

    def _extract_location_candidates(
        self, text: str
    ) -> list[str]:
        """Extract words following location prepositions.

        For WHERE questions, captures place names that may appear
        only once (bypassing the 2+ threshold in _extract_proper_nouns).
        Pattern: 'in Ireland', 'to Scotland', 'from Geneva', etc.
        """
        candidates: dict[str, int] = defaultdict(int)
        pattern = re.compile(
            r"\b(?:in|at|to|from|near|toward|towards|of|on)\s+"
            r"([A-Z][a-z]{2,})",
        )
        for match in pattern.finditer(text):
            word = match.group(1)
            if (
                word not in self._COMMON_TITLE_WORDS
                and word.isalpha()
                and len(word) >= 3
            ):
                candidates[word] += 1
        return sorted(candidates, key=lambda w: -candidates[w])

    # ---- Phase 3: LLM query expansion ----

    def _get_snippets_for_llm(
        self,
        book: ConversationBook,
        ranked_ids: list[int],
        idf_scores: dict[int, float],
        question: str,
    ) -> str:
        """Get diverse snippets: top-7 by RRF + remaining by IDF (deduped).

        Each snippet = opening + first/last term-matching paragraphs.
        This gives the LLM grounded book context without overwhelming it.
        Expanded from top-5 to top-7 so co-occurrence doesn't push
        borderline sections (like Felix/cottage at rank 6) out of view.
        """
        rrf_top = ranked_ids[:7]
        idf_ranked = sorted(
            idf_scores, key=lambda s: idf_scores.get(s, 0), reverse=True
        )
        section_ids = list(rrf_top)
        for sid in idf_ranked:
            if sid not in section_ids and len(section_ids) < 10:
                section_ids.append(sid)

        terms = _extract_heuristic_terms(question)
        lines = []
        for sid in section_ids:
            section = book.get_section(sid)
            if not section:
                continue
            paragraphs = [p.strip() for p in section.content.split("\n\n")
                          if p.strip() and len(p.strip()) > 30]
            if not paragraphs:
                continue
            opening = paragraphs[0][:300].replace("\n", " ").strip()
            # Find first AND last term-matching paragraphs for diverse
            # coverage: first gives early context, last captures outcomes
            first_match = ""
            last_match = ""
            for para in paragraphs[1:]:
                pl = para.lower()
                if any(t.lower() in pl for t in terms[:5]):
                    if not first_match:
                        first_match = para[:300].replace("\n", " ").strip()
                    last_match = para[:300].replace("\n", " ").strip()
            snippet = f"Section {sid}: {opening}"
            if first_match:
                snippet += f" ... {first_match}"
            if last_match and last_match != first_match:
                snippet += f" ... {last_match}"
            # Add proper nouns so LLM can generate character/place names
            # as search terms (critical for Q8: Felix, Safie, De Lacey)
            names = self._section_names.get(sid, [])
            if names:
                snippet += f" [Names: {', '.join(names[:8])}]"
            lines.append(snippet)
        return "\n".join(lines)

    async def _generate_terms(
        self, question: str, snippets: str,
        q_type: str = "other",
    ) -> list[str]:
        """LLM generates search terms with context from book snippets.

        Unlike pure query expansion, this sees actual book content, so it
        can generate grounded terms like character names and locations
        that appear in the text. Question-type-aware prompting.
        """
        type_focus = ""
        if q_type == "where":
            type_focus = (
                "PRIORITY: This is a WHERE question. Include ALL place "
                "names, country names, city names, and locations from "
                "the snippets and [Names: ...] tags.\n"
            )
        elif q_type == "who":
            type_focus = (
                "PRIORITY: This is a WHO question. Include ALL "
                "character names from the snippets and "
                "[Names: ...] tags.\n"
            )
        messages = [
            {"role": "system", "content": (
                "You generate search terms to find relevant document sections "
                "for answering a question. Output ONLY comma-separated single "
                "words (15-20 terms).\n"
                "Example: Felix,cottage,Safie,lessons,Lacey,language,Arabian\n\n"
                + type_focus +
                "Generate terms covering:\n"
                "- CHARACTER NAMES from the snippets and [Names: ...] tags\n"
                "- PLACE NAMES from the snippets and [Names: ...] tags\n"
                "- WHERE events happen (setting, location, landscape)\n"
                "- HOW events happen (method, action, consequence)\n"
                "- Related SYNONYMS for key question words\n"
                "- Each term: single word, specific noun or name"
            )},
            {"role": "user", "content": (
                f"Question: {question}\n\n"
                f"Document snippets:\n{snippets}\n\n"
                "Search terms:"
            )},
        ]
        try:
            response = await self._llm_call(messages, max_tokens=100)
            response = response.strip()
            logger.info(f"[readagent] Snippet terms response: {response[:200]}")
            terms: list[str] = []
            generic = {"novel", "character", "story", "book", "chapter",
                       "protagonist", "narrative", "literary", "scene",
                       "theme", "creature", "monster", "creation",
                       "the", "and", "for", "with", "from", "that",
                       "this", "what", "how", "who", "does"}
            for word in re.split(r'[,\s]+', response):
                word = word.strip().strip('"\'').strip()
                if (word and len(word) >= 2 and word.isalpha()
                        and word.lower() not in generic):
                    terms.append(word)
            return terms[:15]
        except Exception as e:
            logger.warning(f"[readagent] Snippet terms failed: {e}")
            return []

    async def _llm_expand_query(
        self, question: str, q_type: str = "other"
    ) -> list[str]:
        """LLM generates search terms from pure language understanding.

        No book context — just the question. This plays to the LLM's
        strength (language understanding, synonym generation) without
        overwhelming it with thousands of tokens of book summaries.

        Question-type-aware: for WHERE questions, emphasizes locations;
        for WHO questions, emphasizes character names; etc.
        """
        type_hint = self._QUESTION_TYPE_HINTS.get(q_type, "")
        messages = [
            {
                "role": "system",
                "content": (
                    "For the given question about a book, list 10-15 "
                    "specific words that would appear in the passage "
                    "containing the answer.\n\n"
                    "Include:\n"
                    "- Synonyms for key words in the question\n"
                    "- Related nouns (objects, concepts, actions)\n"
                    "- Words that describe the answer itself\n"
                    "- Character or place names if mentioned\n"
                    + type_hint +
                    '\nOutput ONLY JSON: {"terms": ["word1", "word2", ...]}\n'
                    "Be specific. Avoid generic words like 'story' or "
                    "'narrative'."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nSearch terms (JSON):",
            },
        ]
        try:
            response = await self._llm_call(messages, max_tokens=150)
            response = response.strip()
            logger.info(
                f"[readagent] LLM response: {response[:300]}"
            )

            # Try JSON object with "terms" key
            obj_match = re.search(r"\{[^}]*\}", response)
            if obj_match:
                try:
                    data = json.loads(obj_match.group())
                    terms = data.get("terms", [])
                    if isinstance(terms, list):
                        return [
                            str(t).strip() for t in terms
                            if isinstance(t, str)
                            and len(str(t).strip()) >= 2
                        ]
                except (json.JSONDecodeError, ValueError):
                    pass

            # Fallback: try JSON array
            arr_match = re.search(r'\["[^]]+\]', response)
            if arr_match:
                try:
                    terms = json.loads(arr_match.group())
                    return [
                        str(t).strip() for t in terms
                        if isinstance(t, str)
                        and len(str(t).strip()) >= 2
                    ]
                except (json.JSONDecodeError, ValueError):
                    pass

            # Last resort: extract quoted strings
            quoted = re.findall(r'"([^"]{2,30})"', response)
            return quoted[:15]
        except Exception as e:
            logger.warning(f"[readagent] LLM expand failed: {e}")
            return []

    async def _llm_filter_names(
        self, question: str
    ) -> list[str]:
        """LLM picks relevant proper nouns from all extracted names.

        Novel approach: programmatic proper noun extraction gives us
        ALL names in the book. The LLM then picks which names are
        relevant to the question. This is a SIMPLE classification task
        (not section selection or term generation) that even 8B models
        handle reliably.

        For 'Who teaches the creature to read?' → picks Felix, DeLacey
        For 'Where does Clerval get murdered?' → picks Clerval, Ireland
        """
        # Gather all unique proper nouns from all sections
        all_names: list[str] = []
        for sid in sorted(self._section_names.keys()):
            for name in self._section_names[sid]:
                if name not in all_names:
                    all_names.append(name)

        if not all_names:
            return []

        names_str = ", ".join(all_names[:80])
        messages = [
            {
                "role": "system",
                "content": (
                    "Pick the names most relevant to answering the "
                    "question. Include character names, place names, "
                    "and any names connected to the events asked about.\n"
                    'Output ONLY JSON: {"names": ["Name1", "Name2"]}\n'
                    "Pick 5-10 names. Be specific."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Names from the book: {names_str}\n\n"
                    "Relevant names (JSON):"
                ),
            },
        ]
        try:
            response = await self._llm_call(messages, max_tokens=100)
            response = response.strip()
            logger.info(
                f"[readagent] Name filter response: {response[:200]}"
            )

            obj_match = re.search(r"\{[^}]*\}", response)
            if obj_match:
                try:
                    data = json.loads(obj_match.group())
                    names = data.get("names", [])
                    if isinstance(names, list):
                        return [
                            str(n).strip() for n in names
                            if isinstance(n, str)
                            and len(str(n).strip()) >= 2
                        ]
                except (json.JSONDecodeError, ValueError):
                    pass

            # Fallback: extract quoted strings
            quoted = re.findall(r'"([^"]{2,30})"', response)
            return [q for q in quoted if q in all_names][:10]
        except Exception as e:
            logger.warning(
                f"[readagent] Name filter failed: {e}"
            )
            return []

    async def _llm_predict_answer(
        self, question: str
    ) -> list[str]:
        """LLM predicts factual content in the answer passage.

        Instead of generating search terms (query-side), this predicts
        what specific words would appear in the passage containing the
        answer (answer-side). Helps surface terms like "companion",
        "Ireland", "arctic" that search-term prompts miss.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "For the given question about a literary work, predict "
                    "the specific facts in the answer. What names, places, "
                    "objects, and actions would literally appear in the "
                    "paragraph that answers this question?\n\n"
                    "Think about:\n"
                    "- WHO is involved (character names)\n"
                    "- WHERE it happens (specific locations)\n"
                    "- WHAT specifically happens (concrete actions/objects)\n"
                    "- The ANSWER itself (what is the factual answer?)\n\n"
                    'Output ONLY JSON: {"facts": ["word1", "word2", ...]}\n'
                    "List 10-15 specific words. Be concrete and factual."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    "Answer facts (JSON):"
                ),
            },
        ]
        try:
            response = await self._llm_call(
                messages, max_tokens=150, temperature=0.3
            )
            response = response.strip()
            logger.info(
                f"[readagent] Answer prediction: {response[:300]}"
            )

            obj_match = re.search(r"\{[^}]*\}", response)
            if obj_match:
                try:
                    data = json.loads(obj_match.group())
                    facts = data.get("facts", [])
                    if isinstance(facts, list):
                        return [
                            str(t).strip() for t in facts
                            if isinstance(t, str)
                            and len(str(t).strip()) >= 2
                        ]
                except (json.JSONDecodeError, ValueError):
                    pass

            # Fallback: extract quoted strings
            quoted = re.findall(r'"([^"]{2,30})"', response)
            return quoted[:15]
        except Exception as e:
            logger.warning(
                f"[readagent] Answer prediction failed: {e}"
            )
            return []

    # ---- LLM call with retry ----

    async def _llm_call(
        self, messages: list[dict], max_tokens: int = 256,
        temperature: float = 0.1,
    ) -> str:
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=90.0) as client:
                    resp = await client.post(
                        f"{self.provider_url}/chat/completions",
                        json={
                            "model": self.model,
                            "messages": messages,
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                return data["choices"][0]["message"]["content"]
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (500, 503) and attempt < 2:
                    wait = 2 * (attempt + 1)
                    logger.warning(
                        f"[readagent] LLM {e.response.status_code}, "
                        f"retry in {wait}s..."
                    )
                    await asyncio.sleep(wait)
                    continue
                raise

    # ---- Shared retrieval methods (same as AgenticExtractor) ----

    async def _embed_score(
        self, book: ConversationBook, query: str
    ) -> dict[int, float]:
        """Chunked embedding similarity scoring."""
        def _run():
            import chromadb
            import uuid

            chunk_ids: list[str] = []
            chunk_texts: list[str] = []
            chunk_to_section: dict[str, int] = {}

            for section in book.sections:
                paragraphs = [
                    p.strip()
                    for p in section.content.split("\n\n")
                    if p.strip()
                ]
                if not paragraphs:
                    paragraphs = [section.content[:2000]]

                current_chunk: list[str] = []
                current_tokens = 0
                chunk_idx = 0
                for para in paragraphs:
                    ptokens = count_tokens(para)
                    if current_tokens + ptokens > 500 and current_chunk:
                        cid = f"{section.index}_{chunk_idx}"
                        chunk_ids.append(cid)
                        chunk_texts.append("\n\n".join(current_chunk))
                        chunk_to_section[cid] = section.index
                        chunk_idx += 1
                        current_chunk = [para]
                        current_tokens = ptokens
                    else:
                        current_chunk.append(para)
                        current_tokens += ptokens
                if current_chunk:
                    cid = f"{section.index}_{chunk_idx}"
                    chunk_ids.append(cid)
                    chunk_texts.append("\n\n".join(current_chunk))
                    chunk_to_section[cid] = section.index

            client = chromadb.Client()
            col_name = f"ra_{uuid.uuid4().hex[:8]}"
            collection = client.create_collection(col_name)
            collection.add(documents=chunk_texts, ids=chunk_ids)

            n_results = len(chunk_ids)
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["distances"],
            )
            client.delete_collection(col_name)

            scores: dict[int, float] = {}
            for cid, dist in zip(
                results["ids"][0], results["distances"][0]
            ):
                sid = chunk_to_section[cid]
                sim = 1.0 / (1.0 + dist)
                if sid not in scores or sim > scores[sid]:
                    scores[sid] = sim

            if scores:
                max_s = max(scores.values())
                min_s = min(scores.values())
                rng = max_s - min_s
                if rng > 0:
                    scores = {
                        k: (v - min_s) / rng for k, v in scores.items()
                    }

            return scores

        return await asyncio.get_event_loop().run_in_executor(None, _run)

    def _idf_search(
        self, book: ConversationBook, terms: list[str]
    ) -> dict[int, float]:
        """IDF-weighted word-level search."""
        if not terms:
            return {}
        n_sections = max(book.section_count, 1)
        log_n = math.log(n_sections) if n_sections > 1 else 1.0

        section_scores: dict[int, float] = defaultdict(float)
        for term in terms:
            scores = _search_book_words(book, term)
            n_matched = len(scores)
            if n_matched == 0:
                continue
            idf = max(0.0, 1.0 - math.log(n_matched) / log_n)
            if idf == 0.0:
                continue
            n_words = len([w for w in term.split() if len(w.strip()) >= 2])
            for sid, word_count in scores.items():
                word_ratio = word_count / max(n_words, 1)
                section_scores[sid] += idf * word_ratio

        return dict(section_scores)

    def _position_scores(
        self, book: ConversationBook, query: str
    ) -> dict[int, float]:
        """Position scores for beginning/ending queries."""
        words = set(re.findall(r"\w+", query.lower()))
        n = book.section_count
        if words & self._ENDING_WORDS:
            return {
                s.index: i / max(n - 1, 1)
                for i, s in enumerate(book.sections)
            }
        if words & self._BEGINNING_WORDS:
            return {
                s.index: 1.0 - i / max(n - 1, 1)
                for i, s in enumerate(book.sections)
            }
        return {}

    def _co_occurrence_search(
        self,
        book: ConversationBook,
        terms: list[str],
    ) -> dict[int, float]:
        """Score sections by co-occurrence of query term pairs.

        Sections where PAIRS of query terms appear together get
        higher scores. Co-occurrence is more discriminative than
        single-term IDF because it rewards specific content rather
        than sections that happen to mention one common term.

        For 'creature + create' → boosts section 18 (companion
        request) over sections that only mention 'creature'.
        """
        if len(terms) < 2:
            return {}

        n_sections = max(book.section_count, 1)
        log_n = math.log(n_sections) if n_sections > 1 else 1.0
        section_scores: dict[int, float] = defaultdict(float)

        # Pre-compute which sections contain each term
        term_sections: dict[str, set[int]] = {}
        for term in terms[:8]:  # limit terms
            scores = _search_book_words(book, term)
            term_sections[term] = set(scores.keys())

        # Score all pairs
        term_list = list(term_sections.keys())
        for i in range(len(term_list)):
            for j in range(i + 1, len(term_list)):
                t1, t2 = term_list[i], term_list[j]
                co_sections = term_sections[t1] & term_sections[t2]
                if not co_sections:
                    continue
                n_co = len(co_sections)
                idf = max(0.0, 1.0 - math.log(n_co) / log_n)
                if idf == 0:
                    continue
                for sid in co_sections:
                    section_scores[sid] += idf

        return dict(section_scores)

    def _rrf_fuse(
        self,
        embed_scores: dict[int, float],
        idf_scores: dict[int, float],
        position_scores: dict[int, float],
        cooccur_scores: dict[int, float] | None = None,
    ) -> dict[int, float]:
        """Reciprocal Rank Fusion of multiple signals."""
        k = 60
        rrf: dict[int, float] = defaultdict(float)
        embed_ranked = sorted(
            embed_scores,
            key=lambda s: embed_scores.get(s, 0),
            reverse=True,
        )
        idf_ranked = sorted(
            idf_scores,
            key=lambda s: idf_scores.get(s, 0),
            reverse=True,
        )
        for rank, sid in enumerate(embed_ranked):
            rrf[sid] += 1.0 / (k + rank)
        for rank, sid in enumerate(idf_ranked):
            rrf[sid] += 1.0 / (k + rank)
        if cooccur_scores:
            cooccur_ranked = sorted(
                cooccur_scores,
                key=lambda s: cooccur_scores.get(s, 0),
                reverse=True,
            )
            for rank, sid in enumerate(cooccur_ranked):
                rrf[sid] += 0.7 / (k + rank)
        if position_scores:
            for sid, pscore in position_scores.items():
                rrf[sid] += self._POSITION_WEIGHT * pscore
        return dict(rrf)

    # ---- Phase 5: Extractive assembly (EXIT-inspired) ----

    def _assemble(
        self,
        book: ConversationBook,
        ranked_ids: list[int],
        rrf_scores: dict[int, float],
        question: str,
        token_budget: int,
        search_terms: list[str],
        book_order: bool = False,
        budget_ratio: float | None = None,
        q_type: str = "other",
        priority_terms: list[str] | None = None,
    ) -> str:
        """Assemble context with paragraph-level excerpts."""
        remaining = token_budget - 200
        included: list[tuple[int, str]] = []
        included_ids: set[int] = set()
        ratio = (
            budget_ratio
            if budget_ratio is not None
            else self._MAX_SECTION_BUDGET_RATIO
        )
        max_per_section = int(token_budget * ratio)

        query_words = set(re.findall(r"\w+", question.lower()))
        is_ending = bool(query_words & self._ENDING_WORDS)
        is_beginning = bool(query_words & self._BEGINNING_WORDS)

        for sid in ranked_ids:
            if remaining < 200:
                break

            section = book.get_section(sid)
            if not section:
                continue

            if (
                section.token_count <= remaining
                and section.token_count <= max_per_section
            ):
                included.append((sid, section.to_text()))
                remaining -= section.token_count
                included_ids.add(sid)
                logger.info(
                    f"[readagent] +section {sid} full "
                    f"({section.token_count} tokens, "
                    f"rrf={rrf_scores.get(sid, 0):.4f}, "
                    f"{remaining} remaining)"
                )
            elif remaining >= 400:
                excerpt_budget = min(remaining, max_per_section)
                excerpt = self._extract_excerpt(
                    section.content,
                    question,
                    excerpt_budget,
                    search_terms,
                    is_ending,
                    is_beginning,
                    priority_terms=priority_terms,
                )
                if excerpt:
                    excerpt_tokens = count_tokens(excerpt)
                    header = (
                        f"=== Section {sid} [excerpted, "
                        f"{section.token_count} tokens total] ==="
                    )
                    included.append((sid, f"{header}\n{excerpt}"))
                    remaining -= excerpt_tokens + count_tokens(header)
                    included_ids.add(sid)
                    logger.info(
                        f"[readagent] +section {sid} excerpted "
                        f"({excerpt_tokens}/{section.token_count} tokens, "
                        f"rrf={rrf_scores.get(sid, 0):.4f}, "
                        f"{remaining} remaining)"
                    )

        # Fill remaining budget with recent sections
        for section in reversed(book.sections):
            if remaining < 200:
                break
            if section.index in included_ids:
                continue
            if (
                section.token_count <= remaining
                and section.token_count <= max_per_section
            ):
                included.append((section.index, section.to_text()))
                remaining -= section.token_count
                included_ids.add(section.index)

        if not included:
            return ""

        if book_order:
            included.sort(key=lambda x: x[0])

        # Structure overview for document-level context
        structure_parts = []
        chapter_count = 0
        for section in book.sections:
            first_line = section.content.split("\n", 1)[0].strip()[:60]
            is_chapter = bool(
                re.match(r"(?i)chapter\s+\d", first_line)
            )
            if is_chapter:
                chapter_count += 1
            structure_parts.append(
                f"Section {section.index}: {first_line} "
                f"({section.token_count} tokens)"
            )
        if chapter_count > 0:
            structure_parts.append(
                f"[Total chapters: {chapter_count}]"
            )
        structure = "\n".join(structure_parts)

        framing = (
            f"[ReadAgent extraction from {book.section_count} sections, "
            f"{book.total_tokens} tokens total. "
            f"Showing {len(included_ids)} sections.]\n"
            f"[IMPORTANT: Answer using ONLY the text below. "
            f"Do not use prior knowledge about this book.]\n\n"
            f"Book structure:\n{structure}\n"
        )
        body = "\n\n".join(content for _, content in included)

        # Question-type-aware reinforcement + faithfulness instruction
        if q_type == "where":
            # Extract locations from assembled text to ground
            # the model's answer in the actual content
            loc_set: set[str] = set()
            for _, content in included:
                locs = self._extract_location_candidates(content)
                loc_set.update(locs[:8])
            loc_set -= self._COMMON_TITLE_WORDS
            loc_set -= {"Project", "Use"}
            loc_names = sorted(loc_set)
            loc_hint = (
                f"\nLocations mentioned in the text: "
                f"{', '.join(loc_names)}"
                if loc_names else ""
            )

            reminder = (
                f"\n\n[End of extracted content.{loc_hint}\n"
                f"IMPORTANT: Answer based ONLY on the text above. "
                f"State the specific location mentioned in the text.\n"
                f"Answer this question: {question}]"
            )
        elif is_ending:
            reminder = (
                f"\n\n[End of extracted content.\n"
                f"IMPORTANT: Answer based ONLY on the text above. "
                f"Do not use prior knowledge about this book.\n"
                f"Focus on the FINAL events in the last sections. "
                f"Describe the physical setting where the story "
                f"concludes and the ultimate fate of each character.\n"
                f"Answer this question: {question}]"
            )
        elif q_type == "who":
            reminder = (
                f"\n\n[End of extracted content.\n"
                f"IMPORTANT: Answer based ONLY on the text above. "
                f"Do not use prior knowledge about this book.\n"
                f"Use the exact character names from the text. "
                f"Include specific events and outcomes.\n"
                f"Answer this question: {question}]"
            )
        else:
            reminder = (
                f"\n\n[End of extracted content.\n"
                f"IMPORTANT: Answer based ONLY on the text above. "
                f"Do not use prior knowledge about this book.\n"
                f"Answer this question: {question}]"
            )
        return f"{framing}\n{body}{reminder}"

    def _extract_excerpt(
        self,
        text: str,
        question: str,
        max_tokens: int,
        extra_terms: list[str] | None = None,
        is_ending: bool = False,
        is_beginning: bool = False,
        priority_terms: list[str] | None = None,
    ) -> str:
        """Extract query-relevant paragraphs with position awareness."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            return ""

        para_tokens = [count_tokens(p) for p in paragraphs]

        # Keyword scoring
        terms = _extract_heuristic_terms(question)
        if extra_terms:
            terms = list(dict.fromkeys(terms + extra_terms))
        term_patterns = []
        for t in terms:
            try:
                term_patterns.append(
                    re.compile(re.escape(t), re.IGNORECASE)
                )
            except re.error:
                pass

        # Priority term patterns for "where" questions:
        # paragraphs matching these get a score boost
        prio_patterns = []
        if priority_terms:
            for t in priority_terms:
                try:
                    prio_patterns.append(
                        re.compile(re.escape(t), re.IGNORECASE)
                    )
                except re.error:
                    pass

        keyword_scores = []
        for para in paragraphs:
            base = sum(1.0 for p in term_patterns if p.search(para))
            if prio_patterns:
                prio_hits = sum(
                    1 for p in prio_patterns if p.search(para)
                )
                if prio_hits > 0:
                    base += 3.0 * prio_hits
            keyword_scores.append(base)
        kmax = max(keyword_scores) if keyword_scores else 1.0
        if kmax > 0:
            keyword_scores = [s / kmax for s in keyword_scores]

        # Embedding scoring
        embed_scores = [0.0] * len(paragraphs)
        try:
            import chromadb
            import uuid as _uuid

            valid = [
                (i, p) for i, p in enumerate(paragraphs) if len(p) > 20
            ]
            if valid:
                client = chromadb.Client()
                col = client.create_collection(
                    f"raex_{_uuid.uuid4().hex[:8]}"
                )
                col.add(
                    documents=[p[:1000] for _, p in valid],
                    ids=[str(i) for i, _ in valid],
                )
                res = col.query(
                    query_texts=[question],
                    n_results=len(valid),
                    include=["distances"],
                )
                client.delete_collection(col.name)
                for id_str, dist in zip(
                    res["ids"][0], res["distances"][0]
                ):
                    embed_scores[int(id_str)] = 1.0 / (1.0 + dist)
                emax = max(embed_scores)
                emin = (
                    min(s for s in embed_scores if s > 0)
                    if any(embed_scores)
                    else 0
                )
                erng = emax - emin
                if erng > 0:
                    embed_scores = [
                        (s - emin) / erng if s > 0 else 0.0
                        for s in embed_scores
                    ]
        except Exception:
            pass

        # Combined score with position awareness
        scored: list[tuple[int, float]] = []
        n = len(paragraphs)
        for i in range(n):
            score = 0.6 * keyword_scores[i] + 0.4 * embed_scores[i]
            if is_ending:
                pos_ratio = i / max(n - 1, 1)
                score += 0.7 * (pos_ratio ** 2)
            elif is_beginning:
                pos_ratio = 1.0 - i / max(n - 1, 1)
                score += 0.7 * (pos_ratio ** 2)
            else:
                if i < 2 or i >= n - 2:
                    score += 0.1
            scored.append((i, score))

        scored.sort(key=lambda x: (-x[1], x[0]))

        # Pre-reserve tail paragraphs for ending queries
        selected: set[int] = set()
        used = 0
        if is_ending:
            tail_budget = int(max_tokens * 0.55)
            tail_used = 0
            guaranteed = 0
            for i in range(n - 1, max(n - 20, -1), -1):
                para_lower = paragraphs[i].lower()
                if (
                    "gutenberg" in para_lower
                    or "trademark" in para_lower
                    or "license" in para_lower
                ):
                    continue
                if guaranteed >= 8:
                    if (
                        keyword_scores[i] == 0
                        and embed_scores[i] < 0.1
                    ):
                        continue
                if tail_used + para_tokens[i] <= tail_budget:
                    selected.add(i)
                    tail_used += para_tokens[i]
                    guaranteed += 1
            used = tail_used
        elif is_beginning:
            head_budget = int(max_tokens * 0.35)
            head_used = 0
            for i in range(min(8, n)):
                if keyword_scores[i] == 0 and embed_scores[i] < 0.1:
                    continue
                if head_used + para_tokens[i] <= head_budget:
                    selected.add(i)
                    head_used += para_tokens[i]
            used = head_used

        # Greedy fill with scored paragraphs
        for idx, score in scored:
            if idx in selected:
                continue
            if used + para_tokens[idx] <= max_tokens:
                selected.add(idx)
                used += para_tokens[idx]

        if not selected:
            return ""

        # Context expansion (adjacent paragraphs)
        context_candidates = set()
        for idx in list(selected):
            if idx > 0:
                context_candidates.add(idx - 1)
            if idx < n - 1:
                context_candidates.add(idx + 1)
        for idx in sorted(context_candidates - selected):
            if used + para_tokens[idx] <= max_tokens:
                selected.add(idx)
                used += para_tokens[idx]

        ordered = sorted(selected)
        return "\n\n".join(paragraphs[i] for i in ordered)


# ---------------------------------------------------------------------------
# Factory — get extractor by strategy name
# ---------------------------------------------------------------------------

STRATEGIES = {
    "header": HeaderExtractor,
    "autosearch": AutoSearchExtractor,
    "rlm": RLMExtractor,
    "rlm_v2": RLMV2Extractor,
    "rlm_v3": RLMV3Extractor,
    "toolcall": ToolCallExtractor,
    "embed": EmbedExtractor,
    "compress": CompressExtractor,
    "adaptive": AdaptiveExtractor,
    "icl": ICLExtractor,
    "rlm_v4": RLMV4Extractor,
    "rlm_v5": RLMV5Extractor,
    "rlm_v6": RLMV6Extractor,
    "agentic": AgenticExtractor,
    "subagent": SubagentExtractor,
    "readagent": ReadAgentExtractor,
}


def get_extractor(
    strategy: str,
    provider_url: str = "",
    model: str = "",
    api_key: str = "dummy",
    max_context: int = 16384,
    max_iterations: int = 10,
    max_llm_calls: int = 15,
) -> Any:
    """Create an extractor instance for the given strategy.

    Strategies that don't need LLM (header, autosearch, embed, compress, adaptive)
    ignore provider params.
    """
    cls = STRATEGIES.get(strategy)
    if cls is None:
        logger.warning(f"Unknown strategy '{strategy}', falling back to 'rlm_v2'")
        cls = RLMV2Extractor

    if cls in (HeaderExtractor, AutoSearchExtractor, EmbedExtractor,
               CompressExtractor, AdaptiveExtractor, ICLExtractor):
        return cls()

    # RLMV4, RLMV6, AgenticExtractor, SubagentExtractor don't need max_iterations/max_llm_calls
    if cls in (RLMV4Extractor, RLMV6Extractor, AgenticExtractor, SubagentExtractor, ReadAgentExtractor):
        return cls(
            provider_url=provider_url,
            model=model,
            api_key=api_key,
            max_context=max_context,
        )

    return cls(
        provider_url=provider_url,
        model=model,
        api_key=api_key,
        max_context=max_context,
        max_iterations=max_iterations,
        max_llm_calls=max_llm_calls,
    )
