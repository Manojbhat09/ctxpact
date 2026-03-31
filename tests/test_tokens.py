"""Tests for token counting utilities."""

from ctxpact.compaction.tokens import count_message_tokens, count_messages_tokens, count_tokens


class TestTokenCounting:
    def test_count_tokens_basic(self):
        tokens = count_tokens("Hello, world!")
        assert tokens > 0
        assert tokens < 20

    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_message_tokens_includes_overhead(self):
        msg = {"role": "user", "content": "Hello"}
        tokens = count_message_tokens(msg)
        # Should be more than just "Hello" due to framing
        assert tokens > count_tokens("Hello")

    def test_tool_call_tokens(self):
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": '{"query": "test"}',
                    },
                }
            ],
        }
        tokens = count_message_tokens(msg)
        assert tokens >= 10  # Should account for tool call content

    def test_messages_total(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        total = count_messages_tokens(messages)
        assert total > 0
        # Should be sum of individual + conversation overhead
        individual_sum = sum(count_message_tokens(m) for m in messages)
        assert total >= individual_sum
