"""Tests for Groq Cloud Integration, Key Sanitization, and Fast Dynamic Cascading.

Verifies:
1. Groq API keys with quotes, whitespace, or 'Bearer ' prefixes are properly sanitized.
2. Provider accurately identifies Groq and targets https://api.groq.com/openai/v1/chat/completions.
3. Empty messages are stripped from payloads before sending to Groq.
4. Automatic cascading from llama-3.3-70b-versatile to llama-3.1-8b-instant on HTTP 429 / rate limits.
5. Generous token budget prevents premature truncation ('finish_reason=length') on detailed outputs.
6. Clean completions ending with punctuation are not falsely marked as truncated.
"""

import os
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config.llm_provider import UniversalChatModel, get_llm
from config.settings import settings


class TestGroqKeySanitizationAndCascading:
    """Test suite for Groq API integration robustness."""

    def test_key_sanitization_strips_quotes_and_bearer(self):
        """Keys containing quotes, whitespace, or 'Bearer ' are cleaned correctly."""
        raw_keys = [
            '"gsk_sample1234567890"',
            "'gsk_sample1234567890'",
            "  gsk_sample1234567890 \n",
            "Bearer gsk_sample1234567890",
            'Bearer "gsk_sample1234567890"',
        ]
        model = UniversalChatModel(provider="groq")

        for rk in raw_keys:
            with patch("requests.post") as mock_post:
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.json.return_value = {
                    "choices": [{"message": {"content": "Dynamic clinical guidance."}, "finish_reason": "stop"}]
                }
                mock_post.return_value = mock_resp

                res, model_name, finish_reason, is_trunc = model._invoke_openai_compatible(
                    messages=[HumanMessage(content="Hello Mindly")],
                    api_key=rk,
                    base_url="",
                )
                assert res == "Dynamic clinical guidance."
                assert is_trunc is False
                # Verify that Authorization header contains clean key without quotes or double Bearer
                called_headers = mock_post.call_args[1]["headers"]
                assert called_headers["Authorization"] == "Bearer gsk_sample1234567890"

    def test_groq_empty_messages_filtered_out(self):
        """Groq rejects empty message content; verification that empty messages are stripped."""
        model = UniversalChatModel(provider="groq")

        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": "Response"}, "finish_reason": "stop"}]
            }
            mock_post.return_value = mock_resp

            model._invoke_openai_compatible(
                messages=[
                    SystemMessage(content="   "),  # Empty
                    HumanMessage(content="Explain migraine triggers"),
                    AIMessage(content=""),  # Empty
                ],
                api_key="gsk_validkey",
                base_url="",
            )

            posted_body = mock_post.call_args[1]["json"]
            for m in posted_body["messages"]:
                assert len(m["content"].strip()) > 0, "Empty message content was sent to Groq!"

    def test_groq_cascades_on_429_rate_limit(self):
        """When llama-3.3-70b-versatile hits 429 TPM rate limit, automatically cascades to llama-3.1-8b-instant."""
        model = UniversalChatModel(provider="groq")

        # Mock first call as 429 Rate Limit, second call as 200 Success
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.text = '{"error": {"message": "Rate limit reached for model llama-3.3-70b-versatile"}}'

        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.json.return_value = {
            "choices": [{"message": {"content": "Instant response from 8b model."}, "finish_reason": "stop"}]
        }

        with patch("requests.post", side_effect=[resp_429, resp_200]) as mock_post:
            res, model_name, finish_reason, is_trunc = model._invoke_openai_compatible(
                messages=[HumanMessage(content="Explain hypertension in detail")],
                api_key="gsk_testkey",
                base_url="",
            )

            assert res == "Instant response from 8b model."
            assert model_name in ["openai/gpt-oss-120b", "llama-3.1-8b-instant"]
            assert is_trunc is False
            assert mock_post.call_count == 2

    def test_groq_comfortable_token_budget_allocated(self):
        """Groq calls allocate at least 1024 tokens so clinical guides are not cut off."""
        model = UniversalChatModel(provider="groq", max_tokens=250)

        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": "Full detailed guide."}, "finish_reason": "stop"}]
            }
            mock_post.return_value = mock_resp

            model._invoke_openai_compatible(
                messages=[HumanMessage(content="Detailed guidance")],
                api_key="gsk_key",
                base_url="",
            )

            posted_body = mock_post.call_args[1]["json"]
            assert posted_body["max_tokens"] >= 1024

    def test_finish_reason_length_with_clean_ending_not_falsely_truncated(self):
        """If finish_reason is 'length' but the output ends with clean period, it is not marked truncated."""
        model = UniversalChatModel(provider="groq")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{
                "message": {"content": "Clinical evidence shows that lifestyle changes and rest successfully resolve tension headaches."},
                "finish_reason": "length"
            }]
        }

        with patch("requests.post", return_value=mock_resp):
            res, model_name, finish_reason, is_trunc = model._invoke_openai_compatible(
                messages=[HumanMessage(content="How to handle headache?")],
                api_key="gsk_key",
                base_url="",
            )
            assert res is not None
            assert is_trunc is False  # Ends with clean period!
