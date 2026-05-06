"""Tests for LLMClient."""

import os
from unittest.mock import MagicMock, patch

import pytest

from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.services.llm_client import LLMClient, LLMResponse, _calculate_cost


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_creation(self) -> None:
        """Test LLMResponse can be created with required fields."""
        resp = LLMResponse(content="Hello")
        assert resp.content == "Hello"
        assert resp.input_tokens is None
        assert resp.output_tokens is None
        assert resp.cost_usd is None

    def test_with_metadata(self) -> None:
        """Test LLMResponse with all metadata."""
        resp = LLMResponse(
            content="Hello",
            input_tokens=10,
            output_tokens=20,
            cost_usd=0.003,
        )
        assert resp.content == "Hello"
        assert resp.input_tokens == 10
        assert resp.output_tokens == 20
        assert resp.cost_usd == 0.003


class TestCalculateCost:
    """Tests for _calculate_cost function."""

    def test_gpt4o_mini_pricing(self) -> None:
        """Test cost calculation for gpt-4o-mini."""
        # 1000 input tokens = $0.00015, 1000 output tokens = $0.0006
        cost = _calculate_cost("gpt-4o-mini", 1000, 1000)
        assert cost == 0.00075  # 0.00015 + 0.0006

    def test_partial_tokens(self) -> None:
        """Test cost calculation for partial tokens."""
        cost = _calculate_cost("gpt-4o-mini", 100, 50)
        expected = (100 / 1000) * 0.00015 + (50 / 1000) * 0.0006
        assert cost == round(expected, 6)

    def test_unknown_model(self) -> None:
        """Test unknown model returns None."""
        cost = _calculate_cost("unknown-model", 100, 50)
        assert cost is None


class TestLLMClient:
    """Tests for LLMClient."""

    @pytest.fixture
    def mock_settings(self) -> Settings:
        """Create mock settings for testing."""
        return Settings(
            app_env="test",
            log_level="DEBUG",
            openai_api_key="test-key",
            openai_model="gpt-4o-mini",
            timeout_seconds=30,
        )

    @patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"})
    def test_init_with_env_var(self, mock_settings: Settings) -> None:
        """Test client initialization uses environment variable."""
        with patch.object(Settings, "__new__", return_value=mock_settings):
            client = LLMClient()
            assert client._model == "gpt-4o-mini"

    def test_init_requires_api_key(self) -> None:
        """Test client raises error without API key."""
        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(StudentTodoError, match="API key not configured"),
        ):
            LLMClient()

    @patch("multi_agent_research_lab.services.llm_client.OpenAI")
    def test_complete_success(self, mock_openai: MagicMock, mock_settings: Settings) -> None:
        """Test successful completion."""
        # Mock response
        mock_usage = MagicMock(prompt_tokens=10, completion_tokens=20)
        mock_message = MagicMock(content="Hello, world!")
        mock_choice = MagicMock(message=mock_message)
        mock_response = MagicMock(choices=[mock_choice], usage=mock_usage)

        mock_openai.return_value.chat.completions.create.return_value = mock_response

        with patch.object(Settings, "__new__", return_value=mock_settings):
            client = LLMClient()
            result = client.complete("You are helpful", "Say hello")

        assert result.content == "Hello, world!"
        assert result.input_tokens == 10
        assert result.output_tokens == 20
        assert result.cost_usd is not None

    def test_complete_requires_prompts(self, mock_settings: Settings) -> None:
        """Test that empty prompts raise ValueError."""
        with patch.object(Settings, "__new__", return_value=mock_settings):
            client = LLMClient()
            with pytest.raises(ValueError, match="required"):
                client.complete("", "test")
            with pytest.raises(ValueError, match="required"):
                client.complete("test", "")
