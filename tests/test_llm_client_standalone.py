#!/usr/bin/env python3
"""Standalone test for LLMClient - no pytest dependency."""

import sys
from unittest.mock import patch, MagicMock

# Mock the openai module before import
mock_openai_module = MagicMock()
mock_client = MagicMock()
mock_openai_module.OpenAI.return_value = mock_client

# Setup mock response
mock_usage = MagicMock(prompt_tokens=15, completion_tokens=25)
mock_message = MagicMock(content="Test response from OpenAI")
mock_choice = MagicMock(message=mock_message)
mock_response = MagicMock(choices=[mock_choice], usage=mock_usage)
mock_client.chat.completions.create.return_value = mock_response

with patch.dict(sys.modules, {'openai': mock_openai_module}):
    # Now import after mocking
    from multi_agent_research_lab.services.llm_client import LLMClient, LLMResponse, _calculate_cost

def test_calculate_cost():
    """Test cost calculation for various models."""
    print("Testing _calculate_cost...")

    # gpt-4o-mini: $0.00015/1K input, $0.0006/1K output
    cost = _calculate_cost("gpt-4o-mini", 1000, 1000)
    assert cost == 0.00075, f"Expected 0.00075, got {cost}"
    print(f"  ✓ gpt-4o-mini 1000/1000 tokens: ${cost:.6f}")

    # gpt-4o: $0.005/1K input, $0.015/1K output
    cost = _calculate_cost("gpt-4o", 1000, 1000)
    assert cost == 0.020, f"Expected 0.020, got {cost}"
    print(f"  ✓ gpt-4o 1000/1000 tokens: ${cost:.6f}")

    # Partial tokens
    cost = _calculate_cost("gpt-4o-mini", 100, 50)
    expected = (100 / 1000) * 0.00015 + (50 / 1000) * 0.0006
    assert cost == round(expected, 6), f"Expected {expected}, got {cost}"
    print(f"  ✓ gpt-4o-mini 100/50 tokens: ${cost:.6f}")

    # Unknown model
    cost = _calculate_cost("unknown-model", 100, 50)
    assert cost is None, f"Expected None, got {cost}"
    print("  ✓ Unknown model returns None")

    print("✓ All _calculate_cost tests passed!\n")


def test_llm_response():
    """Test LLMResponse dataclass."""
    print("Testing LLMResponse...")

    resp = LLMResponse(content="Hello")
    assert resp.content == "Hello"
    assert resp.input_tokens is None
    assert resp.output_tokens is None
    assert resp.cost_usd is None
    print("  ✓ Default values work")

    resp = LLMResponse(
        content="Hello, world!",
        input_tokens=10,
        output_tokens=20,
        cost_usd=0.003,
    )
    assert resp.content == "Hello, world!"
    assert resp.input_tokens == 10
    assert resp.output_tokens == 20
    assert resp.cost_usd == 0.003
    print("  ✓ All metadata preserved")

    print("✓ All LLMResponse tests passed!\n")


def test_client_empty_prompts():
    """Test that empty prompts raise ValueError."""
    print("Testing empty prompt validation...")

    with patch.dict(sys.modules, {'openai': mock_openai_module}):
        # Need fresh import to test init
        from multi_agent_research_lab.services.llm_client import LLMClient as _TestClient
        # Manually set a valid settings mock
        from multi_agent_research_lab.core.config import Settings
        test_settings = Settings(
            app_env="test",
            openai_api_key="test-key",
            openai_model="gpt-4o-mini",
            timeout_seconds=30,
        )
        with patch.object(Settings, "__new__", return_value=test_settings):
            client = _TestClient()
            try:
                client.complete("", "test")
                print("  ✗ Should have raised for empty system")
                sys.exit(1)
            except ValueError as e:
                assert "required" in str(e)
                print("  ✓ Empty system prompt raises ValueError")

            try:
                client.complete("test", "")
                print("  ✗ Should have raised for empty user")
                sys.exit(1)
            except ValueError as e:
                assert "required" in str(e)
                print("  ✓ Empty user prompt raises ValueError")

    print("✓ All validation tests passed!\n")


def test_client_success():
    """Test successful completion with mocked OpenAI."""
    print("Testing LLMClient.complete success case...")

    with patch.dict(sys.modules, {'openai': mock_openai_module}):
        from multi_agent_research_lab.services.llm_client import LLMClient as _TestClient
        from multi_agent_research_lab.core.config import Settings

        test_settings = Settings(
            app_env="test",
            openai_api_key="test-key",
            openai_model="gpt-4o-mini",
            timeout_seconds=30,
        )
        with patch.object(Settings, "__new__", return_value=test_settings):
            client = _TestClient()
            result = client.complete("You are helpful", "Say hello")

            assert result.content == "Test response from OpenAI"
            assert result.input_tokens == 15
            assert result.output_tokens == 25
            assert result.cost_usd > 0
            print(f"  ✓ Response content: '{result.content}'")
            print(f"  ✓ Tokens: {result.input_tokens} in, {result.output_tokens} out")
            print(f"  ✓ Cost: ${result.cost_usd:.6f}")

    print("✓ All client integration tests passed!\n")


def test_client_retry():
    """Test that retry logic is configured."""
    print("Testing retry configuration...")

    with patch.dict(sys.modules, {'openai': mock_openai_module}):
        from multi_agent_research_lab.services.llm_client import LLMClient as _TestClient
        from multi_agent_research_lab.core.config import Settings
        from tenacity import retry

        test_settings = Settings(
            app_env="test",
            openai_api_key="test-key",
            openai_model="gpt-4o-mini",
            timeout_seconds=30,
        )
        with patch.object(Settings, "__new__", return_value=test_settings):
            client = _TestClient()
            # Check that complete method has retry decorator
            assert hasattr(client.complete, '__wrapped__'), "complete should have retry wrapper"
            print("  ✓ complete() has retry decorator")

    print("✓ Retry configuration test passed!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("LLMClient Test Suite")
    print("=" * 60 + "\n")

    try:
        test_calculate_cost()
        test_llm_response()
        test_client_empty_prompts()
        test_client_success()
        test_client_retry()

        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
