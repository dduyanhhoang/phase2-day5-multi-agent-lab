#!/usr/bin/env python3
"""Integration test - requires valid OPENAI_API_KEY in environment."""

import os
import sys

def test_live_api():
    """Test with real OpenAI API."""
    from multi_agent_research_lab.services.llm_client import LLMClient

    if not os.getenv("OPENAI_API_KEY"):
        print("SKIPPED: OPENAI_API_KEY not set")
        return

    print("Testing with live OpenAI API...")
    client = LLMClient()
    response = client.complete(
        system_prompt="You are a concise assistant.",
        user_prompt="Reply with exactly: 'OK'"
    )
    assert "OK" in response.content, f"Unexpected response: {response.content}"
    assert response.input_tokens is not None
    assert response.output_tokens is not None
    assert response.cost_usd is not None
    print(f"✓ Live API test passed (cost: ${response.cost_usd:.6f})")


def test_model_override():
    """Test using a different model via env var."""
    import os
    from multi_agent_research_lab.services.llm_client import LLMClient
    from multi_agent_research_lab.core.config import Settings

    os.environ["OPENAI_MODEL"] = "gpt-4o"

    # Reload settings to pick up env change
    from importlib import reload
    from multi_agent_research_lab.core import config as cfg
    reload(cfg)

    new_settings = cfg.Settings()  # Will use updated env
    assert new_settings.openai_model == "gpt-4o"
    print("✓ Model override from env works")


if __name__ == "__main__":
    test_live_api()
    test_model_override()
    print("\n✓ All integration tests passed!")
