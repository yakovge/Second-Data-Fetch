#!/usr/bin/env python3
"""
Test Claude Haiku integration
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__)))

from src.core.ai_orchestrator import AIOrchestrator
from src.ai.claude_client import ClaudeClient
from dotenv import load_dotenv

def test_claude_integration():
    """Test Claude integration with the orchestrator."""
    print("Testing Claude Haiku Integration")
    print("=" * 40)

    # Load environment variables
    load_dotenv(override=True)

    # Check for Claude API key
    claude_key = os.getenv('CLAUDE_API_KEY') or os.getenv('ANTHROPIC_API_KEY')
    print(f"Claude API key found: {'Yes' if claude_key else 'No'}")

    if not claude_key:
        print("\nTo test Claude integration:")
        print("1. Get API key from https://console.anthropic.com")
        print("2. Add to .env file as CLAUDE_API_KEY=your_key_here")
        print("3. Uncomment the line in .env file")
        print("\nCurrently testing fallback behavior...")

    try:
        # Test 1: Create orchestrator (should try Claude first, fallback to Gemini)
        print("\n1. Creating AI orchestrator...")
        orchestrator = AIOrchestrator()

        if orchestrator.ai_client:
            client_type = type(orchestrator.ai_client).__name__
            print(f"AI client created: {client_type}")

            if client_type == "ClaudeClient":
                print("[SUCCESS] Successfully using Claude Haiku!")

                # Test basic functionality
                print("\n2. Testing basic Claude functionality...")
                sample_data = {"title": "Test Article", "content": "Test content"}
                structure = orchestrator.ai_client.generate_structure_from_sample(
                    sample_data, "Test context"
                )
                print(f"Structure generation: {'SUCCESS' if structure else 'FAILED'}")

            elif client_type == "GeminiClient":
                print("[WARNING] Fell back to Gemini (but Gemini key is invalid)")
                print("   Add valid CLAUDE_API_KEY to use Claude")
            else:
                print(f"[WARNING] Unexpected client type: {client_type}")
        else:
            print("[ERROR] No AI client could be created")
            print("   Both Claude and Gemini API keys are missing/invalid")

        # Test 3: Full orchestration workflow
        print("\n3. Testing orchestration workflow...")
        try:
            result = orchestrator.orchestrate_fetch("Latest news about technology")
            if result.error:
                print(f"Orchestration completed with fallback (no AI): {result.error}")
            else:
                print("[SUCCESS] Orchestration successful!")
        except Exception as e:
            print(f"Orchestration failed: {str(e)}")

    except Exception as e:
        print(f"Integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_claude_integration()