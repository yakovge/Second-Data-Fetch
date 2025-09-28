#!/usr/bin/env python3
"""
Test the complete AI workflow: Raw text -> AI discovers URLs -> AI infers structure -> Results
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__)))

from src.core.ai_orchestrator import AIOrchestrator
from dotenv import load_dotenv

def test_ai_workflow():
    """Test the complete AI-driven workflow with raw text input."""
    print("Testing Complete AI-Driven Workflow")
    print("=" * 50)

    # Load environment variables
    load_dotenv(override=True)

    # Test scenarios - these are raw text inputs that should trigger full AI workflow
    test_scenarios = [
        "I want to find recent news about artificial intelligence developments",
        "Get me information about climate change policies from news sources",
        "Find articles about cryptocurrency market trends",
        "Search for technology startup funding news"
    ]

    print("\nThis test demonstrates the complete AI workflow:")
    print("1. Raw Text Input")
    print("2. AI URL Discovery (Claude generates relevant URLs)")
    print("3. AI Structure Inference (Claude analyzes fetched data)")
    print("4. AI Implementation Generation (Claude creates custom DataFetch class)")
    print("5. Dynamic Execution (Generated code runs automatically)")
    print("\n" + "=" * 50)

    try:
        # Create orchestrator (will try Claude first, fallback to Gemini)
        orchestrator = AIOrchestrator()

        if orchestrator.ai_client:
            client_type = type(orchestrator.ai_client).__name__
            print(f"AI Client: {client_type}")

            # Test each scenario
            for i, scenario in enumerate(test_scenarios, 1):
                print(f"\n--- Test Scenario {i} ---")
                print(f"Raw Text: '{scenario}'")

                try:
                    # This should trigger the complete AI workflow
                    result = orchestrator.orchestrate_fetch(scenario)

                    if result.error:
                        print(f"Status: Completed with fallback")
                        print(f"Error: {result.error}")
                    else:
                        print(f"Status: SUCCESS!")
                        print(f"URL: {result.url}")
                        print(f"Data Type: {type(result.data)}")
                        print(f"Format: {result.format}")
                        print(f"Method: {result.method}")

                        # Show metadata about AI operations
                        if result.metadata:
                            ai_ops = [key for key in result.metadata.keys() if 'ai_' in key.lower() or 'generated' in key.lower()]
                            if ai_ops:
                                print(f"AI Operations: {ai_ops}")

                        # Show brief data sample
                        if isinstance(result.data, dict):
                            print(f"Data Keys: {list(result.data.keys())[:5]}...")
                        elif isinstance(result.data, str):
                            print(f"Data Preview: {result.data[:100]}...")

                except Exception as e:
                    print(f"Status: FAILED")
                    print(f"Error: {str(e)}")

                print("-" * 30)

        else:
            print("No AI client available - check your API keys")
            print("\nTo enable full AI workflow:")
            print("1. Add valid CLAUDE_API_KEY to .env file")
            print("2. Or add valid GEMINI_API_KEY to .env file")

    except Exception as e:
        print(f"Workflow test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_ai_workflow()