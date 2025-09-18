#!/usr/bin/env python3
"""
AI-Driven DataFetch CLI - Uses the AI orchestration workflow.

This CLI demonstrates the complete AI-driven pipeline:
Raw Text → AI URL Generation → AI Implementation Generation → Dynamic Execution

Usage: python ai_driven_cli.py "Your data request"
"""

import sys
import os
import json
import logging
from typing import Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.ai_orchestrator import AIOrchestrator, AIDataFetchFactory
from src.ai.gemini_client import GeminiClient


def setup_logging():
    """Setup logging for better visibility into the AI workflow."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def fetch_with_ai(raw_text: str, verbose: bool = True) -> Optional[dict]:
    """
    Fetch data using the complete AI orchestration workflow.

    Args:
        raw_text: User's data request
        verbose: Whether to show detailed output

    Returns:
        Dictionary with results or None if failed
    """
    if verbose:
        print("AI-Driven DataFetch System")
        print("=" * 50)
        print(f"Request: {raw_text}")
        print("-" * 50)

    try:
        # Initialize AI orchestrator
        print("Initializing AI orchestrator...")

        # Try to create AI client (will warn if no API key)
        try:
            ai_client = GeminiClient()
            print("AI client initialized successfully")
        except ValueError as e:
            print(f"Warning: {e}")
            print("   Falling back to non-AI implementation")
            ai_client = None

        orchestrator = AIOrchestrator(ai_client=ai_client)

        # Execute the complete AI workflow
        print("\nStarting AI orchestration workflow...")
        print("   Step 1: URL Discovery")
        print("   Step 2: Sample Data Collection")
        print("   Step 3: AI Structure Generation")
        print("   Step 4: AI Implementation Generation")
        print("   Step 5: Dynamic Code Execution")

        result = orchestrator.orchestrate_fetch(raw_text)

        # Display results
        if result.error:
            print(f"\nError: {result.error}")
            return None

        print(f"\nSuccess!")
        print(f"Execution Time: {result.execution_time:.2f}s")
        print(f"URL: {result.url}")
        print(f"Format: {result.format.value}")
        print(f"Method: {result.method.value}")

        if verbose:
            print(f"\nMetadata:")
            for key, value in result.metadata.items():
                print(f"   {key}: {value}")

            print(f"\nData Summary:")
            print(f"   Type: {type(result.data).__name__}")

            if isinstance(result.data, dict):
                print(f"   Keys: {list(result.data.keys())}")
                print(f"   Size: {len(result.data)} items")
            elif isinstance(result.data, list):
                print(f"   Length: {len(result.data)} items")
            elif isinstance(result.data, str):
                print(f"   Length: {len(result.data)} characters")

            print(f"\nSample Data:")
            if isinstance(result.data, dict):
                # Show first few key-value pairs
                for i, (key, value) in enumerate(result.data.items()):
                    if i >= 3:
                        print(f"   ... and {len(result.data) - 3} more items")
                        break
                    value_str = str(value)[:100]
                    if len(str(value)) > 100:
                        value_str += "..."
                    print(f"   {key}: {value_str}")
            elif isinstance(result.data, list) and result.data:
                print(f"   First item: {str(result.data[0])[:100]}...")
                if len(result.data) > 1:
                    print(f"   ... and {len(result.data) - 1} more items")
            else:
                data_str = str(result.data)[:200]
                if len(str(result.data)) > 200:
                    data_str += "..."
                print(f"   {data_str}")

        # Cleanup
        orchestrator.cleanup()

        return {
            'success': True,
            'data': result.data,
            'url': result.url,
            'execution_time': result.execution_time,
            'metadata': result.metadata,
            'format': result.format.value,
            'method': result.method.value
        }

    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        if verbose:
            import traceback
            print("\nFull traceback:")
            traceback.print_exc()
        return None


def demo_factory_pattern():
    """Demonstrate the AI DataFetch Factory pattern."""
    print("\nAI DataFetch Factory Demo")
    print("=" * 40)

    try:
        factory = AIDataFetchFactory()

        # Create a DataFetch instance from text
        datafetch = factory.create_from_text("Get latest Reuters business news")

        print("DataFetch instance created from text")
        print(f"Spec: {datafetch.spec.raw_text}")
        print(f"URLs: {datafetch.spec.urls}")

        # Use it like any DataFetch instance
        result = datafetch.fetch()

        if result.error:
            print(f"Error: {result.error}")
        else:
            print(f"Fetch completed: {result.url}")
            print(f"Data type: {type(result.data).__name__}")

        factory.cleanup()

    except Exception as e:
        print(f"Factory demo error: {str(e)}")


def interactive_mode():
    """Interactive mode for testing AI workflow."""
    setup_logging()

    print("AI-Driven DataFetch Interactive Mode")
    print("This demonstrates the complete AI orchestration workflow")
    print("Type your data requests (or 'quit' to exit)")
    print("-" * 60)

    while True:
        try:
            raw_text = input("\nEnter data request: ").strip()

            if raw_text.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if not raw_text:
                print("Please enter a data request")
                continue

            if raw_text.lower() == 'demo':
                demo_factory_pattern()
                continue

            print()
            fetch_with_ai(raw_text, verbose=True)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main CLI function."""
    if len(sys.argv) == 1:
        # No arguments - start interactive mode
        interactive_mode()
    elif len(sys.argv) == 2:
        # Single argument - process it
        setup_logging()
        raw_text = sys.argv[1]

        if raw_text.lower() == 'demo':
            demo_factory_pattern()
        else:
            fetch_with_ai(raw_text, verbose=True)
    else:
        print("AI-Driven DataFetch CLI")
        print("=" * 30)
        print("Usage:")
        print(f"  {sys.argv[0]}                    # Interactive mode")
        print(f'  {sys.argv[0]} "data request"     # Single request')
        print(f'  {sys.argv[0]} demo               # Factory demo')
        print()
        print("Examples:")
        print(f'  {sys.argv[0]} "Get latest tech news from Reuters"')
        print(f'  {sys.argv[0]} "Find Trump articles from NYT"')
        print(f'  {sys.argv[0]} "Business news headlines from BBC"')
        print()
        print("Note: Set GEMINI_API_KEY environment variable for full AI features")


if __name__ == "__main__":
    main()