#!/usr/bin/env python3
"""
Show AI-generated code without security validation blocking
"""

import sys
import os
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__)))

from src.ai.claude_client import ClaudeClient
from src.core.datafetch import FetchSpec, DataFormat, FetchMethod

def show_ai_generated_code():
    """Show the AI-generated code by bypassing security validation."""
    print("AI Code Generation Demo (Showing Raw AI Output)")
    print("=" * 60)

    load_dotenv(override=True)

    try:
        claude = ClaudeClient()
        print("[SUCCESS] Claude AI connected!")

        spec = FetchSpec(
            raw_text="Get news articles about Python programming",
            urls=["https://realpython.com/tutorials/"],
            expected_format=DataFormat.JSON,
            method=FetchMethod.REQUESTS,
            timeout=30
        )

        print(f"\nRequest: {spec.raw_text}")
        print("\nGenerating implementation...")

        # Generate but catch security error to show the code
        try:
            code = claude.generate_datafetch_implementation(spec)
            print("SUCCESS - Code passed security validation!")
            print("\n" + "="*60)
            print("AI-GENERATED IMPLEMENTATION:")
            print("="*60)
            print(code)
        except Exception as e:
            if "dangerous code pattern" in str(e):
                print("\nSecurity validation blocked the code, but let's see what was generated...")

                # Get the raw response by calling the generation method directly
                # This shows what Claude actually generated before validation
                prompt = claude._create_implementation_prompt(
                    spec.raw_text, spec.urls, spec.expected_format,
                    spec.method, "", "src.core.datafetch.DataFetch"
                )

                raw_response = claude._generate_with_retry(prompt, max_tokens=4096)
                raw_code = claude._extract_code_from_response(raw_response)

                print("\n" + "="*60)
                print("RAW AI-GENERATED CODE (before security validation):")
                print("="*60)
                print(raw_code)
                print("="*60)

                print(f"\nSecurity Error: {str(e)}")
                print("\nAnalysis:")

                # Analyze what the AI generated
                lines = raw_code.split('\n')
                has_class = any('class' in line and 'DataFetch' in line for line in lines)
                has_methods = len([line for line in lines if 'def ' in line])
                has_abstract_methods = any(method in raw_code for method in ['fetch(', 'afetch', 'validate_data', 'extract_structure'])

                print(f"  - Contains DataFetch class: {has_class}")
                print(f"  - Number of methods: {has_methods}")
                print(f"  - Implements abstract methods: {has_abstract_methods}")
                print(f"  - Total lines: {len(lines)}")

                if 'open(' in raw_code:
                    print("  - Note: Contains file operations (triggered security)")
            else:
                raise

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    show_ai_generated_code()