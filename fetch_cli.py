#!/usr/bin/env python3
"""
Simple CLI for DataFetch system - Input raw text, get data back.
Usage: python fetch_cli.py "Your search request"
"""

import sys
import os
import json
from typing import Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.datafetch import FetchSpec, DataFormat, FetchMethod
from src.collectors.http_client import HTTPClient
from src.spec.parser import RawTextParser


def fetch_data(raw_text: str, format_output: bool = True) -> Optional[dict]:
    """
    Main function to fetch data from raw text input.
    
    Args:
        raw_text: User's search request
        format_output: Whether to format output nicely
        
    Returns:
        Dictionary with results or None if failed
    """
    try:
        print(f"Processing: {raw_text}")
        print("-" * 50)
        
        # Step 1: Parse the raw text to extract URLs and intent
        parser = RawTextParser()
        parsed = parser.parse(raw_text)
        
        print(f"Extracted URLs: {parsed.extracted_urls}")
        print(f"Suggested format: {parsed.suggested_format}")
        print(f"Suggested method: {parsed.suggested_method}")
        print(f"Confidence: {parsed.confidence_score:.2f}")
        print()
        
        # Step 2: Create FetchSpec with smart URL inference
        urls = parsed.extracted_urls
        
        # If no URLs extracted, infer from the request text
        if not urls:
            text_lower = raw_text.lower()
            if 'reuters' in text_lower:
                if 'business' in text_lower:
                    urls = ["https://www.reuters.com/business/"]
                elif 'world' in text_lower or 'international' in text_lower:
                    urls = ["https://www.reuters.com/world/"]
                elif 'tech' in text_lower or 'technology' in text_lower:
                    urls = ["https://www.reuters.com/technology/"]
                else:
                    urls = ["https://www.reuters.com/"]
            elif 'nyt' in text_lower or 'new york times' in text_lower or 'times' in text_lower:
                if 'tech' in text_lower or 'technology' in text_lower:
                    urls = ["https://www.nytimes.com/section/technology"]
                elif 'business' in text_lower:
                    urls = ["https://www.nytimes.com/section/business"]
                elif 'politics' in text_lower or 'political' in text_lower:
                    urls = ["https://www.nytimes.com/section/politics"]
                elif 'sports' in text_lower:
                    urls = ["https://www.nytimes.com/section/sports"]
                else:
                    urls = ["https://www.nytimes.com/"]
            elif 'bbc' in text_lower:
                if 'business' in text_lower:
                    urls = ["https://www.bbc.com/news/business"]
                elif 'tech' in text_lower:
                    urls = ["https://www.bbc.com/news/technology"]
                else:
                    urls = ["https://www.bbc.com/news"]
            elif 'cnn' in text_lower:
                if 'business' in text_lower:
                    urls = ["https://edition.cnn.com/business"]
                elif 'politics' in text_lower:
                    urls = ["https://edition.cnn.com/politics"]
                else:
                    urls = ["https://edition.cnn.com/"]
            else:
                # Fallback - still use test endpoint but warn user
                urls = ["https://httpbin.org/json"]
                print("⚠️ Warning: No specific news site mentioned, using test endpoint")
                print("   Try: 'Reuters business news' or 'NYT technology articles'")
        
        print(f"Target URLs: {urls}")
        
        spec = FetchSpec(
            raw_text=raw_text,
            urls=urls,
            expected_format=parsed.suggested_format or DataFormat.HTML,
            method=parsed.suggested_method or FetchMethod.REQUESTS,
            timeout=30,
            retry_count=2
        )
        
        # Step 3: Fetch the data
        print("Fetching data...")
        client = HTTPClient(spec)
        result = client.fetch()
        
        # Step 4: Display results
        if result.error:
            print(f"Error: {result.error}")
            return None
        
        print(f"Success!")
        print(f"URL: {result.url}")
        print(f"Time: {result.execution_time:.2f}s")
        print(f"Data type: {type(result.data)}")
        
        if format_output:
            print("\n" + "="*50)
            print("EXTRACTED DATA:")
            print("="*50)
            
            if isinstance(result.data, dict):
                # Pretty print the data
                for key, value in result.data.items():
                    if isinstance(value, (str, int, float, bool)):
                        print(f"{key}: {value}")
                    elif isinstance(value, list):
                        print(f"{key}: [{len(value)} items]")
                        for i, item in enumerate(value[:3], 1):  # Show first 3
                            print(f"  {i}. {str(item)[:100]}...")
                        if len(value) > 3:
                            print(f"  ... and {len(value)-3} more")
                    else:
                        print(f"{key}: {type(value).__name__}")
            else:
                print(str(result.data)[:500] + "..." if len(str(result.data)) > 500 else str(result.data))
        
        return {
            'success': True,
            'data': result.data,
            'url': result.url,
            'execution_time': result.execution_time,
            'metadata': result.metadata
        }
        
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return None


def interactive_mode():
    """Interactive mode - keep asking for input."""
    print("DataFetch Interactive Mode")
    print("Type your search requests (or 'quit' to exit)")
    print("-" * 50)
    
    while True:
        try:
            raw_text = input("\nEnter search request: ").strip()
            
            if raw_text.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not raw_text:
                print("Please enter a search request")
                continue
            
            print()
            fetch_data(raw_text)
            
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
        raw_text = sys.argv[1]
        fetch_data(raw_text)
    else:
        print("Usage:")
        print(f"  {sys.argv[0]}                    # Interactive mode")
        print(f'  {sys.argv[0]} "search request"   # Single search')
        print()
        print("Examples:")
        print(f'  {sys.argv[0]} "Get NYT tech articles"')
        print(f'  {sys.argv[0]} "Find Trump news from NYT"')
        print(f'  {sys.argv[0]} "Reuters business news"')


if __name__ == "__main__":
    main()