#!/usr/bin/env python3
"""Quick and simple DataFetch CLI."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.datafetch import FetchSpec, DataFormat, FetchMethod
from src.collectors.http_client import HTTPClient


def quick_fetch(text):
    """Simple fetch function."""
    print(f"Searching for: {text}")
    
    text_lower = text.lower()
    
    # Smart URL inference
    if 'trump' in text_lower:
        # Use the specialized Trump fetcher
        from src.implementations.nyt_trump_fetch import NYTTrumpFetch
        spec = FetchSpec(text, ["https://www.nytimes.com/"])
        result = NYTTrumpFetch(spec).fetch()
        print(f"Success: {not result.error}")
        if result.error:
            print(f"Error: {result.error}")
        else:
            print(f"Found {result.metadata.get('articles_found', 0)} Trump articles")
        return
    elif 'reuters' in text_lower:
        if 'business' in text_lower:
            urls = ["https://www.reuters.com/business/"]
        elif 'world' in text_lower:
            urls = ["https://www.reuters.com/world/"]
        else:
            urls = ["https://www.reuters.com/"]
    elif 'nyt' in text_lower or 'times' in text_lower:
        if 'tech' in text_lower:
            urls = ["https://www.nytimes.com/section/technology"]
        elif 'business' in text_lower:
            urls = ["https://www.nytimes.com/section/business"]
        elif 'politics' in text_lower:
            urls = ["https://www.nytimes.com/section/politics"]
        else:
            urls = ["https://www.nytimes.com/"]
    elif 'bbc' in text_lower:
        urls = ["https://www.bbc.com/news"]
    elif 'cnn' in text_lower:
        urls = ["https://edition.cnn.com/"]
    else:
        urls = ["https://example.com"]  # Fallback
        print("⚠️ Warning: No specific news site mentioned")
    
    print(f"Target URL: {urls[0]}")
    
    spec = FetchSpec(text, urls)
    client = HTTPClient(spec)
    result = client.fetch()
    
    print(f"Success: {not result.error}")
    if result.error:
        print(f"Error: {result.error}")
    else:
        print(f"Data type: {type(result.data)}")
        print(f"Time: {result.execution_time:.2f}s")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        quick_fetch(" ".join(sys.argv[1:]))
    else:
        print("Usage: python quick_fetch.py 'your search text'")
        print("Examples:")
        print("  python quick_fetch.py 'NYT technology articles'")
        print("  python quick_fetch.py 'Find Trump news'")