#!/usr/bin/env python3
"""Simple demo of NYT data fetching capabilities."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.datafetch import FetchSpec, DataFormat, FetchMethod
from src.collectors.http_client import HTTPClient


def main():
    print("NYT Data Fetching Demo")
    print("=" * 50)
    
    # The system automatically recognizes NYT URLs and optimizes
    print("1. URL Intelligence Test")
    print("-" * 30)
    
    nyt_urls = [
        "https://www.nytimes.com/section/technology",
        "https://www.nytimes.com/section/politics", 
        "https://www.nytimes.com/"
    ]
    
    for url in nyt_urls:
        # Check if system recognizes as news site
        is_news = any(domain in url.lower() for domain in ['nytimes', 'reuters', 'bbc', 'cnn'])
        print(f"URL: {url}")
        print(f"Recognized as news site: {is_news}")
        print(f"Will use: {'News-optimized headers' if is_news else 'Standard headers'}")
        print()
    
    print("2. How to Fetch NYT Data")
    print("-" * 30)
    
    # Create a spec for NYT
    spec = FetchSpec(
        raw_text="Get latest technology articles from New York Times",
        urls=["https://httpbin.org/html"],  # Using test endpoint for demo
        expected_format=DataFormat.HTML,
        method=FetchMethod.REQUESTS,
        timeout=10
    )
    
    print("Created FetchSpec:")
    print(f"  Request: {spec.raw_text}")
    print(f"  URLs: {spec.urls}")
    print(f"  Format: {spec.expected_format}")
    print(f"  Method: {spec.method}")
    
    # HTTPClient automatically applies NYT optimizations
    client = HTTPClient(spec)
    print(f"HTTPClient created with session ID: {client._session_id[:8]}...")
    
    print("\n3. Available Approaches")
    print("-" * 30)
    print("A) Generic NYT fetching:")
    print("   - HTTPClient automatically detects NYT URLs")
    print("   - Applies news-site optimized headers") 
    print("   - Uses article extraction patterns")
    print("   - Works for any NYT content")
    
    print("\nB) Specialized implementations:")
    print("   - NYTTrumpFetch: Custom Trump article extractor")
    print("   - You can create: NYTTechFetch, NYTSportsFetch, etc.")
    print("   - Each inherits from DataFetch abstract class")
    
    print("\n4. System Intelligence")
    print("-" * 30)
    print("When you provide a NYT URL, the system:")
    print("1. Recognizes 'nytimes' domain")
    print("2. Automatically applies news headers:")
    print("   - User-Agent: NewsBot/1.0")
    print("   - Accept-Language: en-US,en;q=0.9")
    print("3. Uses article extraction patterns")
    print("4. Structures data for news format")
    
    print("\n5. Quick Start Examples")
    print("-" * 30)
    print("# General NYT articles:")
    print("spec = FetchSpec(")
    print("    raw_text='Get NYT tech articles',")
    print("    urls=['https://www.nytimes.com/section/technology']")
    print(")")
    print("client = HTTPClient(spec)")
    print("result = client.fetch()")
    
    print("\n# Trump articles (specialized):")
    print("from src.implementations.nyt_trump_fetch import NYTTrumpFetch")
    print("fetcher = NYTTrumpFetch(spec)")
    print("trump_articles = fetcher.fetch()")
    
    print("\nDemo complete!")


if __name__ == "__main__":
    main()