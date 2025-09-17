#!/usr/bin/env python3
"""Simple examples of searching without custom implementations."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.datafetch import FetchSpec, DataFormat, FetchMethod
from src.collectors.http_client import HTTPClient


def search_nyt_sports():
    """Example: Search NYT Sports without any custom implementation."""
    print("=== NYT Sports Search ===")
    
    spec = FetchSpec(
        raw_text="Get sports news from New York Times",
        urls=["https://httpbin.org/html"],  # Using test endpoint
        expected_format=DataFormat.HTML,
        method=FetchMethod.REQUESTS
    )
    
    # HTTPClient automatically recognizes news sites and optimizes
    client = HTTPClient(spec)
    result = client.fetch()
    
    print(f"Success: {result.error is None}")
    print(f"Data received: {len(str(result.data)) if result.data else 0} characters")
    return result


def search_nyt_politics():
    """Example: Search NYT Politics."""
    print("=== NYT Politics Search ===")
    
    spec = FetchSpec(
        raw_text="Find political articles from NYT",
        urls=["https://httpbin.org/json"],  # Test endpoint returning JSON
        expected_format=DataFormat.JSON,
        method=FetchMethod.REQUESTS
    )
    
    client = HTTPClient(spec)
    result = client.fetch()
    
    print(f"Success: {result.error is None}")
    if isinstance(result.data, dict):
        print(f"JSON keys: {list(result.data.keys())}")
    return result


def search_any_topic(topic, site_url="https://httpbin.org/html"):
    """Generic function to search any topic from any site."""
    print(f"=== Searching for: {topic} ===")
    
    spec = FetchSpec(
        raw_text=f"Find {topic} articles",
        urls=[site_url],
        expected_format=DataFormat.HTML,
        method=FetchMethod.REQUESTS,
        timeout=10
    )
    
    client = HTTPClient(spec)
    result = client.fetch()
    
    print(f"Topic: {topic}")
    print(f"URL: {site_url}")
    print(f"Success: {result.error is None}")
    print(f"Execution time: {result.execution_time:.2f}s")
    
    return result


# Show the pattern
def show_the_pattern():
    """Show the simple pattern that works for everything."""
    print("\n" + "="*50)
    print("THE SIMPLE PATTERN (Works for anything!)")
    print("="*50)
    
    pattern = '''
# 1. Import
from src.core.datafetch import FetchSpec
from src.collectors.http_client import HTTPClient

# 2. Describe what you want
spec = FetchSpec(
    raw_text="Your search description",
    urls=["https://your-target-site.com"]
)

# 3. Search (HTTPClient handles everything automatically)
result = HTTPClient(spec).fetch()

# 4. Use the data
if result.error:
    print("Failed:", result.error)
else:
    print("Got data:", type(result.data))
    '''
    
    print(pattern)


if __name__ == "__main__":
    print("Simple Search Examples (No Custom Implementation Needed)")
    print("=" * 60)
    
    # Example 1
    search_nyt_sports()
    print()
    
    # Example 2  
    search_nyt_politics()
    print()
    
    # Example 3 - Search any topic
    search_any_topic("climate change")
    print()
    search_any_topic("artificial intelligence") 
    print()
    search_any_topic("cryptocurrency")
    print()
    
    # Show the universal pattern
    show_the_pattern()
    
    print("\nREAL WORLD USAGE:")
    print("Just replace 'https://httpbin.org/html' with real URLs like:")
    print("- https://www.nytimes.com/section/sports")
    print("- https://www.nytimes.com/section/politics") 
    print("- https://www.reuters.com/business/")
    print("- https://www.bbc.com/news/technology")
    print("\nHTTPClient will automatically optimize for each site!")