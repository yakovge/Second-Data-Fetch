#!/usr/bin/env python3
"""
Search any content from any news site without custom implementations.
Uses the built-in intelligence of HTTPClient.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.datafetch import FetchSpec, DataFormat, FetchMethod
from src.collectors.http_client import HTTPClient


def search_nyt_general(topic="technology"):
    """Search NYT for any topic using generic HTTPClient."""
    print(f"Searching NYT for: {topic}")
    print("-" * 40)
    
    # Create generic spec - HTTPClient handles the rest
    spec = FetchSpec(
        raw_text=f"Get {topic} articles from New York Times",
        urls=[
            f"https://www.nytimes.com/section/{topic}",
            "https://www.nytimes.com/",
        ],
        expected_format=DataFormat.HTML,  # NYT serves HTML
        method=FetchMethod.REQUESTS,
        timeout=30,
        retry_count=2
    )
    
    # HTTPClient automatically optimizes for NYT
    client = HTTPClient(spec)
    result = client.fetch()
    
    print(f"Status: {'Success' if not result.error else 'Failed'}")
    if result.error:
        print(f"Error: {result.error}")
    else:
        print(f"Data type: {type(result.data)}")
        print(f"URL fetched: {result.url}")
        print(f"Execution time: {result.execution_time:.2f}s")
        
        # HTTPClient automatically extracts articles when it detects news sites
        if isinstance(result.data, dict):
            articles = result.data.get('articles', [])
            print(f"Articles found: {len(articles)}")
            
            # Show first few articles
            for i, article in enumerate(articles[:3], 1):
                if isinstance(article, dict):
                    title = article.get('title', 'No title')[:60]
                    print(f"  {i}. {title}...")
    
    return result


def search_any_news_site(site_url, topic):
    """Search any news website for any topic."""
    print(f"Searching {site_url} for: {topic}")
    print("-" * 40)
    
    spec = FetchSpec(
        raw_text=f"Find {topic} news articles",
        urls=[site_url],
        expected_format=DataFormat.HTML,
        method=FetchMethod.REQUESTS
    )
    
    client = HTTPClient(spec)
    result = client.fetch()
    
    print(f"Status: {'Success' if not result.error else 'Failed'}")
    if not result.error:
        print(f"Content length: {len(str(result.data)) if result.data else 0} chars")
        
        # HTTPClient automatically structures news content
        if isinstance(result.data, dict) and 'articles' in result.data:
            print(f"Structured articles: {len(result.data['articles'])}")
        else:
            print("Raw content extracted (may need custom processing)")
    
    return result


def demonstrate_built_in_intelligence():
    """Show what HTTPClient automatically does for different sites."""
    
    test_cases = [
        ("https://www.nytimes.com/section/business", "business"),
        ("https://www.reuters.com/business/", "business"),  
        ("https://www.bbc.com/news/technology", "technology"),
        ("https://edition.cnn.com/politics", "politics"),
    ]
    
    print("Built-in Intelligence Demo")
    print("=" * 50)
    
    for url, topic in test_cases:
        print(f"Site: {url}")
        
        # Check what HTTPClient will automatically do
        is_news = any(domain in url.lower() 
                     for domain in ['news', 'reuters', 'nytimes', 'bbc', 'cnn', 'guardian'])
        
        print(f"  Auto-detected as news site: {is_news}")
        if is_news:
            print("  Will automatically apply:")
            print("    - News-optimized headers (NewsBot/1.0)")
            print("    - Article extraction patterns")
            print("    - Content structuring for articles")
            print("    - HTML parsing optimizations")
        else:
            print("  Will use standard web scraping")
        print()


if __name__ == "__main__":
    print("Generic Search Examples (No Custom Implementation)")
    print("=" * 60)
    
    try:
        # Example 1: Search NYT for technology
        print("Example 1: NYT Technology Articles")
        search_nyt_general("technology")
        print("\n" + "="*60 + "\n")
        
        # Example 2: Search different topics
        print("Example 2: NYT Business Articles")  
        search_nyt_general("business")
        print("\n" + "="*60 + "\n")
        
        # Example 3: Search other news sites
        print("Example 3: Other News Sites")
        search_any_news_site("https://www.reuters.com/business/", "finance")
        print("\n" + "="*60 + "\n")
        
        # Example 4: Show built-in intelligence
        demonstrate_built_in_intelligence()
        
        print("All examples completed!")
        print("\nKey Point: HTTPClient automatically handles news sites")
        print("No custom implementation needed for basic news searching!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Note: Using test URLs - real fetching may need network access")


# Quick usage examples
def quick_examples():
    """Show the simplest way to search anything."""
    
    print("\n" + "="*40)
    print("QUICK USAGE EXAMPLES")
    print("="*40)
    
    examples = [
        ("NYT Sports", "https://www.nytimes.com/section/sports", "sports"),
        ("NYT Politics", "https://www.nytimes.com/section/politics", "politics"), 
        ("Reuters World", "https://www.reuters.com/world/", "world news"),
        ("BBC Tech", "https://www.bbc.com/news/technology", "technology"),
    ]
    
    for name, url, topic in examples:
        print(f"\n# {name}:")
        print(f"spec = FetchSpec('{topic} articles', ['{url}'])")
        print(f"result = HTTPClient(spec).fetch()")
        print(f"# HTTPClient auto-optimizes for news sites!")


if __name__ == "__main__":
    main()
    quick_examples()