#!/usr/bin/env python3
"""
Fixed CLI for DataFetch system - Uses predefined news sources instead of broken URL extraction.
Usage: python fixed_fetch_cli.py "Your search request"
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.datafetch import FetchSpec, DataFormat, FetchMethod
from src.collectors.http_client import HTTPClient


# Predefined news sources mapping
NEWS_SOURCES = {
    # New York Times
    'nytimes': {
        'base': 'https://www.nytimes.com/',
        'technology': 'https://www.nytimes.com/section/technology',
        'business': 'https://www.nytimes.com/section/business',
        'politics': 'https://www.nytimes.com/section/politics',
        'sports': 'https://www.nytimes.com/section/sports',
        'world': 'https://www.nytimes.com/section/world',
    },
    # Reuters
    'reuters': {
        'base': 'https://www.reuters.com/',
        'business': 'https://www.reuters.com/business/',
        'technology': 'https://www.reuters.com/technology/',
        'world': 'https://www.reuters.com/world/',
        'markets': 'https://www.reuters.com/markets/',
    },
    # BBC
    'bbc': {
        'base': 'https://www.bbc.com/news',
        'business': 'https://www.bbc.com/news/business',
        'technology': 'https://www.bbc.com/news/technology',
        'world': 'https://www.bbc.com/news/world',
    },
    # CNN
    'cnn': {
        'base': 'https://edition.cnn.com/',
        'business': 'https://edition.cnn.com/business',
        'politics': 'https://edition.cnn.com/politics',
        'world': 'https://edition.cnn.com/world',
        'tech': 'https://edition.cnn.com/business/tech',
    }
}


def smart_url_inference(text: str) -> list:
    """
    Smart URL inference using predefined news sources.
    This replaces the broken URL extraction logic.
    """
    text_lower = text.lower().strip()
    urls = []
    
    print(f"Analyzing request: '{text}'")
    
    # Detect news source
    source = None
    if any(word in text_lower for word in ['nyt', 'new york times', 'times']):
        source = 'nytimes'
    elif 'reuters' in text_lower:
        source = 'reuters'
    elif 'bbc' in text_lower:
        source = 'bbc'
    elif 'cnn' in text_lower:
        source = 'cnn'
    
    if not source:
        print("No recognized news source found!")
        print("   Supported: NYT, Reuters, BBC, CNN")
        print("   Try: 'NYT technology articles' or 'Reuters business news'")
        return []
    
    # Detect section/category
    section = 'base'  # default
    if any(word in text_lower for word in ['tech', 'technology']):
        section = 'technology'
    elif 'business' in text_lower:
        section = 'business'  
    elif any(word in text_lower for word in ['politics', 'political']):
        section = 'politics'
    elif 'sports' in text_lower:
        section = 'sports'
    elif any(word in text_lower for word in ['world', 'international']):
        section = 'world'
    elif 'market' in text_lower:
        section = 'markets'
    
    # Get the URL
    if section in NEWS_SOURCES[source]:
        urls = [NEWS_SOURCES[source][section]]
    else:
        urls = [NEWS_SOURCES[source]['base']]
    
    print(f"Detected: {source.upper()} - {section}")
    print(f"Target URL: {urls[0]}")
    
    return urls


def fixed_fetch_data(raw_text: str) -> dict:
    """
    Fixed fetch function that uses predefined news sources.
    """
    print("=" * 60)
    print(f"FIXED DataFetch - Processing: {raw_text}")
    print("=" * 60)
    
    # Use smart URL inference instead of broken extraction
    urls = smart_url_inference(raw_text)
    
    if not urls:
        return {'success': False, 'error': 'No valid news source detected'}
    
    # Handle Trump articles specially
    if 'trump' in raw_text.lower():
        print("Using specialized Trump article fetcher...")
        try:
            from src.implementations.nyt_trump_fetch import NYTTrumpFetch
            spec = FetchSpec(raw_text, ["https://www.nytimes.com/"])
            trump_fetcher = NYTTrumpFetch(spec)
            result = trump_fetcher.fetch()
            
            print(f"Success: {not result.error}")
            if result.error:
                print(f"Error: {result.error}")
            else:
                print(f"Found {result.metadata.get('articles_found', 0)} Trump articles")
                if result.data and isinstance(result.data, dict):
                    articles = result.data.get('articles', [])
                    for i, article in enumerate(articles[:3], 1):
                        if isinstance(article, dict):
                            title = article.get('title', 'No title')[:80]
                            print(f"  {i}. {title}...")
            
            return {
                'success': not result.error,
                'data': result.data,
                'specialized': 'NYTTrumpFetch',
                'articles_found': result.metadata.get('articles_found', 0)
            }
        except Exception as e:
            print(f"Trump fetcher error: {e}")
            # Fall back to regular fetch
    
    # Regular news fetch
    print("Fetching from news source...")
    
    try:
        spec = FetchSpec(
            raw_text=raw_text,
            urls=urls,
            expected_format=DataFormat.HTML,  # Most news sites serve HTML
            method=FetchMethod.REQUESTS,
            timeout=30,
            retry_count=2
        )
        
        client = HTTPClient(spec)
        result = client.fetch()
        
        print(f"Success: {not result.error}")
        print(f"URL: {result.url}")
        print(f"Time: {result.execution_time:.2f}s")
        print(f"Data type: {type(result.data)}")
        
        if result.error:
            print(f"Error: {result.error}")
            return {'success': False, 'error': result.error}
        
        # Show some data info
        if isinstance(result.data, dict):
            print(f"Data keys: {list(result.data.keys())}")
            
            # Show extracted content
            if 'title' in result.data:
                print(f"Page title: {result.data['title'][:100]}...")
            if 'articles' in result.data:
                articles = result.data['articles']
                print(f"Articles found: {len(articles)}")
                for i, article in enumerate(articles[:3], 1):
                    if isinstance(article, dict) and 'title' in article:
                        print(f"  {i}. {article['title'][:60]}...")
        
        return {
            'success': True,
            'data': result.data,
            'url': result.url,
            'execution_time': result.execution_time
        }
        
    except Exception as e:
        print(f"Fetch error: {e}")
        return {'success': False, 'error': str(e)}


def main():
    """Main function."""
    if len(sys.argv) == 1:
        # Interactive mode
        print("Fixed DataFetch Interactive Mode")
        print("Supported sources: NYT, Reuters, BBC, CNN")
        print("Type 'quit' to exit")
        print("-" * 50)
        
        while True:
            try:
                text = input("\nEnter search request: ").strip()
                if text.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                if text:
                    fixed_fetch_data(text)
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
    else:
        # Single command
        text = " ".join(sys.argv[1:])
        fixed_fetch_data(text)


if __name__ == "__main__":
    main()