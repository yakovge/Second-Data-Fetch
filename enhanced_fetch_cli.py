#!/usr/bin/env python3
"""
Enhanced CLI with BrowserClient integration - CLAUDE.md Compliant
Uses both HTTPClient and BrowserClient as tools, following the strategy:
"Use provided clients: Import HTTPClient or BrowserClient, don't create from scratch"
"""

import sys
import os
from typing import Optional
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.datafetch import FetchSpec, DataFormat, FetchMethod
from src.collectors.http_client import HTTPClient
from src.collectors.browser_client import BrowserClient


# CLAUDE.md Compliant: Predefined news sources (security-first approach)
NEWS_SOURCES = {
    'nytimes': {
        'base': 'https://www.nytimes.com/',
        'technology': 'https://www.nytimes.com/section/technology',
        'business': 'https://www.nytimes.com/section/business',
        'politics': 'https://www.nytimes.com/section/politics',
        'sports': 'https://www.nytimes.com/section/sports',
    },
    'reuters': {
        'base': 'https://www.reuters.com/',
        'business': 'https://www.reuters.com/business/',
        'technology': 'https://www.reuters.com/technology/',
        'world': 'https://www.reuters.com/world/',
    },
    'bbc': {
        'base': 'https://www.bbc.com/news',
        'business': 'https://www.bbc.com/news/business',
        'technology': 'https://www.bbc.com/news/technology',
    },
    'cnn': {
        'base': 'https://edition.cnn.com/',
        'business': 'https://edition.cnn.com/business',
        'politics': 'https://edition.cnn.com/politics',
    }
}

# Sites that typically need BrowserClient (JavaScript-heavy or block requests)
BROWSER_PREFERRED_SITES = {
    'reuters.com',
    'wsj.com', 
    'ft.com',
    'bloomberg.com'
}


def smart_url_inference(text: str) -> tuple[list, bool]:
    """
    CLAUDE.md Compliant: Smart URL inference with browser preference detection.
    Returns (urls, use_browser)
    """
    text_lower = text.lower().strip()
    
    print(f"Analyzing: '{text}'")
    
    # Detect source
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
        print("Supported: NYT, Reuters, BBC, CNN")
        return [], False
    
    # Detect section
    section = 'base'
    if any(word in text_lower for word in ['tech', 'technology']):
        section = 'technology'
    elif 'business' in text_lower:
        section = 'business'  
    elif any(word in text_lower for word in ['politics', 'political']):
        section = 'politics'
    elif 'sports' in text_lower:
        section = 'sports'
    elif 'world' in text_lower:
        section = 'world'
    
    # Get URL
    if section in NEWS_SOURCES[source]:
        urls = [NEWS_SOURCES[source][section]]
    else:
        urls = [NEWS_SOURCES[source]['base']]
    
    # Determine if BrowserClient is preferred
    use_browser = any(browser_site in urls[0] for browser_site in BROWSER_PREFERRED_SITES)
    
    print(f"Detected: {source.upper()} - {section}")
    print(f"Target URL: {urls[0]}")
    print(f"Preferred method: {'BrowserClient' if use_browser else 'HTTPClient'}")
    
    return urls, use_browser


def fetch_with_fallback(raw_text: str, urls: list, save_screenshot: bool = False) -> dict:
    """
    CLAUDE.md Compliant: Use collectors as tools with intelligent fallback.
    "Prioritize Requests over Playwright - use browser client only when necessary"
    """
    print("\n" + "="*60)
    print("ENHANCED DataFetch - Using Intelligent Client Selection")
    print("="*60)
    
    # Handle Trump articles with specialized implementation (CLAUDE.md compliant)
    if 'trump' in raw_text.lower():
        return fetch_trump_specialized(raw_text)
    
    # Determine preferred client
    _, use_browser_preferred = smart_url_inference(raw_text)
    
    spec = FetchSpec(
        raw_text=raw_text,
        urls=urls,
        expected_format=DataFormat.HTML,
        method=FetchMethod.PLAYWRIGHT if use_browser_preferred else FetchMethod.REQUESTS,
        timeout=30,
        retry_count=2
    )
    
    # CLAUDE.md Strategy: "Use provided clients as tools"
    if use_browser_preferred:
        print("1st attempt: BrowserClient (preferred for this site)")
        result = try_browser_client(spec, save_screenshot)
        if result['success']:
            return result
        
        print("BrowserClient failed, falling back to HTTPClient...")
        result = try_http_client(spec)
        return result
    else:
        print("1st attempt: HTTPClient (fast, CLAUDE.md preferred)")
        result = try_http_client(spec)
        if result['success']:
            return result
        
        print("HTTPClient failed, falling back to BrowserClient...")
        result = try_browser_client(spec, save_screenshot)
        return result


def try_http_client(spec: FetchSpec) -> dict:
    """CLAUDE.md Compliant: Use HTTPClient as tool."""
    try:
        client = HTTPClient(spec)
        fetch_result = client.fetch()
        
        success = not fetch_result.error
        print(f"HTTPClient result: {'Success' if success else 'Failed'}")
        
        if success:
            print(f"URL: {fetch_result.url}")
            print(f"Time: {fetch_result.execution_time:.2f}s") 
            print(f"Data type: {type(fetch_result.data)}")
            
            if isinstance(fetch_result.data, dict):
                print(f"Data keys: {list(fetch_result.data.keys())}")
        else:
            print(f"Error: {fetch_result.error}")
        
        return {
            'success': success,
            'method': 'HTTPClient',
            'data': fetch_result.data,
            'error': fetch_result.error,
            'url': fetch_result.url,
            'execution_time': fetch_result.execution_time
        }
        
    except Exception as e:
        print(f"HTTPClient exception: {e}")
        return {'success': False, 'method': 'HTTPClient', 'error': str(e)}


def try_browser_client(spec: FetchSpec, save_screenshot: bool = False) -> dict:
    """CLAUDE.md Compliant: Use BrowserClient as tool."""
    try:
        browser_client = BrowserClient(spec)
        fetch_result = browser_client.fetch()
        
        success = not fetch_result.error
        print(f"BrowserClient result: {'Success' if success else 'Failed'}")
        
        if success:
            print(f"URL: {fetch_result.url}")
            print(f"Time: {fetch_result.execution_time:.2f}s")
            print(f"Data type: {type(fetch_result.data)}")
            
            # Screenshot capability
            if save_screenshot and fetch_result.url:
                try:
                    import asyncio
                    screenshot_bytes = asyncio.run(
                        browser_client.capture_screenshot(fetch_result.url)
                    )
                    screenshot_path = f"screenshot_{int(time.time())}.png"
                    Path(screenshot_path).write_bytes(screenshot_bytes)
                    print(f"Screenshot saved: {screenshot_path}")
                except Exception as e:
                    print(f"Screenshot failed: {e}")
            
            if isinstance(fetch_result.data, dict):
                print(f"Data keys: {list(fetch_result.data.keys())}")
        else:
            print(f"Error: {fetch_result.error}")
        
        return {
            'success': success,
            'method': 'BrowserClient',
            'data': fetch_result.data,
            'error': fetch_result.error,
            'url': fetch_result.url,
            'execution_time': fetch_result.execution_time
        }
        
    except Exception as e:
        print(f"BrowserClient exception: {e}")
        return {'success': False, 'method': 'BrowserClient', 'error': str(e)}


def fetch_trump_specialized(raw_text: str) -> dict:
    """CLAUDE.md Compliant: Use specialized DataFetch implementation."""
    
    # Check if user specified a particular news source for Trump content
    text_lower = raw_text.lower()
    if 'reuters' in text_lower:
        print("Trump articles from Reuters - using BrowserClient with Trump filtering...")
        return fetch_trump_from_source(raw_text, "reuters")
    elif 'bbc' in text_lower:
        print("Trump articles from BBC - using HTTPClient with Trump filtering...")
        return fetch_trump_from_source(raw_text, "bbc")
    elif 'cnn' in text_lower:
        print("Trump articles from CNN - using HTTPClient with Trump filtering...")
        return fetch_trump_from_source(raw_text, "cnn")
    else:
        print("Using specialized NYTTrumpFetch implementation...")
    
    try:
        from src.implementations.nyt_trump_fetch import NYTTrumpFetch
        spec = FetchSpec(raw_text, ["https://www.nytimes.com/"])
        trump_fetcher = NYTTrumpFetch(spec)
        result = trump_fetcher.fetch()
        
        success = not result.error
        print(f"NYTTrumpFetch result: {'Success' if success else 'Failed'}")
        
        if success:
            articles_found = result.metadata.get('articles_found', 0)
            print(f"Trump articles found: {articles_found}")
            
            if result.data and isinstance(result.data, dict):
                articles = result.data.get('articles', [])
                for i, article in enumerate(articles[:3], 1):
                    if isinstance(article, dict):
                        title = article.get('title', 'No title')[:80]
                        print(f"  {i}. {title}...")
        else:
            print(f"Error: {result.error}")
        
        return {
            'success': success,
            'method': 'NYTTrumpFetch',
            'data': result.data,
            'articles_found': result.metadata.get('articles_found', 0),
            'error': result.error
        }
        
    except Exception as e:
        print(f"NYTTrumpFetch exception: {e}")
        return {'success': False, 'method': 'NYTTrumpFetch', 'error': str(e)}


def fetch_trump_from_source(raw_text: str, source: str) -> dict:
    """CLAUDE.md Compliant: Fetch Trump articles from specified source using appropriate client."""
    
    # Get the appropriate URL for the source
    urls, use_browser = smart_url_inference(raw_text)
    if not urls:
        return {'success': False, 'error': f'Could not determine URL for {source}'}
    
    spec = FetchSpec(
        raw_text=raw_text,
        urls=urls,
        expected_format=DataFormat.HTML,
        method=FetchMethod.PLAYWRIGHT if use_browser else FetchMethod.REQUESTS,
        timeout=30
    )
    
    # Use appropriate client based on source requirements
    if use_browser:
        print(f"Fetching from {urls[0]} using BrowserClient...")
        result = try_browser_client(spec)
        if not result['success']:
            print("BrowserClient failed, trying HTTPClient...")
            result = try_http_client(spec)
    else:
        print(f"Fetching from {urls[0]} using HTTPClient...")
        result = try_http_client(spec)
        if not result['success']:
            print("HTTPClient failed, trying BrowserClient...")
            result = try_browser_client(spec)
    
    # Add Trump-specific processing to the result
    if result['success'] and result.get('data'):
        # Simple Trump mention detection in the fetched content
        data_str = str(result['data']).lower()
        trump_mentions = data_str.count('trump')
        
        result['trump_mentions'] = trump_mentions
        result['source'] = source.upper()
        
        print(f"Found {trump_mentions} Trump mentions in {source.upper()} content")
        
        if trump_mentions == 0:
            print("No Trump mentions found in the fetched content")
        else:
            print(f"Content appears to contain Trump-related information")
    
    return result


def main():
    """Main CLI function with enhanced capabilities."""
    import time
    
    if len(sys.argv) == 1:
        # Interactive mode
        print("Enhanced DataFetch CLI - HTTPClient + BrowserClient")
        print("Features: Smart fallback, screenshots, specialized implementations")
        print("Following CLAUDE.md strategy: Use collectors as tools")
        print("-" * 60)
        
        while True:
            try:
                text = input("\nEnter search request (or 'quit', 'screenshot'): ").strip()
                
                if text.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                save_screenshot = text.lower().startswith('screenshot')
                if save_screenshot:
                    text = text.replace('screenshot', '').strip()
                    print("Screenshot mode enabled!")
                
                if not text:
                    print("Please enter a search request")
                    continue
                
                # Handle Trump articles specially (CLAUDE.md: use specialized implementations)
                if 'trump' in text.lower():
                    fetch_trump_specialized(text)
                else:
                    urls, _ = smart_url_inference(text)
                    if urls:
                        fetch_with_fallback(text, urls, save_screenshot)
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
    else:
        # Single command mode
        text = " ".join(sys.argv[1:])
        save_screenshot = '--screenshot' in sys.argv
        text = text.replace('--screenshot', '').strip()
        
        # Handle Trump articles specially (CLAUDE.md: use specialized implementations)
        if 'trump' in text.lower():
            fetch_trump_specialized(text)
        else:
            urls, _ = smart_url_inference(text)
            if urls:
                fetch_with_fallback(text, urls, save_screenshot)


if __name__ == "__main__":
    main()