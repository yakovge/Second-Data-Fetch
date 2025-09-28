#!/usr/bin/env python3
"""
Simple script to fetch articles using raw text input.
Follows CLAUDE.md strategy: RawTextParser -> HTTPClient -> Results
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file if available
except ImportError:
    pass  # dotenv not available, continue without it

from src.spec.parser import RawTextParser
from src.collectors.http_client import HTTPClient
from src.collectors.browser_client import BrowserClient
from src.core.datafetch import FetchSpec, FetchMethod


def fetch_articles(query):
    """
    Fetch articles using raw text query.
    Uses AI for URL and structure discovery when not explicitly provided.

    Args:
        query (str): Raw text description like "articles about war in the NYT"

    Returns:
        dict: Results containing articles and metadata
    """
    print(f"Processing query: '{query}'")
    print("-" * 50)

    # Step 1: Parse raw text using RawTextParser (CLAUDE.md strategy)
    parser = RawTextParser()
    spec = parser.parse(query)

    print("SUCCESS: Parsed specification:")
    print(f"  URL: {spec.extracted_urls[0] if spec.extracted_urls else 'No URL extracted'}")
    print(f"  Format: {spec.suggested_format}")
    print(f"  Method: {spec.suggested_method}")
    print(f"  Confidence: {spec.confidence_score:.2f}")

    # Step 2: Use AI for URL discovery if no URLs found (CLAUDE.md strategy)
    # Currently using Claude 3 Haiku for fast, efficient AI processing
    if not spec.extracted_urls or spec.confidence_score < 0.7:
        print("AI: Using Claude Haiku for URL and structure discovery...")
        try:
            from src.ai.claude_client import ClaudeClient

            ai_client = ClaudeClient()  # Uses Claude 3 Haiku by default

            # Generate URLs using AI
            ai_urls = ai_client.generate_urls_from_text(query)
            if ai_urls:
                print(f"  AI discovered URLs: {ai_urls}")

                # Prioritize regular HTML URLs first, then API endpoints as fallback
                prioritized_urls = []
                api_urls = []
                nyt_api_key = os.getenv('NYT_API_KEY')

                for url in ai_urls:
                    if 'api.nytimes.com' in url or '/api/' in url:
                        # Add API key if available for NYT API endpoints
                        if nyt_api_key and 'api.nytimes.com' in url:
                            separator = '&' if '?' in url else '?'
                            url = f"{url}{separator}api-key={nyt_api_key}"
                            print(f"  Added API key to: {url.split('api-key=')[0]}api-key=***")
                        # API endpoints go to separate list
                        api_urls.append(url)
                    else:
                        # Regular URLs go first (more likely to work without auth)
                        prioritized_urls.append(url)

                # Combine: regular URLs first, then API URLs as fallback
                final_urls = prioritized_urls + api_urls
                spec.extracted_urls = final_urls
                # Update confidence score
                spec.confidence_score = min(spec.confidence_score + 0.3, 1.0)
            else:
                print("  AI could not discover relevant URLs")

        except Exception as e:
            print(f"  AI discovery failed: {e}")
            print("  Falling back to rule-based parsing")

    print()

    # Step 3: Use HTTPClient to fetch (CLAUDE.md strategy - use provided clients)
    if not spec.extracted_urls:
        print("ERROR: No URLs available for fetching")
        return {
            'query': query,
            'total_articles': 0,
            'articles': [],
            'fetch_time': 0,
            'status': 'No URLs found'
        }

    # Create FetchSpec
    fetch_spec = FetchSpec(
        raw_text=spec.raw_text,
        urls=spec.extracted_urls,
        expected_format=spec.suggested_format,
        method=spec.suggested_method,
        timeout=30,
        retry_count=3
    )

    # Choose client based on suggested method
    if spec.suggested_method == FetchMethod.PLAYWRIGHT:
        print("  Using Playwright for dynamic content extraction...")
        client = BrowserClient(fetch_spec)
    else:
        print("  Using HTTP client for static content...")
        client = HTTPClient(fetch_spec)

    result = client.fetch()

    print("SUCCESS: Fetch completed:")
    print(f"  Status: {'Success' if not result.error else 'Failed'}")
    print(f"  Items found: {len(result.data) if isinstance(result.data, (list, tuple)) else 1}")
    print(f"  Fetch time: {result.execution_time:.2f}s")
    print()

    # Step 3: Display results
    if result.data:
        print("ARTICLES: Found articles:")
        print("=" * 50)

        # Handle different data types
        if isinstance(result.data, (list, tuple)):
            articles_to_show = result.data[:5]  # Show first 5
            for i, article in enumerate(articles_to_show, 1):
                if isinstance(article, dict):
                    print(f"{i}. {article.get('title', 'No title')}")
                    if article.get('summary'):
                        print(f"   Summary: {article['summary'][:100]}...")
                    if article.get('url'):
                        print(f"   URL: {article['url']}")
                else:
                    print(f"{i}. {str(article)[:100]}...")
                print()
        elif isinstance(result.data, dict):
            # Single article or structured data
            print(f"1. {result.data.get('title', 'No title')}")
            if result.data.get('content'):
                print(f"   Content: {str(result.data['content'])[:200]}...")
        else:
            # Raw data
            print(f"Raw data: {str(result.data)[:300]}...")
    else:
        print("No articles found.")

    return {
        'query': query,
        'total_articles': len(result.data) if isinstance(result.data, (list, tuple)) else 1,
        'articles': result.data,
        'fetch_time': result.execution_time,
        'status': 'Success' if not result.error else 'Failed'
    }


def main():
    """Main function - can be called from command line or imported."""

    # Default query if none provided
    default_query = "articles about war in the NYT"

    # Get query from command line or use default
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = default_query
        print(f"Using default query: '{query}'")
        print("(You can provide a custom query as command line arguments)")
        print()

    try:
        results = fetch_articles(query)

        print("SUMMARY:")
        print(f"  Query: '{results['query']}'")
        print(f"  Found: {results['total_articles']} articles")
        print(f"  Time: {results['fetch_time']:.2f}s")
        print(f"  Status: {results['status']}")

        return results

    except Exception as e:
        print(f"ERROR: {e}")
        print("Make sure your environment is set up correctly:")
        print("  venv\\Scripts\\activate")
        print("  pip install -r requirements.txt")
        return None


if __name__ == "__main__":
    main()