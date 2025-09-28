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
import re


def score_article_relevance(article, query_keywords):
    """
    Score article relevance based on title, summary, and keyword matches.

    Args:
        article (dict): Article with title, summary, etc.
        query_keywords (list): Keywords extracted from user query

    Returns:
        float: Relevance score (higher = more relevant)
    """
    score = 0.0

    # Get article text fields
    title = article.get('title', '').lower()
    summary = article.get('summary', '').lower()
    url = article.get('url', '').lower()

    # Extract keywords from query (simple approach)
    if not query_keywords:
        return 1.0  # Default score if no keywords

    for keyword in query_keywords:
        keyword = keyword.lower()

        # Title matches are most important (3x weight)
        if keyword in title:
            score += 3.0

        # Summary matches are moderately important (2x weight)
        if keyword in summary:
            score += 2.0

        # URL matches are less important (1x weight)
        if keyword in url:
            score += 1.0

    # Boost recent articles (if URL contains 2024/2025)
    if '2024' in url or '2025' in url:
        score += 0.5

    # Boost articles with longer titles (more descriptive)
    if len(title) > 50:
        score += 0.3

    # Boost articles with summaries
    if summary and len(summary) > 20:
        score += 0.2

    return score


def extract_keywords_from_query(query):
    """Extract meaningful keywords from user query."""
    # Remove common stop words and extract meaningful terms
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'about', 'articles', 'news'}

    # Simple tokenization and filtering
    words = re.findall(r'\b\w+\b', query.lower())
    keywords = [word for word in words if len(word) > 2 and word not in stop_words]

    return keywords


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

    # Fetch from all URLs to get diverse content
    print(f"  Fetching from {len(spec.extracted_urls)} URLs for maximum diversity...")
    all_articles = []
    total_fetch_time = 0

    for i, url in enumerate(spec.extracted_urls, 1):
        print(f"  Processing URL {i}/{len(spec.extracted_urls)}: {url}")

        # Create individual fetch spec for each URL
        individual_spec = FetchSpec(
            raw_text=spec.raw_text,
            urls=[url],  # Single URL
            expected_format=spec.suggested_format,
            method=spec.suggested_method,
            timeout=30,
            retry_count=3
        )

        # Choose client based on suggested method
        if spec.suggested_method == FetchMethod.PLAYWRIGHT:
            client = BrowserClient(individual_spec)
        else:
            client = HTTPClient(individual_spec)

        try:
            url_result = client.fetch()
            total_fetch_time += url_result.execution_time

            if not url_result.error and url_result.data:
                if isinstance(url_result.data, (list, tuple)):
                    all_articles.extend(url_result.data)
                else:
                    all_articles.append(url_result.data)

        except Exception as e:
            print(f"    Warning: Failed to fetch from {url}: {e}")
            continue

    # Remove duplicates by URL while preserving order
    seen_urls = set()
    unique_articles = []
    for article in all_articles:
        if isinstance(article, dict):
            article_url = article.get('url', '')
            if article_url and article_url not in seen_urls:
                seen_urls.add(article_url)
                unique_articles.append(article)
        else:
            unique_articles.append(article)

    # Create combined result
    result = type('Result', (), {
        'data': unique_articles,
        'execution_time': total_fetch_time,
        'error': None
    })()

    print("SUCCESS: Fetch completed:")
    print(f"  Status: {'Success' if not result.error else 'Failed'}")
    print(f"  Items found: {len(result.data) if isinstance(result.data, (list, tuple)) else 1}")
    print(f"  Fetch time: {result.execution_time:.2f}s")
    print()

    # Step 3: Display results
    if result.data:
        print("ARTICLES: Found articles (ranked by relevance):")
        print("=" * 50)

        # Handle different data types
        if isinstance(result.data, (list, tuple)):
            # Extract keywords from query for ranking
            query_keywords = extract_keywords_from_query(query)
            print(f"Ranking keywords: {', '.join(query_keywords)}")
            print()

            # Score and sort articles by relevance
            scored_articles = []
            for article in result.data:
                if isinstance(article, dict):
                    score = score_article_relevance(article, query_keywords)
                    scored_articles.append((score, article))
                else:
                    scored_articles.append((0.0, article))

            # Sort by score (highest first) and select top 10 most relevant
            scored_articles.sort(key=lambda x: x[0], reverse=True)

            # Filter for only relevant articles (score > 0) and take top 10
            relevant_articles = [(score, article) for score, article in scored_articles if score > 0]

            # If we have more than 10 relevant articles, take the best 10
            # If we have fewer than 10 relevant articles, supplement with highest-scoring others
            if len(relevant_articles) >= 10:
                articles_to_show = relevant_articles[:10]
                print(f"Showing top 10 most relevant articles (from {len(relevant_articles)} relevant, {len(scored_articles)} total)")
            else:
                articles_to_show = scored_articles[:10]  # Take top 10 regardless of relevance
                print(f"Showing top 10 articles ({len(relevant_articles)} highly relevant, {len(scored_articles)} total)")
            print()
            for i, (score, article) in enumerate(articles_to_show, 1):
                if isinstance(article, dict):
                    print(f"{i}. {article.get('title', 'No title')} [Score: {score:.1f}]")
                    if article.get('summary'):
                        print(f"   Summary: {article['summary'][:100]}...")
                    if article.get('url'):
                        print(f"   URL: {article['url']}")
                else:
                    print(f"{i}. {str(article)[:100]}... [Score: {score:.1f}]")
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