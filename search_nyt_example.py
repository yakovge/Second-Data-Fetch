#!/usr/bin/env python3
"""
Example: How to search for NYT data using the DataFetch system.
Shows both generic and specialized approaches.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.datafetch import FetchSpec, DataFormat, FetchMethod
from src.collectors.http_client import HTTPClient
from src.implementations.nyt_trump_fetch import NYTTrumpFetch


def search_nyt_general(query_topic: str = "technology"):
    """
    Example 1: Generic NYT search using HTTPClient
    The system automatically recognizes NYT and optimizes for news content.
    """
    print(f"🔍 Searching NYT for: {query_topic}")
    print("-" * 50)
    
    # Create FetchSpec - HTTPClient will auto-detect NYT and optimize
    spec = FetchSpec(
        raw_text=f"Get latest {query_topic} articles from New York Times",
        urls=[
            f"https://www.nytimes.com/section/{query_topic}",  # NYT section
            "https://www.nytimes.com/",  # Main page
        ],
        expected_format=DataFormat.HTML,  # NYT serves HTML
        method=FetchMethod.REQUESTS,
        timeout=30,
        retry_count=2
    )
    
    # HTTPClient automatically applies NYT-specific parsing
    client = HTTPClient(spec)
    result = client.fetch()
    
    print(f"✅ Fetch Status: {'Success' if not result.error else 'Error'}")
    print(f"📊 Data Type: {type(result.data)}")
    print(f"🌐 URL Fetched: {result.url}")
    print(f"⏱️  Execution Time: {result.execution_time:.2f}s")
    
    if result.error:
        print(f"❌ Error: {result.error}")
    elif isinstance(result.data, dict):
        print(f"📖 Articles Found: {len(result.data.get('articles', []))}")
    
    return result


def search_nyt_trump_specialized():
    """
    Example 2: Specialized Trump search using NYTTrumpFetch
    Custom DataFetch implementation specifically for Trump articles.
    """
    print(f"🎯 Specialized Trump Article Search")
    print("-" * 50)
    
    # Create spec for Trump articles
    spec = FetchSpec(
        raw_text="Find articles about Trump from New York Times with headlines and content",
        urls=[
            "https://www.nytimes.com/",
            "https://www.nytimes.com/section/politics"
        ],
        expected_format=DataFormat.HTML,
        method=FetchMethod.REQUESTS
    )
    
    # Use specialized NYTTrumpFetch implementation
    trump_fetcher = NYTTrumpFetch(spec)
    result = trump_fetcher.fetch()
    
    print(f"✅ Fetch Status: {'Success' if not result.error else 'Error'}")
    print(f"🎯 Extraction Method: {result.metadata.get('extraction_method')}")
    print(f"🔎 Keywords Searched: {result.metadata.get('keywords_searched')}")
    print(f"📰 Trump Articles Found: {result.metadata.get('articles_found')}")
    
    # Show sample articles
    if isinstance(result.data, dict) and 'articles' in result.data:
        articles = result.data['articles'][:3]  # Show first 3
        for i, article in enumerate(articles, 1):
            print(f"\n📄 Article {i}:")
            print(f"   Title: {article.get('title', 'N/A')[:80]}...")
            print(f"   Relevance: {article.get('relevance_score', 0):.2f}")
            print(f"   Keywords: {article.get('keywords_matched', [])}")
    
    return result


def search_nyt_custom_topic(topic: str):
    """
    Example 3: Create custom DataFetch for specific NYT topics
    Shows the CLAUDE.md pattern for extending the system.
    """
    print(f"🎨 Custom {topic} Search Pattern")
    print("-" * 50)
    
    print("CLAUDE.md Pattern - Create Custom DataFetch:")
    print(f"""
class NYT{topic.title()}Fetch(DataFetch):
    def __init__(self, spec):
        super().__init__(spec)
        self.http_client = HTTPClient(spec)  # Use as tool
    
    def fetch(self) -> FetchResult:
        raw_result = self.http_client.fetch()
        # Add {topic}-specific processing here
        return self._process_for_{topic.lower()}(raw_result)
    
    # Implement other abstract methods...
""")


def demonstrate_url_intelligence():
    """
    Example 4: Show how the system recognizes different news sources
    """
    print("🧠 URL Intelligence Demo")
    print("-" * 50)
    
    test_urls = [
        "https://www.nytimes.com/section/technology",
        "https://www.reuters.com/business/",
        "https://www.bbc.com/news",
        "https://edition.cnn.com/",
        "https://example.com/api"  # Non-news site
    ]
    
    for url in test_urls:
        spec = FetchSpec(
            raw_text=f"Test URL recognition for {url}",
            urls=[url],
            expected_format=DataFormat.HTML
        )
        
        client = HTTPClient(spec)
        is_news = any(
            news_domain in url.lower() 
            for news_domain in ['news', 'reuters', 'nytimes', 'bbc', 'cnn', 'guardian']
        )
        
        print(f"🌐 {url}")
        print(f"   📰 News Site: {'Yes' if is_news else 'No'}")
        print(f"   🔧 Will use: {'News headers & parsing' if is_news else 'Standard parsing'}")
        print()


if __name__ == "__main__":
    print("NYT Data Fetching Examples")
    print("=" * 60)
    
    try:
        # Example 1: General NYT search
        search_nyt_general("technology")
        print("\n" + "="*60 + "\n")
        
        # Example 2: Specialized Trump search
        search_nyt_trump_specialized() 
        print("\n" + "="*60 + "\n")
        
        # Example 3: Custom topic pattern
        search_nyt_custom_topic("climate")
        print("\n" + "="*60 + "\n")
        
        # Example 4: URL intelligence
        demonstrate_url_intelligence()
        
        print("✨ All examples completed!")
        print("\n🚀 To run with real data:")
        print("   python search_nyt_example.py")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("💡 Make sure dependencies are installed: pip install -r requirements.txt")