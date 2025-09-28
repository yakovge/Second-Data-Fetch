#!/usr/bin/env python3
"""
Example usage of the DataFetch Abstraction System.

This script demonstrates how to use the core components of the system
to fetch and process data from news websites.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.datafetch import FetchSpec, DataFormat, FetchMethod
from src.spec.parser import RawTextParser, URLManager, StructureDefinition
from src.collectors.http_client import HTTPClient


def main():
    """Main demonstration function."""
    print("[DEMO] DataFetch Abstraction System - Example Usage")
    print("=" * 60)
    
    # Example 1: Parse user requirements
    print("\n[DEMO] Example 1: Parsing User Requirements")
    print("-" * 40)
    
    raw_text = """
    I need to fetch breaking news articles from Reuters about technology.
    Each article should include the headline, full content, author, and publication date.
    The data should be structured as JSON for easy processing.
    Source: https://reuters.com/technology/
    """
    
    parser = RawTextParser()
    parsed_spec = parser.parse(raw_text)
    
    print(f"Raw text: {raw_text.strip()}")
    print(f"Extracted URLs: {parsed_spec.extracted_urls}")
    print(f"Suggested format: {parsed_spec.suggested_format}")
    print(f"Suggested method: {parsed_spec.suggested_method}")
    print(f"Confidence score: {parsed_spec.confidence_score:.2f}")
    
    # Example 2: Create and validate FetchSpec
    print("\n[ICON] Example 2: Creating FetchSpec")
    print("-" * 40)
    
    spec = FetchSpec(
        raw_text=raw_text.strip(),
        urls=["https://httpbin.org/json"],  # Using httpbin for demo
        expected_format=DataFormat.JSON,
        method=FetchMethod.REQUESTS,
        timeout=10,
        retry_count=2
    )
    
    print(f"Created FetchSpec:")
    print(f"  URLs: {spec.urls}")
    print(f"  Format: {spec.expected_format}")
    print(f"  Method: {spec.method}")
    print(f"  Timeout: {spec.timeout}s")
    print(f"  Retry count: {spec.retry_count}")
    
    # Example 3: Proper DataFetch Implementation (CLAUDE.md Pattern)
    print("\n[ICON] Example 3: Proper DataFetch Implementation")
    print("-" * 40)
    
    print("CLAUDE.md Pattern: Create custom DataFetch subclass")
    print("""
class NewsArticleFetch(DataFetch):
    def __init__(self, spec, cache_client=None):
        super().__init__(spec, cache_client)
        self.http_client = HTTPClient(spec, cache_client)
    
    def fetch(self) -> FetchResult:
        return self.http_client.fetch()
    
    # Implement other required abstract methods...
    """)
    
    print("For a working example, see: fetch_articles.py")
    print("This demonstrates:")
    print("  [ICON] Uses HTTPClient and BrowserClient as tools")
    print("  [ICON] Intelligent article ranking and relevance scoring")
    print("  [ICON] AI-enhanced URL discovery and structure inference")
    print("  [ICON] Real-world news article fetching with ranking")
    
    # Example 4: URL Management
    print("\n[ICON] Example 4: URL Management")
    print("-" * 40)
    
    url_manager = URLManager()
    
    test_urls = [
        "https://example.com/valid",
        "http://localhost:8080/invalid",  # Should be flagged as unsafe
        "https://reuters.com/news",
        "ftp://invalid.com",  # Wrong protocol
    ]
    
    validation_results = url_manager.validate_urls(test_urls)
    
    print("URL validation results:")
    for url, is_valid, error in validation_results:
        status = "VALID" if is_valid else "INVALID"
        error_info = f" ({error})" if error else ""
        print(f"  {url}: {status}{error_info}")
    
    # Example 5: Structure Definition
    print("\n[ICON] Example 5: Data Structure Definition")
    print("-" * 40)
    
    struct_def = StructureDefinition()
    
    sample_structure = {
        "article": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "content": {"type": "string"},
                "author": {"type": "string"}
            },
            "required": ["title", "content"]
        }
    }
    
    schema = struct_def.generate_schema(sample_structure)
    print(f"Generated JSON schema:")
    print(f"  Type: {schema.get('type')}")
    print(f"  Properties: {list(schema.get('properties', {}).keys())}")
    print(f"  Required: {schema.get('required', [])}")
    
    # Validate sample data
    sample_data = {
        "title": "Test Article",
        "content": "This is test content",
        "author": "Test Author"
    }
    
    is_valid, errors = struct_def.validate_data_against_schema(sample_data, schema)
    print(f"Sample data validation: {'VALID' if is_valid else 'INVALID'}")
    if errors:
        print(f"  Errors: {errors}")
    
    print("\n[SUCCESS] Example demonstration complete!")
    print("To run with full functionality, install dependencies with:")
    print("  pip install -r requirements.txt")


if __name__ == "__main__":
    main()