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
    print("🚀 DataFetch Abstraction System - Example Usage")
    print("=" * 60)
    
    # Example 1: Parse user requirements
    print("\n📝 Example 1: Parsing User Requirements")
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
    print("\n🔧 Example 2: Creating FetchSpec")
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
    
    # Example 3: Fetch data (without external dependencies)
    print("\n🌐 Example 3: Data Fetching Simulation")
    print("-" * 40)
    
    try:
        # This would work with installed dependencies
        client = HTTPClient(spec)
        print(f"HTTPClient created successfully")
        print(f"Session ID: {client.session_id}")
        print(f"Configuration: {type(client.config).__name__}")
        
        # Simulate successful result
        from datetime import datetime
        from src.core.datafetch import FetchResult
        
        mock_result = FetchResult(
            url=spec.urls[0],
            data={
                "slideshow": {
                    "author": "Yours Truly",
                    "date": "date of publication",
                    "title": "Sample Slide Show"
                }
            },
            timestamp=datetime.now(),
            format=DataFormat.JSON,
            method=FetchMethod.REQUESTS,
            metadata={"status_code": 200, "mock": True},
            cache_hit=False,
            execution_time=0.5
        )
        
        print(f"Mock fetch result:")
        print(f"  URL: {mock_result.url}")
        print(f"  Data keys: {list(mock_result.data.keys())}")
        print(f"  Status: SUCCESS")
        print(f"  Execution time: {mock_result.execution_time}s")
        print(f"  Cache hit: {mock_result.cache_hit}")
        
    except ImportError as e:
        print(f"Note: Full functionality requires dependencies: {e}")
    
    # Example 4: URL Management
    print("\n🔗 Example 4: URL Management")
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
    print("\n📊 Example 5: Data Structure Definition")
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
    
    print("\n✅ Example demonstration complete!")
    print("To run with full functionality, install dependencies with:")
    print("  pip install -r requirements.txt")


if __name__ == "__main__":
    main()