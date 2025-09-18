"""
Pytest configuration and fixtures for the Data Fetch system tests.
"""

import asyncio
import pytest
import tempfile
import os
from unittest.mock import Mock, MagicMock
from datetime import datetime, timedelta

# Import project modules
from src.core.datafetch import FetchSpec, FetchResult, DataFormat, FetchMethod
from src.cache.redis_client import RedisClient, CacheConfig
from src.storage.s3_manager import S3Manager, StorageConfig
from src.ai.gemini_client import GeminiClient


# Test configuration
@pytest.fixture(scope="session")
def test_config():
    """Test configuration fixture."""
    return {
        "redis_host": "localhost",
        "redis_port": 6379,
        "s3_bucket": "test-datafetch-bucket",
        "gemini_api_key": "test-api-key",
        "test_timeout": 30
    }


# Core fixtures
@pytest.fixture
def sample_fetch_spec():
    """Sample FetchSpec for testing."""
    return FetchSpec(
        raw_text="Fetch news articles from nytimes.com",
        urls=["https://www.nytimes.com/api/news"],
        expected_format=DataFormat.JSON,
        method=FetchMethod.REQUESTS,
        timeout=30,
        retry_count=2
    )


@pytest.fixture
def sample_fetch_result():
    """Sample FetchResult for testing."""
    return FetchResult(
        url="https://www.nytimes.com/api/news",
        data={
            "articles": [
                {
                    "title": "Test Article",
                    "content": "This is test content.",
                    "author": "Test Author",
                    "published": "2024-01-01T12:00:00Z"
                }
            ]
        },
        timestamp=datetime.now(),
        format=DataFormat.JSON,
        method=FetchMethod.REQUESTS,
        metadata={"status_code": 200, "content_type": "application/json"},
        cache_hit=False,
        execution_time=1.5
    )


# Mock fixtures
@pytest.fixture
def mock_redis_client():
    """Mock Redis client for testing."""
    mock_client = Mock(spec=RedisClient)
    mock_client.get.return_value = None
    mock_client.set.return_value = True
    mock_client.delete.return_value = True
    mock_client.exists.return_value = False
    mock_client.health_check.return_value = True
    return mock_client


@pytest.fixture
def mock_s3_manager():
    """Mock S3 manager for testing."""
    mock_manager = Mock(spec=S3Manager)
    mock_manager.store_data.return_value = True
    mock_manager.retrieve_data.return_value = None
    mock_manager.delete_data.return_value = True
    mock_manager.exists.return_value = False
    return mock_manager


@pytest.fixture
def mock_gemini_client():
    """Mock Gemini AI client for testing."""
    mock_client = Mock(spec=GeminiClient)
    mock_client.generate_structure_from_sample.return_value = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "content": {"type": "string"}
        }
    }
    mock_client.generate_urls_from_text.return_value = [
        "https://www.nytimes.com/api/news"
    ]
    mock_client.generate_datafetch_implementation.return_value = """
class GeneratedDataFetch(DataFetch):
    def fetch(self):
        pass
"""
    return mock_client


# Real client fixtures (for integration tests)
@pytest.fixture
def redis_client(test_config):
    """Real Redis client for integration tests."""
    config = CacheConfig(
        host=test_config["redis_host"],
        port=test_config["redis_port"],
        db=15,  # Use test database
        key_prefix="test:datafetch:",
        default_ttl=60  # Short TTL for tests
    )
    
    client = RedisClient(config)
    
    # Check if Redis is available
    if not client.health_check():
        pytest.skip("Redis not available for integration tests")
    
    yield client
    
    # Cleanup
    try:
        client.clear_pattern("*")
        client.close()
    except Exception:
        pass


@pytest.fixture
def s3_manager(test_config):
    """Real S3 manager for integration tests."""
    config = StorageConfig(
        bucket_name=test_config["s3_bucket"],
        region="us-east-1",
        key_prefix="test/datafetch/",
        default_ttl_days=1  # Short TTL for tests
    )
    
    try:
        manager = S3Manager(config)
        
        # Try to create test bucket
        if not manager.create_bucket_if_not_exists():
            pytest.skip("S3 not available or bucket creation failed")
        
        yield manager
        
        # Cleanup test data
        try:
            keys = manager.list_keys(max_keys=100)
            for key in keys:
                manager.delete_data(key)
        except Exception:
            pass
            
    except Exception:
        pytest.skip("S3 not available for integration tests")


@pytest.fixture
def gemini_client(test_config):
    """Real Gemini client for integration tests."""
    api_key = os.getenv('GEMINI_API_KEY') or test_config.get("gemini_api_key")
    
    if not api_key or api_key == "test-api-key":
        pytest.skip("GEMINI_API_KEY not available for integration tests")
    
    try:
        client = GeminiClient(api_key=api_key)
        yield client
    except Exception:
        pytest.skip("Gemini API not available for integration tests")


# Async fixtures
@pytest.fixture
async def async_mock_redis():
    """Async mock Redis client."""
    mock_client = MagicMock()
    mock_client.aget.return_value = None
    mock_client.aset.return_value = True
    mock_client.adelete.return_value = True
    mock_client.aexists.return_value = False
    mock_client.ahealth_check.return_value = True
    return mock_client


@pytest.fixture
async def async_mock_s3():
    """Async mock S3 manager."""
    mock_manager = MagicMock()
    mock_manager.astore_data.return_value = True
    mock_manager.aretrieve_data.return_value = None
    mock_manager.adelete_data.return_value = True
    mock_manager.aexists.return_value = False
    return mock_manager


# Test data fixtures
@pytest.fixture
def sample_news_data():
    """Sample news data for testing."""
    return {
        "articles": [
            {
                "title": "Breaking: Major News Event",
                "content": "This is the full article content with important details.",
                "author": "Jane Reporter",
                "published": "2024-01-01T10:00:00Z",
                "url": "https://example.com/news/article-1",
                "category": "politics",
                "images": [
                    {
                        "url": "https://example.com/images/news1.jpg",
                        "caption": "Event photo",
                        "alt": "Photo from the news event"
                    }
                ]
            },
            {
                "title": "Technology Update: New Framework Released",
                "content": "Details about the new technology framework and its features.",
                "author": "Tech Writer",
                "published": "2024-01-01T14:30:00Z",
                "url": "https://example.com/news/article-2",
                "category": "technology"
            }
        ],
        "meta": {
            "total": 2,
            "page": 1,
            "last_updated": "2024-01-01T15:00:00Z"
        }
    }


@pytest.fixture
def sample_html_content():
    """Sample HTML content for testing."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test News Article</title>
        <meta property="og:title" content="Test News Article">
        <meta property="og:description" content="This is a test article">
        <meta property="article:author" content="Test Author">
        <meta property="article:published_time" content="2024-01-01T12:00:00Z">
        <script type="application/ld+json">
        {
            "@context": "https://schema.org",
            "@type": "Article",
            "headline": "Test News Article",
            "author": "Test Author",
            "datePublished": "2024-01-01T12:00:00Z"
        }
        </script>
    </head>
    <body>
        <h1 class="headline">Test News Article</h1>
        <div class="article-body">
            <p>This is the first paragraph of the article.</p>
            <p>This is the second paragraph with more content.</p>
        </div>
        <span class="author">By Test Author</span>
        <time datetime="2024-01-01T12:00:00Z">January 1, 2024</time>
    </body>
    </html>
    """


@pytest.fixture
def temp_directory():
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


# Pytest hooks
def pytest_configure(config):
    """Configure pytest with custom settings."""
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on patterns."""
    for item in items:
        # Add unit marker to test files starting with test_
        if "test_" in item.fspath.basename:
            if "integration" not in item.fspath.basename:
                item.add_marker(pytest.mark.unit)
        
        # Add integration marker to integration test files
        if "integration" in item.fspath.basename:
            item.add_marker(pytest.mark.integration)
        
        # Add slow marker to tests that likely take time
        if any(keyword in item.name.lower() for keyword in ['fetch', 'download', 'upload', 'ai', 'llm']):
            item.add_marker(pytest.mark.slow)


# Async test support
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Skip markers for missing dependencies
def skip_if_no_redis():
    """Skip test if Redis is not available."""
    try:
        import redis
        client = redis.Redis(host='localhost', port=6379, db=0)
        client.ping()
        return False
    except Exception:
        return True


def skip_if_no_s3():
    """Skip test if AWS S3 is not available."""
    try:
        import boto3
        # Try to get AWS credentials
        session = boto3.Session()
        credentials = session.get_credentials()
        return credentials is None
    except Exception:
        return True


def skip_if_no_playwright():
    """Skip test if Playwright is not available."""
    try:
        from playwright.sync_api import sync_playwright
        return False
    except ImportError:
        return True


# Environment setup
@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Setup test environment variables."""
    # Set test environment variables
    monkeypatch.setenv("DATAFETCH_ENV", "test")
    monkeypatch.setenv("DATAFETCH_LOG_LEVEL", "DEBUG")
    
    # Mock sensitive environment variables for tests
    if not os.getenv("GEMINI_API_KEY"):
        monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")
    
    if not os.getenv("AWS_ACCESS_KEY_ID"):
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access-key")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-key")