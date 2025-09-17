import pytest
import json
from unittest.mock import Mock, patch, MagicMock
import responses
import aiohttp
from datetime import datetime, timedelta

from src.collectors.http_client import HTTPClient, RequestConfig
from src.core.datafetch import FetchSpec, FetchResult, DataFormat, FetchMethod, FetchError, ValidationError


class TestHTTPClient:
    """Test cases for HTTPClient implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.spec = FetchSpec(
            raw_text="Fetch news articles from example.com",
            urls=["https://example.com/api/news"],
            expected_format=DataFormat.JSON,
            method=FetchMethod.REQUESTS,
            timeout=30,
            retry_count=2
        )
        
        self.mock_cache = Mock()
        self.mock_storage = Mock()
        self.mock_ai = Mock()
    
    def test_init_basic(self):
        """Test basic initialization."""
        client = HTTPClient(self.spec)
        
        assert client._spec == self.spec
        assert isinstance(client.config, RequestConfig)
        assert client.session is None  # Lazy initialization
    
    def test_init_with_dependencies(self):
        """Test initialization with dependencies."""
        client = HTTPClient(self.spec, self.mock_cache, self.mock_storage, self.mock_ai)
        
        assert client._cache_client == self.mock_cache
        assert client._storage_client == self.mock_storage
        assert client._ai_client == self.mock_ai
    
    def test_config_creation_news_site(self):
        """Test configuration creation for news sites."""
        news_spec = FetchSpec(
            raw_text="Get news from Reuters",
            urls=["https://reuters.com/api/news"],
            expected_format=DataFormat.JSON,
            method=FetchMethod.REQUESTS
        )
        
        client = HTTPClient(news_spec)
        
        # Should use news-specific headers
        assert "NewsBot" in client.config.headers['User-Agent']
        assert client.config.headers['Accept-Language'] == 'en-US,en;q=0.9'
    
    def test_config_creation_custom_headers(self):
        """Test configuration with custom headers."""
        spec_with_headers = FetchSpec(
            raw_text="Fetch data",
            urls=["https://example.com/api"],
            custom_headers={"Authorization": "Bearer token123"},
            expected_format=DataFormat.JSON,
            method=FetchMethod.REQUESTS
        )
        
        client = HTTPClient(spec_with_headers)
        
        assert client.config.headers["Authorization"] == "Bearer token123"
    
    @responses.activate
    def test_fetch_success_json(self):
        """Test successful JSON fetch."""
        responses.add(
            responses.GET,
            "https://example.com/api/news",
            json={"articles": [{"title": "Test Article", "content": "Test content"}]},
            status=200
        )
        
        client = HTTPClient(self.spec)
        result = client.fetch()
        
        assert isinstance(result, FetchResult)
        assert result.url == "https://example.com/api/news"
        assert result.data == {"articles": [{"title": "Test Article", "content": "Test content"}]}
        assert result.method == FetchMethod.REQUESTS
        assert result.error is None
        assert result.cache_hit is False
    
    @responses.activate
    def test_fetch_success_html(self):
        """Test successful HTML fetch with article extraction."""
        html_content = """
        <html>
        <head><title>Test Page</title></head>
        <body>
            <h1>Test Article Title</h1>
            <div class="article-body">
                <p>This is test content.</p>
                <p>More content here.</p>
            </div>
            <span class="author">John Doe</span>
            <time datetime="2024-01-01T12:00:00Z">2024-01-01</time>
        </body>
        </html>
        """
        
        html_spec = FetchSpec(
            raw_text="Get HTML article",
            urls=["https://example.com/article"],
            expected_format=DataFormat.HTML,
            method=FetchMethod.REQUESTS
        )
        
        responses.add(
            responses.GET,
            "https://example.com/article",
            body=html_content,
            status=200,
            content_type="text/html"
        )
        
        client = HTTPClient(html_spec)
        result = client.fetch()
        
        assert isinstance(result, FetchResult)
        assert isinstance(result.data, dict)
        assert "title" in result.data
        assert "content" in result.data
        assert result.error is None
    
    @responses.activate
    def test_fetch_with_retry(self):
        """Test fetch with retry on failure."""
        # First request fails, second succeeds
        responses.add(
            responses.GET,
            "https://example.com/api/news",
            json={"error": "server error"},
            status=500
        )
        responses.add(
            responses.GET,
            "https://example.com/api/news",
            json={"articles": []},
            status=200
        )
        
        client = HTTPClient(self.spec)
        result = client.fetch()
        
        assert result.error is None
        assert result.data == {"articles": []}
        # Should have made 2 requests (1 failure + 1 success)
        assert len(responses.calls) == 2
    
    @responses.activate
    def test_fetch_all_retries_fail(self):
        """Test fetch when all retries fail."""
        # All requests fail
        for _ in range(self.spec.retry_count + 1):
            responses.add(
                responses.GET,
                "https://example.com/api/news",
                json={"error": "server error"},
                status=500
            )
        
        client = HTTPClient(self.spec)
        result = client.fetch()
        
        assert result.error is not None
        assert "Failed to fetch" in result.error
        assert len(responses.calls) == self.spec.retry_count + 1
    
    @responses.activate
    def test_fetch_with_cache_hit(self):
        """Test fetch with cache hit."""
        cached_result = FetchResult(
            url="https://example.com/api/news",
            data={"cached": "data"},
            timestamp=datetime.now(),
            format=DataFormat.JSON,
            method=FetchMethod.REQUESTS,
            metadata={},
            cache_hit=True
        )
        
        self.mock_cache.get.return_value = b'cached_data'  # Simulate cache hit
        
        with patch('pickle.loads', return_value=cached_result):
            client = HTTPClient(self.spec, self.mock_cache)
            result = client.fetch()
        
        assert result.cache_hit is True
        assert result.data == {"cached": "data"}
        assert len(responses.calls) == 0  # No HTTP request made
    
    @responses.activate
    def test_fetch_with_cache_store(self):
        """Test fetch with cache storage."""
        responses.add(
            responses.GET,
            "https://example.com/api/news",
            json={"articles": []},
            status=200
        )
        
        self.mock_cache.get.return_value = None  # Cache miss
        self.mock_cache.setex.return_value = True
        
        client = HTTPClient(self.spec, self.mock_cache)
        result = client.fetch()
        
        assert result.error is None
        self.mock_cache.setex.assert_called_once()  # Should store in cache
    
    def test_process_response_json(self):
        """Test JSON response processing."""
        client = HTTPClient(self.spec)
        
        json_data = {"test": "data"}
        content = json.dumps(json_data).encode('utf-8')
        
        result = client._process_response_content(content, "application/json", 200)
        assert result == json_data
    
    def test_process_response_xml(self):
        """Test XML response processing."""
        xml_spec = FetchSpec(
            raw_text="Get XML data",
            urls=["https://example.com/api.xml"],
            expected_format=DataFormat.XML,
            method=FetchMethod.REQUESTS
        )
        
        client = HTTPClient(xml_spec)
        
        xml_content = """<?xml version="1.0"?>
        <root>
            <item id="1">
                <title>Test Item</title>
                <description>Test description</description>
            </item>
        </root>"""
        
        result = client._process_response_content(xml_content.encode('utf-8'), "application/xml", 200)
        
        assert isinstance(result, dict)
        # The XML parser processes the root element's children
        assert "item" in result
        assert result["item"]["@attributes"]["id"] == "1"
        assert result["item"]["title"] == "Test Item"
    
    def test_extract_json_from_text(self):
        """Test JSON extraction from HTML/text."""
        client = HTTPClient(self.spec)
        
        html_with_json_ld = """
        <html>
        <head>
            <script type="application/ld+json">
            {
                "@context": "https://schema.org",
                "@type": "Article",
                "headline": "Test Article",
                "author": "John Doe"
            }
            </script>
        </head>
        <body>Content</body>
        </html>
        """
        
        result = client._extract_json_from_text(html_with_json_ld)
        
        assert isinstance(result, dict)
        assert result["@type"] == "Article"
        assert result["headline"] == "Test Article"
    
    def test_validate_data_basic(self):
        """Test basic data validation."""
        client = HTTPClient(self.spec)
        
        # Without validation rules, all data is considered valid
        assert client.validate_data({"test": "data"}) is True
        assert client.validate_data("text") is True
        assert client.validate_data([1, 2, 3]) is True
        assert client.validate_data(None) is True  # No rules = always valid
        assert client.validate_data("") is True
        assert client.validate_data({}) is True
        assert client.validate_data([]) is True
    
    def test_extract_structure(self):
        """Test structure extraction from sample data."""
        client = HTTPClient(self.spec)
        
        sample_data = {
            "title": "Test Article",
            "author": "John Doe",
            "published": True,
            "views": 42,
            "tags": ["news", "test"]
        }
        
        structure = client.extract_structure(sample_data)
        
        assert structure["title"]["type"] == "string"
        assert structure["published"]["type"] == "boolean"
        assert structure["views"]["type"] == "integer"
        assert structure["tags"]["type"] == "array"
    
    def test_sanitize_input(self):
        """Test input sanitization."""
        client = HTTPClient(self.spec)
        
        # Basic sanitization - check that dangerous tags are escaped
        result1 = client.sanitize_input("<script>alert('xss')</script>")
        assert "&lt;script&gt;" in result1 and "&lt;/script&gt;" in result1
        
        result2 = client.sanitize_input("javascript:alert('xss')")
        assert "alert(&#x27;xss&#x27;)" in result2  # javascript: prefix removed, quotes escaped
        
        result3 = client.sanitize_input('<img onload="alert(1)">')
        assert "&lt;img" in result3  # HTML tags escaped
    
    def test_cache_key_generation(self):
        """Test cache key generation."""
        client = HTTPClient(self.spec)
        
        url = "https://example.com/test"
        key1 = client.get_cache_key(url)
        key2 = client.get_cache_key(url)
        
        assert key1 == key2  # Same URL should generate same key
        assert key1.startswith("datafetch:")
        assert len(key1) > 20  # Should be a reasonable length hash
    
    @responses.activate
    def test_metrics_collection(self):
        """Test metrics collection during fetch."""
        responses.add(
            responses.GET,
            "https://example.com/api/news",
            json={"test": "data"},
            status=200
        )
        
        client = HTTPClient(self.spec)
        client.fetch()
        
        metrics = client.get_metrics()
        assert "total_execution_time" in metrics
        assert "urls_fetched" in metrics
        assert metrics["urls_fetched"] == 1
        assert metrics["total_execution_time"] > 0
    
    @pytest.mark.asyncio
    async def test_async_fetch_success(self):
        """Test successful async fetch."""
        with patch('aiohttp.ClientSession') as mock_session:
            # Mock response
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.headers = {'content-type': 'application/json'}
            # Make read() return a coroutine that yields bytes
            async def mock_read():
                return json.dumps({"test": "data"}).encode()
            mock_response.read = mock_read
            mock_response.raise_for_status.return_value = None
            mock_response.url = "https://example.com/api/news"
            
            # Mock session context manager
            mock_session_instance = MagicMock()
            mock_session_instance.__aenter__.return_value = mock_session_instance
            mock_session_instance.__aexit__.return_value = None
            mock_session_instance.get.return_value.__aenter__.return_value = mock_response
            mock_session_instance.get.return_value.__aexit__.return_value = None
            mock_session.return_value = mock_session_instance
            
            client = HTTPClient(self.spec)
            result = await client.afetch()
            
            assert isinstance(result, FetchResult)
            assert result.data == {"test": "data"}
            assert result.error is None
    
    def test_session_management(self):
        """Test HTTP session management."""
        client = HTTPClient(self.spec)
        
        # Session should be created lazily
        assert client.session is None
        
        # Get session
        session1 = client._get_session()
        assert session1 is not None
        assert client.session is session1
        
        # Should reuse same session
        session2 = client._get_session()
        assert session1 is session2
    
    def test_cleanup(self):
        """Test resource cleanup."""
        client = HTTPClient(self.spec)
        
        # Create session
        session = client._get_session()
        assert session is not None
        
        # Test cleanup
        with patch.object(session, 'close') as mock_close:
            client.__del__()
            mock_close.assert_called_once()


class TestRequestConfig:
    """Test cases for RequestConfig dataclass."""
    
    def test_request_config_creation(self):
        """Test RequestConfig creation."""
        config = RequestConfig(
            headers={"User-Agent": "Test"},
            timeout=30,
            verify_ssl=True,
            allow_redirects=True,
            max_redirects=10,
            retry_count=3,
            backoff_factor=1.0
        )
        
        assert config.headers == {"User-Agent": "Test"}
        assert config.timeout == 30
        assert config.verify_ssl is True
        assert config.allow_redirects is True
        assert config.max_redirects == 10
        assert config.retry_count == 3
        assert config.backoff_factor == 1.0


class TestHTTPClientIntegration:
    """Integration tests for HTTPClient."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.spec = FetchSpec(
            raw_text="Fetch test data",
            urls=["https://httpbin.org/json"],  # Real endpoint for integration tests
            expected_format=DataFormat.JSON,
            method=FetchMethod.REQUESTS,
            timeout=10,
            retry_count=1
        )
    
    @pytest.mark.integration
    @responses.activate
    def test_real_http_request(self):
        """Test actual HTTP request (when integration tests are enabled)."""
        # Mock httpbin.org response
        responses.add(
            responses.GET,
            "https://httpbin.org/json",
            json={
                "slideshow": {
                    "author": "Yours Truly",
                    "date": "date of publication",
                    "title": "Sample Slide Show"
                }
            },
            status=200
        )
        
        client = HTTPClient(self.spec)
        result = client.fetch()
        
        assert result.error is None
        assert isinstance(result.data, dict)
        assert "slideshow" in result.data
        assert result.metadata["status_code"] == 200
    
    @responses.activate
    def test_streaming_fetch(self):
        """Test streaming fetch with multiple URLs."""
        multi_url_spec = FetchSpec(
            raw_text="Fetch from multiple sources",
            urls=[
                "https://example.com/api/1",
                "https://example.com/api/2",
                "https://example.com/api/3"
            ],
            expected_format=DataFormat.JSON,
            method=FetchMethod.REQUESTS
        )
        
        # Mock all endpoints
        for i, url in enumerate(multi_url_spec.urls):
            responses.add(
                responses.GET,
                url,
                json={"source": i + 1},
                status=200
            )
        
        client = HTTPClient(multi_url_spec)
        results = list(client.fetch_stream())
        
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.error is None
            assert result.data == {"source": i + 1}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])