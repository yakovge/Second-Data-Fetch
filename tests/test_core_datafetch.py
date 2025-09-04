import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.core.datafetch import (
    DataFetch, FetchResult, FetchSpec, DataFormat, FetchMethod, 
    CacheStrategy, ValidationError, FetchError, SecurityError
)


class ConcreteDataFetch(DataFetch):
    """Concrete implementation for testing the abstract base class."""
    
    def fetch(self) -> FetchResult:
        return FetchResult(
            url=self._spec.urls[0],
            data={"test": "data"},
            timestamp=datetime.now(),
            format=self._spec.expected_format,
            method=self._spec.method,
            metadata={},
            cache_hit=False
        )
    
    async def afetch(self) -> FetchResult:
        return self.fetch()
    
    def fetch_stream(self):
        yield self.fetch()
    
    async def afetch_stream(self):
        yield self.fetch()
    
    def validate_data(self, data) -> bool:
        return data is not None
    
    def extract_structure(self, sample_data):
        return {"type": "object"}
    
    def sanitize_input(self, raw_input: str) -> str:
        return raw_input.strip()


class TestDataFetch:
    """Test cases for the DataFetch abstract base class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.valid_spec = FetchSpec(
            raw_text="Test news articles from example.com",
            urls=["https://example.com/news"],
            structure_definition=None,
            expected_format=DataFormat.JSON,
            method=FetchMethod.REQUESTS,
            cache_strategy=CacheStrategy.REDIS,
            cache_ttl=timedelta(hours=1),
            retry_count=3,
            timeout=30
        )
        
        self.mock_cache = Mock()
        self.mock_storage = Mock()
        self.mock_ai = Mock()
    
    def test_init_valid_spec(self):
        """Test initialization with valid spec."""
        datafetch = ConcreteDataFetch(self.valid_spec)
        
        assert datafetch.spec == self.valid_spec
        assert datafetch.session_id is not None
        assert len(datafetch.session_id) > 0
    
    def test_init_with_clients(self):
        """Test initialization with client dependencies."""
        datafetch = ConcreteDataFetch(
            self.valid_spec, 
            self.mock_cache, 
            self.mock_storage, 
            self.mock_ai
        )
        
        assert datafetch._cache_client == self.mock_cache
        assert datafetch._storage_client == self.mock_storage
        assert datafetch._ai_client == self.mock_ai
    
    def test_spec_validation_empty_text(self):
        """Test spec validation fails with empty raw text."""
        invalid_spec = FetchSpec(
            raw_text="",
            urls=["https://example.com"],
            expected_format=DataFormat.JSON,
            method=FetchMethod.REQUESTS
        )
        
        with pytest.raises(ValidationError, match="raw_text cannot be empty"):
            ConcreteDataFetch(invalid_spec)
    
    def test_spec_validation_no_urls(self):
        """Test spec validation fails with no URLs."""
        invalid_spec = FetchSpec(
            raw_text="Test data",
            urls=[],
            expected_format=DataFormat.JSON,
            method=FetchMethod.REQUESTS
        )
        
        with pytest.raises(ValidationError, match="At least one URL must be provided"):
            ConcreteDataFetch(invalid_spec)
    
    def test_spec_validation_unsafe_url(self):
        """Test spec validation fails with unsafe URL."""
        invalid_spec = FetchSpec(
            raw_text="Test data",
            urls=["http://localhost:8080/test"],
            expected_format=DataFormat.JSON,
            method=FetchMethod.REQUESTS
        )
        
        with pytest.raises(SecurityError, match="Unsafe URL detected"):
            ConcreteDataFetch(invalid_spec)
    
    def test_spec_validation_invalid_retry_count(self):
        """Test spec validation fails with invalid retry count."""
        invalid_spec = FetchSpec(
            raw_text="Test data",
            urls=["https://example.com"],
            retry_count=15,  # Too high
            expected_format=DataFormat.JSON,
            method=FetchMethod.REQUESTS
        )
        
        with pytest.raises(ValidationError, match="retry_count must be between 0 and 10"):
            ConcreteDataFetch(invalid_spec)
    
    def test_spec_validation_invalid_timeout(self):
        """Test spec validation fails with invalid timeout."""
        invalid_spec = FetchSpec(
            raw_text="Test data",
            urls=["https://example.com"],
            timeout=500,  # Too high
            expected_format=DataFormat.JSON,
            method=FetchMethod.REQUESTS
        )
        
        with pytest.raises(ValidationError, match="timeout must be between 1 and 300 seconds"):
            ConcreteDataFetch(invalid_spec)
    
    def test_is_safe_url_valid(self):
        """Test _is_safe_url with valid URLs."""
        datafetch = ConcreteDataFetch(self.valid_spec)
        
        valid_urls = [
            "https://example.com",
            "http://example.com",
            "https://news.example.com/articles",
            "https://api.example.com/data?param=value"
        ]
        
        for url in valid_urls:
            assert datafetch._is_safe_url(url), f"URL should be safe: {url}"
    
    def test_is_safe_url_invalid(self):
        """Test _is_safe_url with invalid URLs."""
        datafetch = ConcreteDataFetch(self.valid_spec)
        
        invalid_urls = [
            "ftp://example.com",
            "javascript:alert('xss')",
            "http://localhost:8080",
            "https://127.0.0.1:3000",
            "http://0.0.0.0",
            "https://example.com<script>",
            "not-a-url",
            ""
        ]
        
        for url in invalid_urls:
            assert not datafetch._is_safe_url(url), f"URL should be unsafe: {url}"
    
    def test_cache_key_generation(self):
        """Test cache key generation."""
        datafetch = ConcreteDataFetch(self.valid_spec)
        
        url = "https://example.com/test"
        cache_key = datafetch.get_cache_key(url)
        
        assert cache_key.startswith("datafetch:")
        assert len(cache_key) > len("datafetch:")
        
        # Same URL should generate same key
        cache_key2 = datafetch.get_cache_key(url)
        assert cache_key == cache_key2
    
    def test_metrics_recording(self):
        """Test metrics recording functionality."""
        datafetch = ConcreteDataFetch(self.valid_spec)
        
        # Initially empty
        metrics = datafetch.get_metrics()
        assert metrics == {}
        
        # Record some metrics
        datafetch._record_metric("test_metric", 42)
        datafetch._increment_metric("counter_metric", 5)
        datafetch._increment_metric("counter_metric", 3)
        
        metrics = datafetch.get_metrics()
        assert metrics["test_metric"] == 42
        assert metrics["counter_metric"] == 8
        
        # Reset metrics
        datafetch.reset_metrics()
        metrics = datafetch.get_metrics()
        assert metrics == {}
    
    def test_session_id_uniqueness(self):
        """Test that session IDs are unique across instances."""
        datafetch1 = ConcreteDataFetch(self.valid_spec)
        datafetch2 = ConcreteDataFetch(self.valid_spec)
        
        assert datafetch1.session_id != datafetch2.session_id
        
        # Session IDs should be valid UUIDs
        import uuid
        uuid.UUID(datafetch1.session_id)  # Should not raise
        uuid.UUID(datafetch2.session_id)  # Should not raise
    
    def test_spec_property_readonly(self):
        """Test that spec property is read-only."""
        datafetch = ConcreteDataFetch(self.valid_spec)
        
        # Should be able to read
        spec = datafetch.spec
        assert spec == self.valid_spec
        
        # Should not be able to modify (returns copy)
        spec.raw_text = "Modified text"
        assert datafetch.spec.raw_text == "Test news articles from example.com"


class TestFetchResult:
    """Test cases for FetchResult dataclass."""
    
    def test_fetch_result_creation(self):
        """Test FetchResult creation with all fields."""
        result = FetchResult(
            url="https://example.com",
            data={"test": "data"},
            timestamp=datetime.now(),
            format=DataFormat.JSON,
            method=FetchMethod.REQUESTS,
            metadata={"status": 200},
            cache_hit=False,
            execution_time=1.5,
            error=None
        )
        
        assert result.url == "https://example.com"
        assert result.data == {"test": "data"}
        assert result.format == DataFormat.JSON
        assert result.method == FetchMethod.REQUESTS
        assert result.cache_hit is False
        assert result.execution_time == 1.5
        assert result.error is None
    
    def test_fetch_result_defaults(self):
        """Test FetchResult creation with default values."""
        result = FetchResult(
            url="https://example.com",
            data={"test": "data"},
            timestamp=datetime.now(),
            format=DataFormat.JSON,
            method=FetchMethod.REQUESTS,
            metadata={}
        )
        
        assert result.cache_hit is False  # Default value
        assert result.execution_time == 0.0  # Default value
        assert result.error is None  # Default value


class TestFetchSpec:
    """Test cases for FetchSpec dataclass."""
    
    def test_fetch_spec_creation(self):
        """Test FetchSpec creation with all fields."""
        spec = FetchSpec(
            raw_text="Test description",
            urls=["https://example.com"],
            structure_definition={"type": "object"},
            expected_format=DataFormat.JSON,
            method=FetchMethod.REQUESTS,
            cache_strategy=CacheStrategy.REDIS,
            cache_ttl=timedelta(hours=2),
            retry_count=5,
            timeout=60,
            custom_headers={"User-Agent": "Test"},
            validation_rules={"required": ["title"]}
        )
        
        assert spec.raw_text == "Test description"
        assert spec.urls == ["https://example.com"]
        assert spec.structure_definition == {"type": "object"}
        assert spec.expected_format == DataFormat.JSON
        assert spec.method == FetchMethod.REQUESTS
        assert spec.cache_strategy == CacheStrategy.REDIS
        assert spec.cache_ttl == timedelta(hours=2)
        assert spec.retry_count == 5
        assert spec.timeout == 60
        assert spec.custom_headers == {"User-Agent": "Test"}
        assert spec.validation_rules == {"required": ["title"]}
    
    def test_fetch_spec_defaults(self):
        """Test FetchSpec creation with default values."""
        spec = FetchSpec(
            raw_text="Test description",
            urls=["https://example.com"]
        )
        
        assert spec.structure_definition is None
        assert spec.expected_format == DataFormat.JSON
        assert spec.method == FetchMethod.REQUESTS
        assert spec.cache_strategy == CacheStrategy.REDIS
        assert spec.cache_ttl == timedelta(hours=1)
        assert spec.retry_count == 3
        assert spec.timeout == 30
        assert spec.custom_headers is None
        assert spec.validation_rules is None


class TestEnums:
    """Test cases for enum classes."""
    
    def test_data_format_enum(self):
        """Test DataFormat enum values."""
        assert DataFormat.JSON.value == "json"
        assert DataFormat.XML.value == "xml"
        assert DataFormat.HTML.value == "html"
        assert DataFormat.TEXT.value == "text"
        assert DataFormat.CSV.value == "csv"
    
    def test_fetch_method_enum(self):
        """Test FetchMethod enum values."""
        assert FetchMethod.REQUESTS.value == "requests"
        assert FetchMethod.PLAYWRIGHT.value == "playwright"
        assert FetchMethod.HYBRID.value == "hybrid"
    
    def test_cache_strategy_enum(self):
        """Test CacheStrategy enum values."""
        assert CacheStrategy.NO_CACHE.value == "no_cache"
        assert CacheStrategy.MEMORY.value == "memory"
        assert CacheStrategy.REDIS.value == "redis"
        assert CacheStrategy.S3.value == "s3"
        assert CacheStrategy.HYBRID.value == "hybrid"


class TestExceptions:
    """Test cases for custom exceptions."""
    
    def test_validation_error(self):
        """Test ValidationError exception."""
        with pytest.raises(ValidationError):
            raise ValidationError("Validation failed")
    
    def test_fetch_error(self):
        """Test FetchError exception."""
        with pytest.raises(FetchError):
            raise FetchError("Fetch operation failed")
    
    def test_security_error(self):
        """Test SecurityError exception."""
        with pytest.raises(SecurityError):
            raise SecurityError("Security check failed")


# Integration tests
class TestDataFetchIntegration:
    """Integration tests for DataFetch functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.spec = FetchSpec(
            raw_text="Fetch news from example.com",
            urls=["https://example.com/api/news"],
            expected_format=DataFormat.JSON,
            method=FetchMethod.REQUESTS
        )
    
    def test_basic_fetch_workflow(self):
        """Test basic fetch workflow."""
        datafetch = ConcreteDataFetch(self.spec)
        
        # Test sync fetch
        result = datafetch.fetch()
        assert isinstance(result, FetchResult)
        assert result.url == "https://example.com/api/news"
        assert result.data == {"test": "data"}
        assert result.method == FetchMethod.REQUESTS
        assert result.cache_hit is False
    
    @pytest.mark.asyncio
    async def test_async_fetch_workflow(self):
        """Test async fetch workflow."""
        datafetch = ConcreteDataFetch(self.spec)
        
        # Test async fetch
        result = await datafetch.afetch()
        assert isinstance(result, FetchResult)
        assert result.url == "https://example.com/api/news"
        assert result.data == {"test": "data"}
        assert result.method == FetchMethod.REQUESTS
    
    def test_streaming_fetch_workflow(self):
        """Test streaming fetch workflow."""
        datafetch = ConcreteDataFetch(self.spec)
        
        # Test sync streaming
        results = list(datafetch.fetch_stream())
        assert len(results) == 1
        assert isinstance(results[0], FetchResult)
    
    @pytest.mark.asyncio
    async def test_async_streaming_fetch_workflow(self):
        """Test async streaming fetch workflow."""
        datafetch = ConcreteDataFetch(self.spec)
        
        # Test async streaming
        results = []
        async for result in datafetch.afetch_stream():
            results.append(result)
        
        assert len(results) == 1
        assert isinstance(results[0], FetchResult)
    
    def test_data_validation(self):
        """Test data validation functionality."""
        datafetch = ConcreteDataFetch(self.spec)
        
        # Valid data
        assert datafetch.validate_data({"test": "data"}) is True
        assert datafetch.validate_data("text data") is True
        assert datafetch.validate_data([1, 2, 3]) is True
        
        # Invalid data
        assert datafetch.validate_data(None) is False
    
    def test_structure_extraction(self):
        """Test structure extraction functionality."""
        datafetch = ConcreteDataFetch(self.spec)
        
        sample_data = {"title": "Test", "content": "Content"}
        structure = datafetch.extract_structure(sample_data)
        
        assert isinstance(structure, dict)
        assert structure["type"] == "object"
    
    def test_input_sanitization(self):
        """Test input sanitization functionality."""
        datafetch = ConcreteDataFetch(self.spec)
        
        dirty_input = "  test input with whitespace  "
        clean_input = datafetch.sanitize_input(dirty_input)
        
        assert clean_input == "test input with whitespace"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])