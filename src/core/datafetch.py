from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, AsyncGenerator, Generator
from datetime import datetime, timedelta
from enum import Enum
import asyncio
from pydantic import BaseModel, Field, field_validator, ConfigDict


class FetchMethod(Enum):
    REQUESTS = "requests"
    PLAYWRIGHT = "playwright"
    HYBRID = "hybrid"


class DataFormat(Enum):
    JSON = "json"
    XML = "xml"
    HTML = "html"
    TEXT = "text"
    CSV = "csv"


class CacheStrategy(Enum):
    NO_CACHE = "no_cache"
    MEMORY = "memory"
    REDIS = "redis"
    S3 = "s3"
    HYBRID = "hybrid"


class FetchResult(BaseModel):
    url: str = Field(..., description="URL that was fetched")
    data: Any = Field(..., description="Fetched data")
    timestamp: datetime = Field(..., description="When the fetch occurred")
    format: DataFormat = Field(..., description="Format of the fetched data")
    method: FetchMethod = Field(..., description="Method used for fetching")
    metadata: Dict[str, Any] = Field(..., description="Additional metadata")
    cache_hit: bool = Field(False, description="Whether result came from cache")
    execution_time: float = Field(0.0, ge=0, description="Execution time in seconds")
    error: Optional[str] = Field(None, description="Error message if fetch failed")

    # Note: FetchResult is not frozen to allow post-creation updates of metrics


class FetchSpec(BaseModel):
    raw_text: str = Field(..., description="Raw user-provided text describing the fetch request")
    urls: List[str] = Field(..., min_length=1, description="List of URLs to fetch from")
    structure_definition: Optional[Dict[str, Any]] = Field(None, description="Expected data structure definition")
    expected_format: DataFormat = Field(DataFormat.JSON, description="Expected output format")
    method: FetchMethod = Field(FetchMethod.REQUESTS, description="Method to use for fetching")
    cache_strategy: CacheStrategy = Field(CacheStrategy.REDIS, description="Caching strategy")
    cache_ttl: timedelta = Field(timedelta(hours=1), description="Cache time-to-live")
    retry_count: int = Field(3, ge=0, le=10, description="Number of retry attempts")
    timeout: int = Field(30, gt=0, le=300, description="Timeout in seconds")
    custom_headers: Optional[Dict[str, str]] = Field(None, description="Custom HTTP headers")
    validation_rules: Optional[Dict[str, Any]] = Field(None, description="Data validation rules")

    model_config = ConfigDict(frozen=True)

    @field_validator('raw_text')
    @classmethod
    def validate_raw_text(cls, v):
        if not v.strip():
            raise ValueError("raw_text cannot be empty")
        return v


class ValidationError(Exception):
    pass


class FetchError(Exception):
    pass


class SecurityError(Exception):
    pass


class DataFetch(ABC):
    """
    Abstract base class for data fetching operations.

    This class is IMMUTABLE - AI agents can only implement it, not modify it.
    It provides a robust, secure, and efficient framework for fetching structured
    data from web sources, particularly news websites.

    Key Design Principles:
    1. Security First: All inputs are sanitized, no code injection possible
    2. Efficiency: Built-in caching and concurrent execution support
    3. Reliability: Comprehensive error handling and retry mechanisms
    4. Flexibility: Support for both sync/async, multiple fetch methods
    5. Observability: Detailed logging and metrics collection
    """

    # Allowed domains for news website fetching
    ALLOWED_DOMAINS = [
        'nytimes.com', 'www.nytimes.com',
        'reuters.com', 'www.reuters.com',
        'bbc.com', 'www.bbc.com', 'bbc.co.uk', 'www.bbc.co.uk',
        'cnn.com', 'www.cnn.com',
        'apnews.com', 'www.apnews.com',
        'wsj.com', 'www.wsj.com',
        'washingtonpost.com', 'www.washingtonpost.com',
        'theguardian.com', 'www.theguardian.com',
        # Test domains
        'httpbin.org', 'www.httpbin.org',
        'example.org', 'www.example.org',
        'jsonplaceholder.typicode.com'
    ]
    
    def __init__(self, 
                 spec: FetchSpec,
                 cache_client: Optional[Any] = None,
                 storage_client: Optional[Any] = None,
                 ai_client: Optional[Any] = None):
        """
        Initialize DataFetch with specification and optional clients.
        
        Args:
            spec: FetchSpec containing all fetch parameters
            cache_client: Redis or other caching client
            storage_client: S3 or other storage client  
            ai_client: Gemini or other AI client for structure generation
        """
        # Sanitize raw_text before validation
        if hasattr(spec, 'model_dump'):
            sanitized_spec_dict = spec.model_dump()
        elif hasattr(spec, 'dict'):
            sanitized_spec_dict = spec.dict()
        else:
            sanitized_spec_dict = spec.__dict__
        sanitized_spec_dict['raw_text'] = self.sanitize_input(sanitized_spec_dict['raw_text'])
        if isinstance(spec, FetchSpec):
            sanitized_spec = FetchSpec(**sanitized_spec_dict)
        else:
            sanitized_spec = spec
        self._spec = self._validate_spec(sanitized_spec)
        self._cache_client = cache_client
        self._storage_client = storage_client
        self._ai_client = ai_client
        self._session_id = self._generate_session_id()
        self._metrics = {}
    
    @property
    def spec(self) -> FetchSpec:
        """Read-only access to fetch specification."""
        return self._spec
    
    @property 
    def session_id(self) -> str:
        """Unique session identifier for this fetch operation."""
        return self._session_id
    
    @abstractmethod
    def fetch(self) -> FetchResult:
        """
        Synchronous fetch operation.
        
        Returns:
            FetchResult containing fetched data and metadata
            
        Raises:
            FetchError: If fetch operation fails
            ValidationError: If data doesn't match expected structure
            SecurityError: If security validation fails
        """
        pass
    
    @abstractmethod
    async def afetch(self) -> FetchResult:
        """
        Asynchronous fetch operation.
        
        Returns:
            FetchResult containing fetched data and metadata
            
        Raises:
            FetchError: If fetch operation fails
            ValidationError: If data doesn't match expected structure  
            SecurityError: If security validation fails
        """
        pass
    
    @abstractmethod
    def fetch_stream(self) -> Generator[FetchResult, None, None]:
        """
        Streaming fetch operation for large datasets.
        
        Yields:
            FetchResult objects as they become available
        """
        pass
    
    @abstractmethod  
    async def afetch_stream(self) -> AsyncGenerator[FetchResult, None]:
        """
        Asynchronous streaming fetch operation.
        
        Yields:
            FetchResult objects as they become available
        """
        pass
    
    @abstractmethod
    def validate_data(self, data: Any) -> bool:
        """
        Validate fetched data against expected structure.
        
        Args:
            data: Raw fetched data
            
        Returns:
            True if data is valid, False otherwise
            
        Raises:
            ValidationError: If validation rules are malformed
        """
        pass
    
    @abstractmethod
    def extract_structure(self, sample_data: Any) -> Dict[str, Any]:
        """
        Extract data structure from sample data using AI if available.
        
        Args:
            sample_data: Sample of fetched data
            
        Returns:
            Dictionary defining the data structure
            
        Raises:
            FetchError: If structure extraction fails
        """
        pass
    
    def sanitize_input(self, raw_input: str) -> str:
        """
        Sanitize user input to prevent prompt injection attacks.

        Args:
            raw_input: Raw user input

        Returns:
            Sanitized input safe for AI processing

        Raises:
            SecurityError: If input contains malicious content
        """
        import re
        import html

        # Remove potential injection patterns first (before escaping)
        sanitized = re.sub(r'<.*?>', '', raw_input)
        sanitized = re.sub(r'(?i)(ignore|override|forget)\s+(previous|instructions)', '', sanitized)

        # Check for remaining dangerous characters
        if re.search(r'[<>"\';`]', sanitized):
            raise SecurityError("Input contains malicious content")

        # Escape any remaining HTML entities
        sanitized = html.escape(sanitized)

        return sanitized
    
    def _validate_spec(self, spec: FetchSpec) -> FetchSpec:
        """
        Validate fetch specification for security and completeness.
        
        Args:
            spec: FetchSpec to validate
            
        Returns:
            Validated FetchSpec
            
        Raises:
            ValidationError: If spec is invalid
            SecurityError: If spec contains security risks
        """
        if not spec.raw_text.strip():
            raise ValidationError("raw_text cannot be empty")
            
        if not spec.urls:
            raise ValidationError("At least one URL must be provided")
            
        for url in spec.urls:
            if not self._is_safe_url(url):
                raise SecurityError(f"Unsafe URL detected: {url}")
                
        # Note: Pydantic validation now handles retry_count and timeout constraints
            
        return spec
    
    def _is_safe_url(self, url: str) -> bool:
        """
        Check if URL is safe to fetch from.

        Args:
            url: URL to validate

        Returns:
            True if URL is safe, False otherwise
        """
        import re
        from urllib.parse import urlparse

        try:
            parsed = urlparse(url)

            if parsed.scheme not in ['http', 'https']:
                return False

            if not parsed.netloc:
                return False

            # Check against domain whitelist
            domain_allowed = any(
                parsed.netloc.lower() == domain.lower() or
                parsed.netloc.lower().endswith('.' + domain.lower())
                for domain in self.ALLOWED_DOMAINS
            )
            if not domain_allowed:
                return False

            # Block localhost and private IPs
            if any(blocked in parsed.netloc.lower() for blocked in
                   ['localhost', '127.0.0.1', '0.0.0.0', '::1']):
                return False

            # Check for malicious characters
            if re.search(r'[<>"\']', url):
                return False

            return True
        except Exception:
            return False
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        import uuid
        return str(uuid.uuid4())
    
    def get_cache_key(self, url: str) -> str:
        """
        Generate cache key for URL.

        Args:
            url: URL to generate key for

        Returns:
            Cache key string
        """
        import hashlib
        content = f"{url}:{self._spec.method.value}:{self._spec.expected_format.value}:{self._session_id}"
        return f"datafetch:{hashlib.md5(content.encode()).hexdigest()}"

    def _get_from_cache(self, key: str) -> Optional[Any]:
        """
        Get data from cache with null check fallback.

        Args:
            key: Cache key

        Returns:
            Cached data or None if not found or cache unavailable
        """
        if self._cache_client is None:
            if self._spec.cache_strategy != CacheStrategy.NO_CACHE:
                self._record_metric('cache_fallback', 'memory')
            return None

        try:
            return self._cache_client.get(key)
        except Exception:
            self._record_metric('cache_error', 1)
            return None

    def _set_cache(self, key: str, value: Any) -> None:
        """
        Set data in cache with null check fallback.

        Args:
            key: Cache key
            value: Data to cache
        """
        if self._cache_client is None:
            return

        try:
            self._cache_client.set(key, value, ttl=self._spec.cache_ttl)
        except Exception:
            self._record_metric('cache_error', 1)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for this fetch session.
        
        Returns:
            Dictionary containing metrics data
        """
        return self._metrics.copy()
    
    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self._metrics = {}
        
    def _record_metric(self, key: str, value: Any) -> None:
        """
        Record a performance metric for observability.

        Args:
            key: Metric name
            value: Metric value
        """
        self._metrics[key] = value

    def _increment_metric(self, key: str, amount: int = 1) -> None:
        """
        Increment a counter metric.

        Args:
            key: Metric name
            amount: Amount to increment by
        """
        self._metrics[key] = self._metrics.get(key, 0) + amount