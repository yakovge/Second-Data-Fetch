from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, AsyncGenerator, Generator
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import asyncio
from pydantic import BaseModel, Field, validator


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


@dataclass
class FetchResult:
    url: str
    data: Any
    timestamp: datetime
    format: DataFormat
    method: FetchMethod
    metadata: Dict[str, Any]
    cache_hit: bool = False
    execution_time: float = 0.0
    error: Optional[str] = None


@dataclass
class FetchSpec:
    raw_text: str
    urls: List[str]
    structure_definition: Optional[Dict[str, Any]] = None
    expected_format: DataFormat = DataFormat.JSON
    method: FetchMethod = FetchMethod.REQUESTS
    cache_strategy: CacheStrategy = CacheStrategy.REDIS
    cache_ttl: timedelta = timedelta(hours=1)
    retry_count: int = 3
    timeout: int = 30
    custom_headers: Optional[Dict[str, str]] = None
    validation_rules: Optional[Dict[str, Any]] = None


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
        self._spec = self._validate_spec(spec)
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
    
    @abstractmethod
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
        pass
    
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
                
        if spec.retry_count < 0 or spec.retry_count > 10:
            raise ValidationError("retry_count must be between 0 and 10")
            
        if spec.timeout <= 0 or spec.timeout > 300:
            raise ValidationError("timeout must be between 1 and 300 seconds")
            
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
                
            if any(blocked in parsed.netloc.lower() for blocked in 
                   ['localhost', '127.0.0.1', '0.0.0.0', '::1']):
                return False
                
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
        content = f"{url}:{self._spec.method.value}:{self._spec.expected_format.value}"
        return f"datafetch:{hashlib.md5(content.encode()).hexdigest()}"
    
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
        """Record a performance metric."""
        self._metrics[key] = value
        
    def _increment_metric(self, key: str, amount: int = 1) -> None:
        """Increment a counter metric."""
        self._metrics[key] = self._metrics.get(key, 0) + amount