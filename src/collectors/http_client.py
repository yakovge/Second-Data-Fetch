import asyncio
import time
from typing import Dict, List, Optional, Any, AsyncGenerator, Generator
from datetime import datetime, timedelta
import requests
import aiohttp
import json
from urllib.parse import urlparse, urljoin
from dataclasses import dataclass
import logging

from ..core.datafetch import DataFetch, FetchResult, FetchSpec, DataFormat, FetchMethod, FetchError, ValidationError


@dataclass
class RequestConfig:
    headers: Dict[str, str]
    timeout: int
    verify_ssl: bool
    allow_redirects: bool
    max_redirects: int
    retry_count: int
    backoff_factor: float


class HTTPClient(DataFetch):
    """
    HTTP client implementation using Requests library.
    
    Optimized for news websites and structured data fetching.
    Includes robust error handling, retries, and caching.
    """
    
    DEFAULT_HEADERS = {
        'User-Agent': 'DataFetch/1.0 (+https://example.com/bot)',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    NEWS_HEADERS = {
        'User-Agent': 'Mozilla/5.0 (compatible; NewsBot/1.0; +https://example.com/newsbot)',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache'
    }
    
    def __init__(self, 
                 spec: FetchSpec,
                 cache_client: Optional[Any] = None,
                 storage_client: Optional[Any] = None,
                 ai_client: Optional[Any] = None):
        super().__init__(spec, cache_client, storage_client, ai_client)
        
        self.config = self._create_request_config()
        self.session = None
        self._setup_logging()
    
    def _create_request_config(self) -> RequestConfig:
        """Create request configuration from spec."""
        # Determine if this is a news website
        is_news_site = any(
            news_domain in url.lower() 
            for url in self._spec.urls 
            for news_domain in ['news', 'reuters', 'nytimes', 'bbc', 'cnn', 'guardian']
        )
        
        headers = self.NEWS_HEADERS.copy() if is_news_site else self.DEFAULT_HEADERS.copy()
        
        # Add custom headers if provided
        if self._spec.custom_headers:
            headers.update(self._spec.custom_headers)
        
        return RequestConfig(
            headers=headers,
            timeout=self._spec.timeout,
            verify_ssl=True,
            allow_redirects=True,
            max_redirects=10,
            retry_count=self._spec.retry_count,
            backoff_factor=1.0
        )
    
    def _setup_logging(self):
        """Setup logging for HTTP client."""
        self.logger = logging.getLogger(f"datafetch.http.{self._session_id[:8]}")
        self.logger.setLevel(logging.INFO)
    
    def _get_session(self) -> requests.Session:
        """Get or create requests session."""
        if self.session is None:
            self.session = requests.Session()
            self.session.headers.update(self.config.headers)
        return self.session
    
    def fetch(self) -> FetchResult:
        """
        Synchronous fetch operation using requests.
        """
        start_time = time.time()
        
        try:
            # Try cache first if available
            if self._cache_client:
                cached_result = self._try_cache(self._spec.urls[0])
                if cached_result:
                    return cached_result
            
            # Fetch from primary URL
            url = self._spec.urls[0]
            self.logger.info(f"Fetching URL: {url}")
            
            result = self._fetch_single_url(url)
            
            # Store in cache if available
            if self._cache_client and result.error is None:
                self._store_in_cache(url, result)
            
            # Record metrics
            execution_time = time.time() - start_time
            self._record_metric('total_execution_time', execution_time)
            self._record_metric('urls_fetched', 1)
            
            result.execution_time = execution_time
            return result
            
        except Exception as e:
            self.logger.error(f"Fetch failed: {str(e)}")
            execution_time = time.time() - start_time
            return FetchResult(
                url=self._spec.urls[0],
                data=None,
                timestamp=datetime.now(),
                format=self._spec.expected_format,
                method=FetchMethod.REQUESTS,
                metadata={'error': str(e)},
                error=str(e),
                execution_time=execution_time
            )
    
    async def afetch(self) -> FetchResult:
        """
        Asynchronous fetch operation using aiohttp.
        """
        start_time = time.time()
        
        try:
            # Try cache first if available
            if self._cache_client:
                cached_result = self._try_cache(self._spec.urls[0])
                if cached_result:
                    return cached_result
            
            # Fetch from primary URL
            url = self._spec.urls[0]
            self.logger.info(f"Async fetching URL: {url}")
            
            result = await self._afetch_single_url(url)
            
            # Store in cache if available
            if self._cache_client and result.error is None:
                self._store_in_cache(url, result)
            
            # Record metrics
            execution_time = time.time() - start_time
            self._record_metric('total_execution_time', execution_time)
            self._record_metric('urls_fetched', 1)
            
            result.execution_time = execution_time
            return result
            
        except Exception as e:
            self.logger.error(f"Async fetch failed: {str(e)}")
            execution_time = time.time() - start_time
            return FetchResult(
                url=self._spec.urls[0],
                data=None,
                timestamp=datetime.now(),
                format=self._spec.expected_format,
                method=FetchMethod.REQUESTS,
                metadata={'error': str(e)},
                error=str(e),
                execution_time=execution_time
            )
    
    def fetch_stream(self) -> Generator[FetchResult, None, None]:
        """
        Streaming fetch operation for multiple URLs.
        """
        for url in self._spec.urls:
            try:
                self.logger.info(f"Streaming fetch URL: {url}")
                result = self._fetch_single_url(url)
                yield result
            except Exception as e:
                self.logger.error(f"Stream fetch failed for {url}: {str(e)}")
                yield FetchResult(
                    url=url,
                    data=None,
                    timestamp=datetime.now(),
                    format=self._spec.expected_format,
                    method=FetchMethod.REQUESTS,
                    metadata={'error': str(e)},
                    error=str(e)
                )
    
    async def afetch_stream(self) -> AsyncGenerator[FetchResult, None]:
        """
        Asynchronous streaming fetch operation.
        """
        connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
        
        async with aiohttp.ClientSession(
            headers=self.config.headers,
            timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            connector=connector
        ) as session:
            
            tasks = []
            for url in self._spec.urls:
                task = self._afetch_single_url_with_session(session, url)
                tasks.append(task)
            
            # Execute requests concurrently
            for coro in asyncio.as_completed(tasks):
                try:
                    result = await coro
                    yield result
                except Exception as e:
                    self.logger.error(f"Async stream fetch failed: {str(e)}")
                    yield FetchResult(
                        url="unknown",
                        data=None,
                        timestamp=datetime.now(),
                        format=self._spec.expected_format,
                        method=FetchMethod.REQUESTS,
                        metadata={'error': str(e)},
                        error=str(e)
                    )
    
    def _fetch_single_url(self, url: str) -> FetchResult:
        """Fetch data from a single URL with retries."""
        session = self._get_session()
        last_exception = None
        
        for attempt in range(self.config.retry_count + 1):
            try:
                if attempt > 0:
                    wait_time = self.config.backoff_factor * (2 ** (attempt - 1))
                    time.sleep(wait_time)
                    self.logger.info(f"Retry attempt {attempt} for {url}")
                
                response = session.get(
                    url,
                    timeout=self.config.timeout,
                    verify=self.config.verify_ssl,
                    allow_redirects=self.config.allow_redirects
                )
                
                response.raise_for_status()
                
                # Process response based on expected format
                data = self._process_response(response)
                
                # Validate data if validation rules exist
                if not self.validate_data(data):
                    raise ValidationError("Data validation failed")
                
                return FetchResult(
                    url=url,
                    data=data,
                    timestamp=datetime.now(),
                    format=self._spec.expected_format,
                    method=FetchMethod.REQUESTS,
                    metadata={
                        'status_code': response.status_code,
                        'content_type': response.headers.get('content-type', ''),
                        'content_length': len(response.content),
                        'final_url': response.url,
                        'attempt': attempt + 1
                    },
                    cache_hit=False
                )
                
            except requests.exceptions.RequestException as e:
                last_exception = e
                self.logger.warning(f"Request failed (attempt {attempt + 1}): {str(e)}")
                continue
            except Exception as e:
                last_exception = e
                self.logger.error(f"Processing failed: {str(e)}")
                break
        
        # All retries failed
        raise FetchError(f"Failed to fetch {url} after {self.config.retry_count + 1} attempts: {str(last_exception)}")
    
    async def _afetch_single_url(self, url: str) -> FetchResult:
        """Async fetch data from a single URL with retries."""
        connector = aiohttp.TCPConnector(verify_ssl=self.config.verify_ssl)
        
        async with aiohttp.ClientSession(
            headers=self.config.headers,
            timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            connector=connector
        ) as session:
            return await self._afetch_single_url_with_session(session, url)
    
    async def _afetch_single_url_with_session(self, session: aiohttp.ClientSession, url: str) -> FetchResult:
        """Async fetch with existing session."""
        last_exception = None
        
        for attempt in range(self.config.retry_count + 1):
            try:
                if attempt > 0:
                    wait_time = self.config.backoff_factor * (2 ** (attempt - 1))
                    await asyncio.sleep(wait_time)
                    self.logger.info(f"Async retry attempt {attempt} for {url}")
                
                async with session.get(url) as response:
                    response.raise_for_status()
                    
                    # Read response content
                    content = await response.read()
                    
                    # Process response based on expected format
                    data = self._process_response_content(
                        content, 
                        response.headers.get('content-type', ''),
                        response.status
                    )
                    
                    # Validate data if validation rules exist
                    if not self.validate_data(data):
                        raise ValidationError("Data validation failed")
                    
                    return FetchResult(
                        url=url,
                        data=data,
                        timestamp=datetime.now(),
                        format=self._spec.expected_format,
                        method=FetchMethod.REQUESTS,
                        metadata={
                            'status_code': response.status,
                            'content_type': response.headers.get('content-type', ''),
                            'content_length': len(content),
                            'final_url': str(response.url),
                            'attempt': attempt + 1
                        },
                        cache_hit=False
                    )
                    
            except aiohttp.ClientError as e:
                last_exception = e
                self.logger.warning(f"Async request failed (attempt {attempt + 1}): {str(e)}")
                continue
            except Exception as e:
                last_exception = e
                self.logger.error(f"Async processing failed: {str(e)}")
                break
        
        # All retries failed
        raise FetchError(f"Failed to async fetch {url} after {self.config.retry_count + 1} attempts: {str(last_exception)}")
    
    def _process_response(self, response: requests.Response) -> Any:
        """Process HTTP response based on expected format."""
        content_type = response.headers.get('content-type', '').lower()
        return self._process_response_content(response.content, content_type, response.status_code)
    
    def _process_response_content(self, content: bytes, content_type: str, status_code: int) -> Any:
        """Process response content based on expected format."""
        try:
            text_content = content.decode('utf-8')
        except UnicodeDecodeError:
            text_content = content.decode('utf-8', errors='ignore')
        
        if self._spec.expected_format == DataFormat.JSON:
            if 'application/json' in content_type:
                return json.loads(text_content)
            else:
                # Try to extract JSON from HTML or other formats
                return self._extract_json_from_text(text_content)
        
        elif self._spec.expected_format == DataFormat.XML:
            return self._parse_xml(text_content)
        
        elif self._spec.expected_format == DataFormat.HTML:
            return self._parse_html(text_content)
        
        elif self._spec.expected_format == DataFormat.CSV:
            return self._parse_csv(text_content)
        
        else:  # DataFormat.TEXT
            return text_content
    
    def _extract_json_from_text(self, text: str) -> Any:
        """Extract JSON data from text content."""
        import re
        
        # Try to find JSON-LD structured data
        json_ld_pattern = r'<script type="application/ld\+json"[^>]*>(.*?)</script>'
        matches = re.findall(json_ld_pattern, text, re.DOTALL | re.IGNORECASE)
        
        if matches:
            try:
                return json.loads(matches[0].strip())
            except json.JSONDecodeError:
                pass
        
        # Try to find other JSON patterns
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, text)
        
        for match in matches:
            try:
                data = json.loads(match)
                if isinstance(data, dict) and len(data) > 1:
                    return data
            except json.JSONDecodeError:
                continue
        
        # Return raw text if no JSON found
        return text
    
    def _parse_xml(self, text: str) -> Any:
        """Parse XML content."""
        try:
            from xml.etree import ElementTree as ET
            root = ET.fromstring(text)
            return self._xml_to_dict(root)
        except ET.ParseError as e:
            self.logger.warning(f"XML parsing failed: {str(e)}")
            return text
    
    def _xml_to_dict(self, element) -> Dict[str, Any]:
        """Convert XML element to dictionary."""
        result = {}
        
        # Add attributes
        if element.attrib:
            result['@attributes'] = element.attrib
        
        # Add text content
        if element.text and element.text.strip():
            if len(element) == 0:
                return element.text.strip()
            else:
                result['#text'] = element.text.strip()
        
        # Add child elements
        for child in element:
            child_data = self._xml_to_dict(child)
            
            if child.tag in result:
                # Convert to list if multiple elements with same tag
                if not isinstance(result[child.tag], list):
                    result[child.tag] = [result[child.tag]]
                result[child.tag].append(child_data)
            else:
                result[child.tag] = child_data
        
        return result
    
    def _parse_html(self, text: str) -> Any:
        """Parse HTML content using BeautifulSoup."""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(text, 'html.parser')
            
            # Extract structured data based on common news website patterns
            return self._extract_article_data(soup)
            
        except ImportError:
            self.logger.warning("BeautifulSoup not available, returning raw HTML")
            return text
    
    def _extract_article_data(self, soup) -> Dict[str, Any]:
        """Extract article data from HTML using common patterns."""
        data = {}
        
        # Extract title
        title_selectors = [
            'h1',
            '.headline',
            '.title',
            '[property="og:title"]',
            'title'
        ]
        
        for selector in title_selectors:
            title_elem = soup.select_one(selector)
            if title_elem:
                data['title'] = title_elem.get_text(strip=True) or title_elem.get('content', '')
                break
        
        # Extract content
        content_selectors = [
            '.article-body',
            '.content',
            '.post-content',
            '[property="articleBody"]',
            'main p'
        ]
        
        for selector in content_selectors:
            content_elems = soup.select(selector)
            if content_elems:
                data['content'] = '\n'.join(elem.get_text(strip=True) for elem in content_elems)
                break
        
        # Extract metadata
        author_elem = soup.select_one('[property="article:author"], .author, .byline')
        if author_elem:
            data['author'] = author_elem.get_text(strip=True) or author_elem.get('content', '')
        
        date_elem = soup.select_one('[property="article:published_time"], .date, time')
        if date_elem:
            data['published_date'] = date_elem.get('datetime') or date_elem.get_text(strip=True)
        
        return data
    
    def _parse_csv(self, text: str) -> List[Dict[str, Any]]:
        """Parse CSV content."""
        try:
            import csv
            from io import StringIO
            
            reader = csv.DictReader(StringIO(text))
            return list(reader)
        except Exception as e:
            self.logger.warning(f"CSV parsing failed: {str(e)}")
            return [{'raw_content': text}]
    
    def validate_data(self, data: Any) -> bool:
        """Validate fetched data against expected structure."""
        if not self._spec.validation_rules:
            return True
        
        try:
            # Basic validation - check if data is not empty
            if data is None:
                return False
            
            if isinstance(data, str) and not data.strip():
                return False
            
            if isinstance(data, dict) and not data:
                return False
            
            if isinstance(data, list) and not data:
                return False
            
            # Additional validation rules would be implemented here
            # based on the validation_rules in the spec
            
            return True
            
        except Exception as e:
            self.logger.error(f"Data validation error: {str(e)}")
            return False
    
    def extract_structure(self, sample_data: Any) -> Dict[str, Any]:
        """Extract data structure from sample data."""
        if isinstance(sample_data, dict):
            structure = {}
            for key, value in sample_data.items():
                structure[key] = self._infer_field_type(value)
            return structure
        
        elif isinstance(sample_data, list) and sample_data:
            return {
                'type': 'array',
                'items': self.extract_structure(sample_data[0])
            }
        
        else:
            return {'type': type(sample_data).__name__}
    
    def _infer_field_type(self, value: Any) -> Dict[str, Any]:
        """Infer field type from value."""
        if isinstance(value, dict):
            return {
                'type': 'object',
                'properties': {k: self._infer_field_type(v) for k, v in value.items()}
            }
        elif isinstance(value, list):
            return {
                'type': 'array',
                'items': self._infer_field_type(value[0]) if value else {'type': 'string'}
            }
        elif isinstance(value, bool):
            return {'type': 'boolean'}
        elif isinstance(value, int):
            return {'type': 'integer'}
        elif isinstance(value, float):
            return {'type': 'number'}
        else:
            return {'type': 'string'}
    
    def sanitize_input(self, raw_input: str) -> str:
        """Sanitize user input to prevent security issues."""
        import html
        import re
        
        # HTML escape
        sanitized = html.escape(raw_input)
        
        # Remove script tags and potentially dangerous content
        sanitized = re.sub(r'<script[^>]*>.*?</script>', '', sanitized, flags=re.DOTALL | re.IGNORECASE)
        sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r'on\w+\s*=', '', sanitized, flags=re.IGNORECASE)
        
        return sanitized.strip()
    
    def _try_cache(self, url: str) -> Optional[FetchResult]:
        """Try to get result from cache."""
        if not self._cache_client:
            return None
        
        try:
            cache_key = self.get_cache_key(url)
            cached_data = self._cache_client.get(cache_key)
            
            if cached_data:
                self.logger.info(f"Cache hit for {url}")
                self._increment_metric('cache_hits')
                
                # Deserialize cached result
                import pickle
                result = pickle.loads(cached_data)
                result.cache_hit = True
                return result
        except Exception as e:
            self.logger.warning(f"Cache retrieval failed: {str(e)}")
        
        self._increment_metric('cache_misses')
        return None
    
    def _store_in_cache(self, url: str, result: FetchResult) -> None:
        """Store result in cache."""
        if not self._cache_client:
            return
        
        try:
            cache_key = self.get_cache_key(url)
            
            # Serialize result
            import pickle
            cached_data = pickle.dumps(result)
            
            # Store with TTL
            ttl_seconds = int(self._spec.cache_ttl.total_seconds())
            self._cache_client.setex(cache_key, ttl_seconds, cached_data)
            
            self.logger.info(f"Stored result in cache for {url}")
            self._increment_metric('cache_stores')
            
        except Exception as e:
            self.logger.warning(f"Cache storage failed: {str(e)}")
    
    def __del__(self):
        """Cleanup resources."""
        if self.session:
            self.session.close()