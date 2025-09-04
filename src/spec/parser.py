import re
import json
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import urlparse, urljoin
from dataclasses import dataclass
from enum import Enum

from ..core.datafetch import DataFormat, FetchMethod, CacheStrategy


class SpecType(Enum):
    RAW_TEXT = "raw_text"
    URL_LIST = "url_list"
    STRUCTURE_DEF = "structure_definition"


@dataclass
class ParsedSpec:
    raw_text: str
    extracted_urls: List[str]
    inferred_structure: Optional[Dict[str, Any]]
    suggested_format: DataFormat
    suggested_method: FetchMethod
    confidence_score: float
    metadata: Dict[str, Any]


class RawTextParser:
    """
    Parses raw text descriptions to extract URLs and infer data requirements.
    
    Focuses on news website patterns and common data extraction scenarios.
    """
    
    # Common news website patterns
    NEWS_DOMAINS = {
        'nytimes.com', 'reuters.com', 'telegraph.co.uk', 'bbc.com',
        'cnn.com', 'guardian.com', 'washingtonpost.com', 'wsj.com',
        'bloomberg.com', 'ft.com', 'economist.com', 'ap.org'
    }
    
    # URL extraction patterns
    URL_PATTERNS = [
        r'https?://[^\s<>"]+',
        r'www\.[^\s<>"]+',
        r'[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s<>"]*)?'
    ]
    
    # Data type indicators
    DATA_TYPE_INDICATORS = {
        'article': ['article', 'news', 'story', 'post', 'blog'],
        'headlines': ['headline', 'title', 'header'],
        'metadata': ['author', 'date', 'timestamp', 'category', 'tag'],
        'content': ['content', 'body', 'text', 'paragraph'],
        'links': ['link', 'url', 'href', 'reference'],
        'images': ['image', 'photo', 'picture', 'media']
    }
    
    def __init__(self):
        self.compiled_url_patterns = [re.compile(pattern, re.IGNORECASE) 
                                     for pattern in self.URL_PATTERNS]
    
    def parse(self, raw_text: str) -> ParsedSpec:
        """
        Parse raw text to extract URLs and infer data structure requirements.
        
        Args:
            raw_text: User's description of data requirements
            
        Returns:
            ParsedSpec containing extracted information
        """
        # Clean and normalize input
        cleaned_text = self._clean_text(raw_text)
        
        # Extract URLs
        extracted_urls = self._extract_urls(cleaned_text)
        
        # Infer data structure based on text analysis
        inferred_structure = self._infer_structure(cleaned_text)
        
        # Suggest format and method based on analysis
        suggested_format = self._suggest_format(cleaned_text, extracted_urls)
        suggested_method = self._suggest_method(cleaned_text, extracted_urls)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence(
            cleaned_text, extracted_urls, inferred_structure
        )
        
        # Collect metadata
        metadata = self._extract_metadata(cleaned_text, extracted_urls)
        
        return ParsedSpec(
            raw_text=raw_text,
            extracted_urls=extracted_urls,
            inferred_structure=inferred_structure,
            suggested_format=suggested_format,
            suggested_method=suggested_method,
            confidence_score=confidence_score,
            metadata=metadata
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize input text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove potentially harmful characters
        text = re.sub(r'[<>"\'\`]', '', text)
        
        return text
    
    def _extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text using multiple patterns."""
        urls = []
        
        for pattern in self.compiled_url_patterns:
            matches = pattern.findall(text)
            urls.extend(matches)
        
        # Clean and validate URLs
        cleaned_urls = []
        for url in urls:
            cleaned_url = self._clean_url(url)
            if cleaned_url and self._is_valid_url(cleaned_url):
                cleaned_urls.append(cleaned_url)
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(cleaned_urls))
    
    def _clean_url(self, url: str) -> str:
        """Clean and normalize URL."""
        url = url.strip()
        
        # Add protocol if missing
        if not url.startswith(('http://', 'https://')):
            if url.startswith('www.'):
                url = 'https://' + url
            elif '.' in url and not url.startswith('//'):
                url = 'https://' + url
        
        return url
    
    def _is_valid_url(self, url: str) -> bool:
        """Validate URL structure."""
        try:
            parsed = urlparse(url)
            return bool(parsed.netloc and parsed.scheme in ['http', 'https'])
        except Exception:
            return False
    
    def _infer_structure(self, text: str) -> Optional[Dict[str, Any]]:
        """Infer data structure from text description."""
        structure = {}
        text_lower = text.lower()
        
        # Check for data type indicators
        for data_type, indicators in self.DATA_TYPE_INDICATORS.items():
            if any(indicator in text_lower for indicator in indicators):
                structure[data_type] = self._get_field_spec(data_type)
        
        # Check for list/array indicators
        if any(word in text_lower for word in ['list', 'array', 'multiple', 'all']):
            structure['_is_list'] = True
        
        # Check for nested data indicators
        if any(word in text_lower for word in ['nested', 'hierarchy', 'tree']):
            structure['_is_nested'] = True
        
        return structure if structure else None
    
    def _get_field_spec(self, data_type: str) -> Dict[str, Any]:
        """Get field specification for data type."""
        field_specs = {
            'article': {
                'type': 'object',
                'required': ['title', 'content'],
                'properties': {
                    'title': {'type': 'string'},
                    'content': {'type': 'string'},
                    'url': {'type': 'string', 'format': 'uri'}
                }
            },
            'headlines': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'title': {'type': 'string'},
                        'url': {'type': 'string', 'format': 'uri'}
                    }
                }
            },
            'metadata': {
                'type': 'object',
                'properties': {
                    'author': {'type': 'string'},
                    'date': {'type': 'string', 'format': 'date-time'},
                    'category': {'type': 'string'}
                }
            },
            'content': {'type': 'string'},
            'links': {
                'type': 'array',
                'items': {'type': 'string', 'format': 'uri'}
            },
            'images': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'url': {'type': 'string', 'format': 'uri'},
                        'alt': {'type': 'string'},
                        'caption': {'type': 'string'}
                    }
                }
            }
        }
        
        return field_specs.get(data_type, {'type': 'string'})
    
    def _suggest_format(self, text: str, urls: List[str]) -> DataFormat:
        """Suggest data format based on text and URLs."""
        text_lower = text.lower()
        
        # Check for explicit format mentions
        if 'json' in text_lower:
            return DataFormat.JSON
        elif 'xml' in text_lower:
            return DataFormat.XML
        elif 'csv' in text_lower:
            return DataFormat.CSV
        elif 'html' in text_lower:
            return DataFormat.HTML
        
        # Infer from URL patterns
        for url in urls:
            if any(ext in url.lower() for ext in ['.json', '/api/', 'api.']):
                return DataFormat.JSON
            elif any(ext in url.lower() for ext in ['.xml', '.rss', '.atom']):
                return DataFormat.XML
            elif '.csv' in url.lower():
                return DataFormat.CSV
        
        # Default to JSON for structured data
        return DataFormat.JSON
    
    def _suggest_method(self, text: str, urls: List[str]) -> FetchMethod:
        """Suggest fetch method based on text and URLs."""
        text_lower = text.lower()
        
        # Check for JavaScript/dynamic content indicators
        if any(word in text_lower for word in 
               ['javascript', 'dynamic', 'spa', 'react', 'vue', 'angular']):
            return FetchMethod.PLAYWRIGHT
        
        # Check URL domains for known SPA sites
        for url in urls:
            try:
                domain = urlparse(url).netloc.lower()
                # Some news sites that heavily use JavaScript
                if any(spa_domain in domain for spa_domain in 
                       ['twitter.com', 'facebook.com', 'instagram.com']):
                    return FetchMethod.PLAYWRIGHT
            except Exception:
                continue
        
        # Default to requests for news sites
        return FetchMethod.REQUESTS
    
    def _calculate_confidence(self, text: str, urls: List[str], 
                            structure: Optional[Dict[str, Any]]) -> float:
        """Calculate confidence score for parsed specification."""
        confidence = 0.0
        
        # Base confidence from text quality
        if len(text.split()) >= 5:
            confidence += 0.3
        
        # Confidence from URL extraction
        if urls:
            confidence += 0.4
            # Bonus for news domains
            for url in urls:
                try:
                    domain = urlparse(url).netloc.lower()
                    if any(news_domain in domain for news_domain in self.NEWS_DOMAINS):
                        confidence += 0.1
                        break
                except Exception:
                    continue
        
        # Confidence from structure inference
        if structure and len(structure) > 1:
            confidence += 0.3
        
        return min(confidence, 1.0)
    
    def _extract_metadata(self, text: str, urls: List[str]) -> Dict[str, Any]:
        """Extract metadata from text and URLs."""
        metadata = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'url_count': len(urls),
            'domains': []
        }
        
        # Extract domains
        for url in urls:
            try:
                domain = urlparse(url).netloc
                if domain and domain not in metadata['domains']:
                    metadata['domains'].append(domain)
            except Exception:
                continue
        
        # Check for time-sensitive indicators
        time_indicators = ['latest', 'recent', 'today', 'live', 'breaking']
        if any(indicator in text.lower() for indicator in time_indicators):
            metadata['time_sensitive'] = True
        
        return metadata


class URLManager:
    """
    Manages URL validation, normalization, and discovery.
    """
    
    def __init__(self):
        self.parser = RawTextParser()
    
    def discover_urls(self, raw_text: str) -> List[str]:
        """
        Discover URLs from raw text description.
        
        Args:
            raw_text: Text description that may contain URLs
            
        Returns:
            List of discovered URLs
        """
        parsed = self.parser.parse(raw_text)
        return parsed.extracted_urls
    
    def validate_urls(self, urls: List[str]) -> List[Tuple[str, bool, Optional[str]]]:
        """
        Validate a list of URLs.
        
        Args:
            urls: List of URLs to validate
            
        Returns:
            List of tuples (url, is_valid, error_message)
        """
        results = []
        
        for url in urls:
            try:
                if self._is_safe_url(url):
                    results.append((url, True, None))
                else:
                    results.append((url, False, "URL failed safety checks"))
            except Exception as e:
                results.append((url, False, str(e)))
        
        return results
    
    def normalize_urls(self, urls: List[str]) -> List[str]:
        """
        Normalize URLs to standard format.
        
        Args:
            urls: List of URLs to normalize
            
        Returns:
            List of normalized URLs
        """
        normalized = []
        
        for url in urls:
            try:
                # Clean the URL
                cleaned = self.parser._clean_url(url)
                
                # Parse and rebuild
                parsed = urlparse(cleaned)
                if parsed.netloc and parsed.scheme:
                    normalized_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                    if parsed.query:
                        normalized_url += f"?{parsed.query}"
                    normalized.append(normalized_url)
            except Exception:
                continue
        
        return normalized
    
    def _is_safe_url(self, url: str) -> bool:
        """Check if URL is safe to fetch from."""
        try:
            parsed = urlparse(url)
            
            if parsed.scheme not in ['http', 'https']:
                return False
                
            if not parsed.netloc:
                return False
                
            # Block local/private addresses
            if any(blocked in parsed.netloc.lower() for blocked in 
                   ['localhost', '127.0.0.1', '0.0.0.0', '::1', '10.', '192.168.']):
                return False
                
            return True
        except Exception:
            return False


class StructureDefinition:
    """
    Manages data structure definitions and validation schemas.
    """
    
    def __init__(self):
        pass
    
    def generate_schema(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate JSON schema from structure definition.
        
        Args:
            structure: Structure definition dictionary
            
        Returns:
            JSON schema dictionary
        """
        if not structure:
            return {"type": "object"}
        
        schema = {
            "type": "object",
            "properties": {}
        }
        
        required = []
        
        for key, value in structure.items():
            if key.startswith('_'):
                continue
                
            if isinstance(value, dict) and 'type' in value:
                schema["properties"][key] = value
                if value.get('required', False):
                    required.append(key)
            else:
                schema["properties"][key] = {"type": "string"}
        
        if required:
            schema["required"] = required
        
        # Handle special flags
        if structure.get('_is_list'):
            schema = {
                "type": "array",
                "items": schema
            }
        
        return schema
    
    def validate_data_against_schema(self, data: Any, schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate data against JSON schema.
        
        Args:
            data: Data to validate
            schema: JSON schema to validate against
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        try:
            # Basic type validation
            expected_type = schema.get('type', 'object')
            
            if expected_type == 'object' and not isinstance(data, dict):
                errors.append(f"Expected object, got {type(data).__name__}")
                return False, errors
            
            if expected_type == 'array' and not isinstance(data, list):
                errors.append(f"Expected array, got {type(data).__name__}")
                return False, errors
            
            # Property validation for objects
            if expected_type == 'object' and isinstance(data, dict):
                properties = schema.get('properties', {})
                required = schema.get('required', [])
                
                # Check required properties
                for req_prop in required:
                    if req_prop not in data:
                        errors.append(f"Missing required property: {req_prop}")
                
                # Validate existing properties
                for prop, value in data.items():
                    if prop in properties:
                        prop_schema = properties[prop]
                        is_valid, prop_errors = self.validate_data_against_schema(value, prop_schema)
                        if not is_valid:
                            errors.extend([f"{prop}: {error}" for error in prop_errors])
            
            # Array item validation
            if expected_type == 'array' and isinstance(data, list):
                items_schema = schema.get('items', {})
                for i, item in enumerate(data):
                    is_valid, item_errors = self.validate_data_against_schema(item, items_schema)
                    if not is_valid:
                        errors.extend([f"Item {i}: {error}" for error in item_errors])
            
            return len(errors) == 0, errors
            
        except Exception as e:
            return False, [f"Validation error: {str(e)}"]