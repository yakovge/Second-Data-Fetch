#!/usr/bin/env python3
"""
NYT Trump Article Fetcher - Proper DataFetch Implementation
Following CLAUDE.md strategy: Create custom DataFetch subclass that uses HTTPClient as a tool.
"""

import sys
import os
from typing import Dict, List, Optional, Any, AsyncGenerator, Generator
from datetime import datetime
import re

# Add project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.datafetch import DataFetch, FetchResult, FetchSpec, FetchError, ValidationError, SecurityError
from src.collectors.http_client import HTTPClient
from bs4 import BeautifulSoup


class NYTTrumpFetch(DataFetch):
    """
    DataFetch implementation for finding Trump-related articles on NYT.
    
    This follows the CLAUDE.md pattern:
    - Inherits from DataFetch abstract class
    - Uses HTTPClient as a tool (doesn't modify it)
    - Implements all abstract methods
    - Custom logic for Trump article extraction
    """
    
    def __init__(self, 
                 spec: FetchSpec,
                 cache_client: Optional[Any] = None,
                 storage_client: Optional[Any] = None,
                 ai_client: Optional[Any] = None):
        """Initialize NYT Trump fetcher."""
        super().__init__(spec, cache_client, storage_client, ai_client)
        
        # Use HTTPClient as a tool (CLAUDE.md pattern)
        self.http_client = HTTPClient(
            spec=spec,
            cache_client=cache_client,
            storage_client=storage_client,
            ai_client=ai_client
        )
        
        # Extract keywords for Trump searching
        self.keywords = self._extract_keywords()
    
    def fetch(self) -> FetchResult:
        """
        Fetch Trump articles from NYT.
        Uses HTTPClient then processes for Trump content.
        """
        try:
            # Use HTTPClient to get raw data
            raw_result = self.http_client.fetch()
            
            # Process the data for Trump articles
            trump_articles = self._extract_trump_articles(raw_result.data)
            
            # Create new FetchResult with processed Trump data
            return FetchResult(
                url=raw_result.url,
                data=trump_articles,
                timestamp=raw_result.timestamp,
                format=raw_result.format,
                method=raw_result.method,
                metadata={
                    **raw_result.metadata,
                    'extraction_method': 'NYTTrumpFetch',
                    'articles_found': len(trump_articles.get('articles', [])),
                    'keywords_searched': self.keywords
                },
                cache_hit=raw_result.cache_hit,
                execution_time=raw_result.execution_time
            )
            
        except Exception as e:
            raise FetchError(f"NYT Trump fetch failed: {str(e)}")
    
    async def afetch(self) -> FetchResult:
        """Async version of fetch."""
        # For now, use sync version
        # In full implementation, would use async HTTPClient methods
        return self.fetch()
    
    def fetch_stream(self) -> Generator[FetchResult, None, None]:
        """Stream Trump articles one by one."""
        result = self.fetch()
        
        if isinstance(result.data, dict) and 'articles' in result.data:
            articles = result.data['articles']
            
            for i, article in enumerate(articles):
                # Yield each article as separate FetchResult
                yield FetchResult(
                    url=article.get('url', result.url),
                    data=article,
                    timestamp=result.timestamp,
                    format=result.format,
                    method=result.method,
                    metadata={
                        **result.metadata,
                        'article_index': i + 1,
                        'total_articles': len(articles)
                    },
                    cache_hit=result.cache_hit,
                    execution_time=result.execution_time / len(articles)
                )
        else:
            yield result
    
    async def afetch_stream(self) -> AsyncGenerator[FetchResult, None]:
        """Async stream version."""
        # Convert sync generator to async
        for result in self.fetch_stream():
            yield result
    
    def validate_data(self, data: Any) -> bool:
        """Validate that we got Trump articles."""
        try:
            if not isinstance(data, dict):
                return False
            
            # Check required fields
            required_fields = ['articles', 'extraction_method', 'total_found']
            if not all(field in data for field in required_fields):
                return False
            
            # Check that we have articles
            articles = data.get('articles', [])
            if not isinstance(articles, list):
                return False
            
            # Check that articles have Trump content
            trump_mentions = 0
            for article in articles:
                if isinstance(article, dict):
                    article_text = str(article).lower()
                    if 'trump' in article_text:
                        trump_mentions += 1
            
            # At least some articles should mention Trump
            return trump_mentions > 0
            
        except Exception as e:
            self.logger.error(f"Data validation error: {str(e)}")
            return False
    
    def extract_structure(self, sample_data: Any) -> Dict[str, Any]:
        """Extract structure for Trump articles."""
        
        # Use AI if available
        if self._ai_client:
            try:
                context = f"User request: {self._spec.raw_text} | NYT Trump articles"
                ai_structure = self._ai_client.generate_structure_from_sample(sample_data, context)
                if ai_structure and isinstance(ai_structure, dict):
                    return ai_structure
            except Exception as e:
                self.logger.warning(f"AI structure extraction failed: {e}")
        
        # Fallback: Define Trump article structure
        if isinstance(sample_data, dict) and 'articles' in sample_data:
            return {
                'type': 'object',
                'description': 'NYT articles mentioning Trump',
                'properties': {
                    'articles': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'title': {'type': 'string', 'description': 'Article headline'},
                                'url': {'type': 'string', 'format': 'uri', 'description': 'Article URL'},
                                'summary': {'type': 'string', 'description': 'Article summary'},
                                'relevance_score': {'type': 'number', 'description': 'Trump relevance (0-1)'},
                                'keywords_matched': {'type': 'array', 'items': {'type': 'string'}}
                            },
                            'required': ['title', 'url', 'summary']
                        }
                    },
                    'total_found': {'type': 'integer'},
                    'extraction_method': {'type': 'string'}
                },
                'required': ['articles', 'total_found']
            }
        else:
            return {'type': 'object', 'description': 'No Trump articles found'}
    
    def sanitize_input(self, raw_input: str) -> str:
        """Sanitize input for Trump article searches."""
        if not raw_input:
            return ""
        
        # Remove dangerous patterns
        dangerous_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'data:',
            r'vbscript:',
        ]
        
        sanitized = str(raw_input)
        for pattern in dangerous_patterns:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        
        # Limit length
        max_length = 1000
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length] + "..."
        
        return sanitized.strip()
    
    def _extract_keywords(self) -> List[str]:
        """Extract keywords from user request."""
        text = self._spec.raw_text.lower()
        
        # Remove filler words
        filler_words = {
            'i', 'want', 'get', 'show', 'find', 'fetch', 'need', 'looking', 'for',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'from',
            'that', 'mentions', 'about', 'article', 'articles', 'news', 'story'
        }
        
        words = re.findall(r'\b\w+\b', text)
        keywords = [word for word in words if len(word) > 2 and word not in filler_words]
        
        # Always include 'trump' if not explicitly mentioned
        if 'trump' not in keywords and 'trump' in text:
            keywords.insert(0, 'trump')
        
        return keywords[:5]  # Top 5 keywords
    
    def _extract_trump_articles(self, html_or_data: Any) -> Dict[str, Any]:
        """Extract Trump articles from HTML or data."""
        
        # If HTTPClient already processed it into structured data, use that
        if isinstance(html_or_data, dict) and 'articles' in html_or_data:
            # Filter for Trump content
            all_articles = html_or_data['articles']
            trump_articles = []
            
            for article in all_articles:
                if self._article_mentions_trump(article):
                    trump_articles.append(article)
            
            return {
                'extraction_method': 'NYTTrumpFetch',
                'total_found': len(trump_articles),
                'articles': trump_articles[:5],  # Top 5
                'keywords_searched': self.keywords,
                'source': 'New York Times'
            }
        
        # If we got raw HTML, parse it ourselves
        elif isinstance(html_or_data, str) and '<html' in html_or_data.lower():
            return self._parse_html_for_trump_articles(html_or_data)
        
        else:
            return {
                'extraction_method': 'NYTTrumpFetch',
                'total_found': 0,
                'articles': [],
                'error': 'Unable to process data format',
                'keywords_searched': self.keywords
            }
    
    def _article_mentions_trump(self, article: Dict[str, Any]) -> bool:
        """Check if article mentions Trump."""
        article_text = str(article).lower()
        return any(keyword in article_text for keyword in self.keywords)
    
    def _parse_html_for_trump_articles(self, html: str) -> Dict[str, Any]:
        """Parse HTML for Trump articles (fallback method)."""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove noise
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                element.decompose()
            
            articles = []
            found_titles = set()
            
            # NYT-specific selectors
            selectors = [
                'article', 'a[href*="/202"]', 'h1, h2, h3',
                '[data-testid*="headline"]', '.story-wrapper'
            ]
            
            for selector in selectors:
                elements = soup.select(selector)
                
                for elem in elements:
                    try:
                        text = elem.get_text().strip()
                        if not text or len(text) < 20:
                            continue
                        
                        # Check for Trump mentions
                        text_lower = text.lower()
                        trump_keywords = [kw for kw in self.keywords if kw in text_lower]
                        
                        if trump_keywords and text not in found_titles:
                            found_titles.add(text)
                            
                            # Extract URL
                            url = ""
                            if elem.name == 'a' and elem.get('href'):
                                href = elem.get('href')
                                url = f"https://www.nytimes.com{href}" if href.startswith('/') else href
                            
                            articles.append({
                                'title': text[:150],
                                'url': url,
                                'summary': text[:300],
                                'keywords_matched': trump_keywords,
                                'relevance_score': len(trump_keywords) / len(self.keywords)
                            })
                            
                    except Exception:
                        continue
            
            # Sort by relevance
            articles.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            return {
                'extraction_method': 'NYTTrumpFetch_HTML_parser',
                'total_found': len(articles),
                'articles': articles[:5],
                'keywords_searched': self.keywords,
                'source': 'New York Times'
            }
            
        except Exception as e:
            return {
                'extraction_method': 'NYTTrumpFetch_HTML_parser',
                'total_found': 0,
                'articles': [],
                'error': str(e),
                'keywords_searched': self.keywords
            }