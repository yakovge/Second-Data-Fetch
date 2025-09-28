#!/usr/bin/env python3
"""
NYT General Article Fetcher - DataFetch Implementation for Any NYT Content
Follows CLAUDE.md strategy exactly: Inherits from DataFetch, uses HTTPClient as tool, implements all abstract methods.
"""

import sys
import os
from typing import Dict, List, Optional, Any, AsyncGenerator, Generator
from datetime import datetime
import re
import logging

# Add project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.datafetch import DataFetch, FetchResult, FetchSpec, FetchError, ValidationError, SecurityError
from src.collectors.http_client import HTTPClient
from bs4 import BeautifulSoup


class NYTGeneralFetch(DataFetch):
    """
    DataFetch implementation for general NYT article searching.

    This follows the CLAUDE.md pattern exactly:
    - Inherits from DataFetch abstract class (IMMUTABLE)
    - Uses HTTPClient as a tool (doesn't modify it)
    - Implements all abstract methods
    - Custom logic for general NYT article extraction
    - Supports any search topic via raw_text parsing
    """

    def __init__(self,
                 spec: FetchSpec,
                 cache_client: Optional[Any] = None,
                 storage_client: Optional[Any] = None,
                 ai_client: Optional[Any] = None):
        """Initialize NYT general fetcher following CLAUDE.md pattern."""
        super().__init__(spec, cache_client, storage_client, ai_client)

        # Use HTTPClient as a tool (CLAUDE.md pattern - don't modify existing components)
        self.http_client = HTTPClient(
            spec=spec,
            cache_client=cache_client,
            storage_client=storage_client,
            ai_client=ai_client
        )

        # Extract search keywords from user's raw_text
        self.search_keywords = self._extract_search_keywords()
        self.logger = logging.getLogger(f"nyt_general.{self._session_id[:8]}")

    def fetch(self) -> FetchResult:
        """
        Fetch articles from NYT based on user's search criteria.
        Uses HTTPClient then processes for relevant content.
        """
        try:
            # Use HTTPClient to get raw data (existing component, don't modify)
            raw_result = self.http_client.fetch()

            if raw_result.error:
                raise FetchError(f"HTTPClient failed: {raw_result.error}")

            # If HTTPClient parsed it to structured data, we need the raw HTML to extract URLs
            # Force getting raw HTML by using the requests session directly
            if isinstance(raw_result.data, dict) and not isinstance(raw_result.data, str):
                # Get raw HTML for URL extraction
                raw_html = self._get_raw_html_from_httpclient()
                extracted_articles = self._extract_relevant_articles(raw_html)
            else:
                # Process the data for relevant articles
                extracted_articles = self._extract_relevant_articles(raw_result.data)

            # Create new FetchResult with processed data
            return FetchResult(
                url=raw_result.url,
                data=extracted_articles,
                timestamp=raw_result.timestamp,
                format=raw_result.format,
                method=raw_result.method,
                metadata={
                    **raw_result.metadata,
                    'extraction_method': 'NYTGeneralFetch',
                    'articles_found': len(extracted_articles.get('articles', [])),
                    'search_keywords': self.search_keywords,
                    'user_query': self._spec.raw_text
                },
                cache_hit=raw_result.cache_hit,
                execution_time=raw_result.execution_time
            )

        except Exception as e:
            self.logger.error(f"NYT general fetch failed: {str(e)}")
            raise FetchError(f"NYT general fetch failed: {str(e)}")

    async def afetch(self) -> FetchResult:
        """Async version of fetch using HTTPClient's async capabilities."""
        try:
            # Use HTTPClient's async method (existing component)
            raw_result = await self.http_client.afetch()

            if raw_result.error:
                raise FetchError(f"HTTPClient async failed: {raw_result.error}")

            # Process the data for relevant articles
            extracted_articles = self._extract_relevant_articles(raw_result.data)

            return FetchResult(
                url=raw_result.url,
                data=extracted_articles,
                timestamp=raw_result.timestamp,
                format=raw_result.format,
                method=raw_result.method,
                metadata={
                    **raw_result.metadata,
                    'extraction_method': 'NYTGeneralFetch_async',
                    'articles_found': len(extracted_articles.get('articles', [])),
                    'search_keywords': self.search_keywords
                },
                cache_hit=raw_result.cache_hit,
                execution_time=raw_result.execution_time
            )

        except Exception as e:
            self.logger.error(f"NYT async fetch failed: {str(e)}")
            raise FetchError(f"NYT async fetch failed: {str(e)}")

    def fetch_stream(self) -> Generator[FetchResult, None, None]:
        """Stream articles one by one for large result sets."""
        try:
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
                            'total_articles': len(articles),
                            'stream_mode': True
                        },
                        cache_hit=result.cache_hit,
                        execution_time=result.execution_time / len(articles) if articles else 0
                    )
            else:
                yield result

        except Exception as e:
            self.logger.error(f"Stream fetch failed: {str(e)}")
            yield FetchResult(
                url=self._spec.urls[0] if self._spec.urls else "unknown",
                data=None,
                timestamp=datetime.now(),
                format=self._spec.expected_format,
                method=self._spec.method,
                metadata={'error': str(e)},
                error=str(e)
            )

    async def afetch_stream(self) -> AsyncGenerator[FetchResult, None]:
        """Async stream version using HTTPClient's streaming capabilities."""
        try:
            # Use HTTPClient's async stream (existing component)
            async for raw_result in self.http_client.afetch_stream():
                if raw_result.error:
                    yield raw_result
                    continue

                # Process each result
                extracted_articles = self._extract_relevant_articles(raw_result.data)

                yield FetchResult(
                    url=raw_result.url,
                    data=extracted_articles,
                    timestamp=raw_result.timestamp,
                    format=raw_result.format,
                    method=raw_result.method,
                    metadata={
                        **raw_result.metadata,
                        'extraction_method': 'NYTGeneralFetch_async_stream',
                        'articles_found': len(extracted_articles.get('articles', []))
                    },
                    cache_hit=raw_result.cache_hit,
                    execution_time=raw_result.execution_time
                )

        except Exception as e:
            self.logger.error(f"Async stream failed: {str(e)}")
            yield FetchResult(
                url="unknown",
                data=None,
                timestamp=datetime.now(),
                format=self._spec.expected_format,
                method=self._spec.method,
                metadata={'error': str(e)},
                error=str(e)
            )

    def validate_data(self, data: Any) -> bool:
        """Validate that we got relevant NYT articles."""
        try:
            if not isinstance(data, dict):
                return False

            # Check required fields for our structure
            required_fields = ['articles', 'extraction_method', 'total_found']
            if not all(field in data for field in required_fields):
                return False

            # Check that articles is a list
            articles = data.get('articles', [])
            if not isinstance(articles, list):
                return False

            # If we have search keywords, check relevance
            if self.search_keywords and articles:
                relevant_count = 0
                for article in articles:
                    if self._article_is_relevant(article):
                        relevant_count += 1

                # At least some articles should be relevant
                return relevant_count > 0

            # If no specific search keywords, any articles are valid
            return True

        except Exception as e:
            self.logger.error(f"Data validation error: {str(e)}")
            return False

    def extract_structure(self, sample_data: Any) -> Dict[str, Any]:
        """Extract data structure from sample NYT data."""

        # Use AI if available (CLAUDE.md pattern - use existing AI client)
        if self._ai_client:
            try:
                context = f"User query: {self._spec.raw_text} | NYT general articles"
                ai_structure = self._ai_client.generate_structure_from_sample(sample_data, context)
                if ai_structure and isinstance(ai_structure, dict):
                    return ai_structure
            except Exception as e:
                self.logger.warning(f"AI structure extraction failed: {e}")

        # Fallback: Define general NYT article structure
        if isinstance(sample_data, dict) and 'articles' in sample_data:
            return {
                'type': 'object',
                'description': 'NYT articles matching search criteria',
                'properties': {
                    'articles': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'title': {'type': 'string', 'description': 'Article headline'},
                                'url': {'type': 'string', 'format': 'uri', 'description': 'Article URL'},
                                'summary': {'type': 'string', 'description': 'Article summary/excerpt'},
                                'author': {'type': 'string', 'description': 'Article author'},
                                'published_date': {'type': 'string', 'description': 'Publication date'},
                                'section': {'type': 'string', 'description': 'NYT section (e.g., Business, Opinion)'},
                                'relevance_score': {'type': 'number', 'description': 'Relevance to search (0-1)'},
                                'keywords_matched': {'type': 'array', 'items': {'type': 'string'}}
                            },
                            'required': ['title', 'url']
                        }
                    },
                    'total_found': {'type': 'integer'},
                    'search_keywords': {'type': 'array', 'items': {'type': 'string'}},
                    'extraction_method': {'type': 'string'}
                },
                'required': ['articles', 'total_found']
            }
        else:
            return {'type': 'object', 'description': 'No articles found'}

    def sanitize_input(self, raw_input: str) -> str:
        """Sanitize input using parent class method (CLAUDE.md pattern - use existing security)."""
        # Use parent's sanitization (existing security from DataFetch)
        return super().sanitize_input(raw_input)

    def _extract_search_keywords(self) -> List[str]:
        """Extract search keywords from user's raw_text."""
        if not self._spec.raw_text:
            return []

        text = self._spec.raw_text.lower()

        # Remove common filler words
        filler_words = {
            'i', 'want', 'get', 'show', 'find', 'fetch', 'need', 'looking', 'for',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'from',
            'that', 'about', 'article', 'articles', 'news', 'story', 'nyt', 'times'
        }

        # Extract meaningful words
        words = re.findall(r'\b\w+\b', text)
        keywords = [word for word in words if len(word) > 2 and word not in filler_words]

        # Return top 10 keywords
        return keywords[:10]

    def _get_raw_html_from_httpclient(self) -> str:
        """Get raw HTML from NYT bypassing HTTPClient's parsing."""
        try:
            import requests
            session = requests.Session()
            session.headers.update(self.http_client.config.headers)

            response = session.get(
                self._spec.urls[0],
                timeout=self.http_client.config.timeout,
                verify=self.http_client.config.verify_ssl
            )
            response.raise_for_status()
            return response.text

        except Exception as e:
            self.logger.error(f"Failed to get raw HTML: {e}")
            return ""

    def _extract_relevant_articles(self, html_or_data: Any) -> Dict[str, Any]:
        """Extract relevant articles from HTML or structured data."""

        # If HTTPClient already processed it into structured data, filter it
        if isinstance(html_or_data, dict) and 'articles' in html_or_data:
            all_articles = html_or_data['articles']
            relevant_articles = []

            for article in all_articles:
                if self._article_is_relevant(article):
                    relevant_articles.append(article)

            return {
                'extraction_method': 'NYTGeneralFetch',
                'total_found': len(relevant_articles),
                'articles': relevant_articles[:10],  # Top 10
                'search_keywords': self.search_keywords,
                'source': 'New York Times'
            }

        # If HTTPClient returned structured data with title/content
        elif isinstance(html_or_data, dict) and ('title' in html_or_data or 'content' in html_or_data):
            if self._article_is_relevant(html_or_data):
                return {
                    'extraction_method': 'NYTGeneralFetch',
                    'total_found': 1,
                    'articles': [html_or_data],
                    'search_keywords': self.search_keywords,
                    'source': 'New York Times'
                }
            else:
                return {
                    'extraction_method': 'NYTGeneralFetch',
                    'total_found': 0,
                    'articles': [],
                    'search_keywords': self.search_keywords,
                    'source': 'New York Times'
                }

        # If we got raw HTML, parse it ourselves
        elif isinstance(html_or_data, str) and '<html' in html_or_data.lower():
            return self._parse_html_for_articles(html_or_data)

        else:
            return {
                'extraction_method': 'NYTGeneralFetch',
                'total_found': 0,
                'articles': [],
                'error': 'Unable to process data format',
                'search_keywords': self.search_keywords
            }

    def _article_is_relevant(self, article: Dict[str, Any]) -> bool:
        """Check if article is relevant to search keywords."""
        if not self.search_keywords:
            return True  # No specific keywords = all articles relevant

        article_text = str(article).lower()
        matches = sum(1 for keyword in self.search_keywords if keyword in article_text)

        # Article is relevant if it matches at least 1 keyword
        return matches > 0

    def _parse_html_for_articles(self, html: str) -> Dict[str, Any]:
        """Parse HTML for articles (fallback method using BeautifulSoup)."""
        try:
            soup = BeautifulSoup(html, 'html.parser')

            # Clean up HTML (remove noise)
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                element.decompose()

            articles = []
            found_titles = set()

            # Enhanced NYT-specific selectors for article links
            selectors = [
                'a[href*="/2024/"]',  # 2024 article URLs
                'a[href*="/2023/"]',  # 2023 article URLs
                'a[href*="/interactive/"]',  # Interactive articles
                'a[href*="/opinion/"]',  # Opinion pieces
                'a[href*="/business/"]',  # Business section
                'a[href*="/world/"]',  # World news
                'a[href*="/us/"]',  # US news
                'a[href*="/politics/"]',  # Politics
                'a[href*="/technology/"]',  # Technology
                'a[href^="/"]',  # All relative NYT links
                'article a',  # Links within article elements
                '[data-testid*="headline"] a',  # Headlines with links
                'h1 a, h2 a, h3 a',  # Headline links
            ]

            for selector in selectors:
                try:
                    elements = soup.select(selector)

                    for elem in elements:
                        article_data = self._extract_article_from_element(elem, found_titles)
                        if article_data:
                            articles.append(article_data)

                except Exception as e:
                    self.logger.debug(f"Selector {selector} failed: {e}")
                    continue

            # Filter for relevance and remove duplicates
            relevant_articles = []
            for article in articles:
                if self._article_is_relevant(article):
                    relevant_articles.append(article)

            # Sort by relevance score
            relevant_articles.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)

            return {
                'extraction_method': 'NYTGeneralFetch_HTML_parser',
                'total_found': len(relevant_articles),
                'articles': relevant_articles[:10],
                'search_keywords': self.search_keywords,
                'source': 'New York Times'
            }

        except Exception as e:
            self.logger.error(f"HTML parsing failed: {str(e)}")
            return {
                'extraction_method': 'NYTGeneralFetch_HTML_parser',
                'total_found': 0,
                'articles': [],
                'error': str(e),
                'search_keywords': self.search_keywords
            }

    def _extract_article_from_element(self, elem, found_titles: set) -> Optional[Dict[str, Any]]:
        """Extract article data from a single HTML element."""
        try:
            text = elem.get_text(strip=True)
            if not text or len(text) < 20 or text in found_titles:
                return None

            found_titles.add(text)

            # Calculate relevance
            text_lower = text.lower()
            matched_keywords = [kw for kw in self.search_keywords if kw in text_lower]
            relevance_score = len(matched_keywords) / len(self.search_keywords) if self.search_keywords else 0.5

            # Skip if not relevant enough
            if self.search_keywords and relevance_score == 0:
                return None

            # Extract URL - check element itself and parent elements
            url = ""
            if elem.name == 'a' and elem.get('href'):
                href = elem.get('href')
                url = self._normalize_nyt_url(href)
            else:
                # Look for parent link or child links
                parent_link = elem.find_parent('a')
                if parent_link and parent_link.get('href'):
                    url = self._normalize_nyt_url(parent_link.get('href'))
                else:
                    # Look for child links
                    child_link = elem.find('a')
                    if child_link and child_link.get('href'):
                        url = self._normalize_nyt_url(child_link.get('href'))

            # Determine if this is a headline or content
            is_headline = elem.name in ['h1', 'h2', 'h3'] or 'headline' in str(elem.get('class', []))

            return {
                'title': text[:200] if is_headline else text[:100],
                'url': url,
                'summary': text[:400] + ('...' if len(text) > 400 else ''),
                'keywords_matched': matched_keywords,
                'relevance_score': relevance_score,
                'element_type': elem.name,
                'is_headline': is_headline
            }

        except Exception as e:
            self.logger.debug(f"Element extraction failed: {e}")
            return None

    def _normalize_nyt_url(self, href: str) -> str:
        """Normalize NYT URL to full format."""
        if not href:
            return ""

        # Remove query parameters and fragments for cleaner URLs
        href = href.split('?')[0].split('#')[0]

        if href.startswith('/'):
            return f"https://www.nytimes.com{href}"
        elif href.startswith('http'):
            return href
        else:
            return f"https://www.nytimes.com/{href}"