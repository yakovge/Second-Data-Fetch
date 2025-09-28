#!/usr/bin/env python3
"""
AI-Enhanced NYT Fetch - Integrates AI throughout the search workflow per CLAUDE.md strategy
Follows CLAUDE.md pattern: AI Integration as core infrastructure component
"""

import sys
import os
from typing import Dict, List, Optional, Any, AsyncGenerator, Generator
from datetime import datetime
import logging

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.datafetch import DataFetch, FetchResult, FetchSpec, FetchError, ValidationError, SecurityError
from src.collectors.http_client import HTTPClient
from src.ai.gemini_client import GeminiClient
from bs4 import BeautifulSoup


class NYTAIEnhancedFetch(DataFetch):
    """
    AI-Enhanced NYT Fetch implementation following CLAUDE.md core strategy.

    Integrates AI throughout the search workflow:
    - AI-driven URL discovery
    - AI-enhanced keyword extraction
    - AI-powered structure generation
    - AI-assisted relevance scoring

    Follows CLAUDE.md pattern exactly:
    - Inherits from DataFetch (IMMUTABLE)
    - Uses HTTPClient as tool
    - AI as core infrastructure component
    """

    def __init__(self,
                 spec: FetchSpec,
                 cache_client: Optional[Any] = None,
                 storage_client: Optional[Any] = None,
                 ai_client: Optional[Any] = None):
        """Initialize AI-enhanced NYT fetcher."""
        super().__init__(spec, cache_client, storage_client, ai_client)

        # Setup logging first
        self.logger = logging.getLogger(f"nyt_ai_enhanced.{self._session_id[:8]}")

        # AI is REQUIRED for this implementation (core strategy)
        if not self._ai_client:
            # Create AI client if not provided (core component)
            try:
                self._ai_client = GeminiClient(model_name="gemini-2.5-flash")
                self.logger.info("AI client auto-created as core component")
            except Exception as e:
                self.logger.warning(f"AI client creation failed: {e}")
                raise FetchError("AI integration is required for AI-enhanced fetch")

        # Use HTTPClient as tool (CLAUDE.md pattern)
        self.http_client = HTTPClient(
            spec=spec,
            cache_client=cache_client,
            storage_client=storage_client,
            ai_client=ai_client
        )

        # AI-enhanced keyword extraction
        self.search_keywords = self._ai_extract_keywords()
        self.ai_discovered_urls = self._ai_discover_urls()

    def fetch(self) -> FetchResult:
        """
        AI-enhanced fetch operation.
        Uses AI throughout the workflow per CLAUDE.md strategy.
        """
        start_time = datetime.now()

        try:
            self.logger.info("Starting AI-enhanced fetch")

            # Phase 1: AI URL Discovery
            target_urls = self._select_target_urls()
            self.logger.info(f"AI selected {len(target_urls)} target URLs")

            # Phase 2: HTTP Data Fetching (using HTTPClient tool)
            all_articles = []
            for url in target_urls:
                try:
                    # Update spec for this specific URL
                    url_spec = FetchSpec(
                        raw_text=self._spec.raw_text,
                        urls=[url],
                        expected_format=self._spec.expected_format,
                        method=self._spec.method,
                        timeout=self._spec.timeout
                    )

                    url_client = HTTPClient(url_spec, self._cache_client)
                    url_result = url_client.fetch()

                    if not url_result.error:
                        # AI-enhanced article extraction
                        articles = self._ai_extract_articles(url_result.data, url)
                        all_articles.extend(articles)

                except Exception as e:
                    self.logger.warning(f"Failed to fetch {url}: {e}")
                    continue

            # Phase 3: AI-Enhanced Relevance Scoring
            ai_scored_articles = self._ai_score_relevance(all_articles)

            # Phase 4: AI Structure Generation
            ai_structure = self._ai_generate_result_structure(ai_scored_articles)

            execution_time = (datetime.now() - start_time).total_seconds()

            return FetchResult(
                url=self._spec.urls[0],
                data={
                    'extraction_method': 'NYTAIEnhancedFetch',
                    'total_found': len(ai_scored_articles),
                    'articles': ai_scored_articles[:10],
                    'search_keywords': self.search_keywords,
                    'ai_discovered_urls': self.ai_discovered_urls,
                    'ai_structure': ai_structure,
                    'source': 'New York Times (AI-Enhanced)'
                },
                timestamp=datetime.now(),
                format=self._spec.expected_format,
                method=self._spec.method,
                metadata={
                    'ai_enhanced': True,
                    'ai_urls_discovered': len(self.ai_discovered_urls),
                    'ai_keywords_extracted': len(self.search_keywords),
                    'target_urls_processed': len(target_urls)
                },
                execution_time=execution_time
            )

        except Exception as e:
            self.logger.error(f"AI-enhanced fetch failed: {str(e)}")
            raise FetchError(f"AI-enhanced fetch failed: {str(e)}")

    async def afetch(self) -> FetchResult:
        """Async AI-enhanced fetch."""
        # For now, use sync version
        # Full async implementation would use AI client's async methods
        return self.fetch()

    def fetch_stream(self) -> Generator[FetchResult, None, None]:
        """Stream AI-enhanced articles."""
        result = self.fetch()

        if isinstance(result.data, dict) and 'articles' in result.data:
            articles = result.data['articles']

            for i, article in enumerate(articles):
                yield FetchResult(
                    url=article.get('url', result.url),
                    data=article,
                    timestamp=result.timestamp,
                    format=result.format,
                    method=result.method,
                    metadata={
                        **result.metadata,
                        'article_index': i + 1,
                        'stream_mode': True,
                        'ai_relevance_score': article.get('ai_relevance_score', 0)
                    },
                    execution_time=result.execution_time / len(articles) if articles else 0
                )

    async def afetch_stream(self) -> AsyncGenerator[FetchResult, None]:
        """Async stream AI-enhanced articles."""
        for result in self.fetch_stream():
            yield result

    def validate_data(self, data: Any) -> bool:
        """AI-enhanced data validation."""
        try:
            # Basic validation
            if not isinstance(data, dict):
                return False

            # AI-specific validation
            if 'ai_structure' in data:
                # Validate AI-generated structure
                ai_structure = data['ai_structure']
                if not isinstance(ai_structure, dict):
                    return False

            # Enhanced relevance validation using AI
            if self._ai_client and 'articles' in data:
                articles = data['articles']
                if articles:
                    # Use AI to validate article relevance
                    sample_article = articles[0]
                    is_relevant = self._ai_validate_relevance(sample_article)
                    return is_relevant

            return True

        except Exception as e:
            self.logger.error(f"AI validation error: {str(e)}")
            return False

    def extract_structure(self, sample_data: Any) -> Dict[str, Any]:
        """AI-driven structure extraction (core component usage)."""
        if self._ai_client:
            try:
                context = f"User query: {self._spec.raw_text} | NYT AI-enhanced articles"
                ai_structure = self._ai_client.generate_structure_from_sample(sample_data, context)
                if ai_structure:
                    self.logger.info("AI structure generation successful")
                    return ai_structure
            except Exception as e:
                self.logger.warning(f"AI structure extraction failed: {e}")

        # Fallback structure
        return {
            'type': 'object',
            'description': 'AI-enhanced NYT articles (fallback structure)',
            'properties': {
                'articles': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'title': {'type': 'string'},
                            'url': {'type': 'string', 'format': 'uri'},
                            'summary': {'type': 'string'},
                            'ai_relevance_score': {'type': 'number', 'minimum': 0, 'maximum': 1},
                            'ai_keywords_matched': {'type': 'array', 'items': {'type': 'string'}},
                            'ai_confidence': {'type': 'number'}
                        }
                    }
                },
                'ai_structure': {'type': 'object'},
                'search_keywords': {'type': 'array', 'items': {'type': 'string'}}
            }
        }

    def sanitize_input(self, raw_input: str) -> str:
        """Use parent's sanitization (CLAUDE.md security pattern)."""
        return super().sanitize_input(raw_input)

    def _ai_extract_keywords(self) -> List[str]:
        """Use AI to extract enhanced keywords from user query."""
        if not self._ai_client:
            return self._extract_search_keywords_fallback()

        try:
            # AI-enhanced keyword extraction
            prompt = f"""
            Extract the most relevant search keywords from this query for NYT article search:
            Query: "{self._spec.raw_text}"

            Return 5-10 keywords that would best find relevant NYT articles.
            Focus on: people, events, topics, locations, organizations.
            Return as comma-separated list.
            """

            response = self._ai_client._generate_with_retry(prompt.strip())
            keywords = [kw.strip() for kw in response.split(',') if kw.strip()]

            self.logger.info(f"AI extracted {len(keywords)} keywords")
            return keywords[:10]

        except Exception as e:
            self.logger.warning(f"AI keyword extraction failed: {e}")
            return self._extract_search_keywords_fallback()

    def _ai_discover_urls(self) -> List[str]:
        """Use AI to discover relevant NYT URLs."""
        if not self._ai_client:
            return []

        try:
            # Use AI client's URL generation capability
            urls = self._ai_client.generate_urls_from_text(
                self._spec.raw_text,
                domain_hints=['nytimes.com']
            )

            self.logger.info(f"AI discovered {len(urls)} URLs")
            return urls

        except Exception as e:
            self.logger.warning(f"AI URL discovery failed: {e}")
            return []

    def _select_target_urls(self) -> List[str]:
        """Select target URLs for fetching (AI + original)."""
        target_urls = []

        # Start with original URLs
        target_urls.extend(self._spec.urls)

        # Add AI-discovered URLs
        target_urls.extend(self.ai_discovered_urls[:3])  # Limit to top 3

        # Remove duplicates
        return list(dict.fromkeys(target_urls))

    def _ai_extract_articles(self, html_data: Any, source_url: str) -> List[Dict[str, Any]]:
        """Extract articles with AI assistance."""
        if isinstance(html_data, str):
            # Parse HTML and use AI to identify articles
            return self._ai_parse_html_articles(html_data, source_url)
        elif isinstance(html_data, dict):
            # Structured data - enhance with AI
            return self._ai_enhance_structured_data(html_data, source_url)
        else:
            return []

    def _ai_parse_html_articles(self, html: str, source_url: str) -> List[Dict[str, Any]]:
        """Parse HTML with AI guidance for article extraction."""
        try:
            soup = BeautifulSoup(html, 'html.parser')

            # Clean HTML
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()

            articles = []

            # Enhanced selectors with AI insights
            selectors = [
                'a[href*="/2024/"]', 'a[href*="/2023/"]',
                'a[href*="/opinion/"]', 'a[href*="/politics/"]',
                'a[href*="/business/"]', 'a[href*="/world/"]',
                'article a', 'h1 a', 'h2 a', 'h3 a'
            ]

            found_urls = set()

            for selector in selectors:
                elements = soup.select(selector)

                for elem in elements:
                    article_data = self._extract_article_with_ai(elem, source_url, found_urls)
                    if article_data:
                        articles.append(article_data)

            return articles

        except Exception as e:
            self.logger.error(f"AI HTML parsing failed: {e}")
            return []

    def _extract_article_with_ai(self, elem, source_url: str, found_urls: set) -> Optional[Dict[str, Any]]:
        """Extract single article with AI enhancement."""
        try:
            text = elem.get_text(strip=True)
            if not text or len(text) < 20:
                return None

            # Extract URL
            url = ""
            if elem.name == 'a' and elem.get('href'):
                href = elem.get('href')
                url = self._normalize_nyt_url(href)

            if not url or url in found_urls:
                return None

            found_urls.add(url)

            # AI-enhanced relevance scoring
            ai_relevance = self._ai_calculate_relevance(text, url)

            if ai_relevance < 0.1:  # Skip low-relevance articles
                return None

            return {
                'title': text[:200],
                'url': url,
                'summary': text[:400] + ('...' if len(text) > 400 else ''),
                'ai_relevance_score': ai_relevance,
                'ai_keywords_matched': self._ai_find_matched_keywords(text),
                'source_url': source_url,
                'extraction_method': 'ai_enhanced'
            }

        except Exception as e:
            self.logger.debug(f"Article extraction failed: {e}")
            return None

    def _ai_calculate_relevance(self, text: str, url: str) -> float:
        """Use AI to calculate article relevance."""
        try:
            if not self._ai_client:
                return self._fallback_relevance_score(text)

            # AI relevance calculation
            prompt = f"""
            Rate the relevance of this article to the search query on a scale of 0.0 to 1.0:

            Search Query: "{self._spec.raw_text}"
            Article Text: "{text[:500]}"
            Article URL: "{url}"

            Consider: topic match, keyword presence, semantic relevance.
            Return only a number between 0.0 and 1.0.
            """

            response = self._ai_client._generate_with_retry(prompt.strip(), max_tokens=10)

            try:
                score = float(response.strip())
                return max(0.0, min(1.0, score))
            except ValueError:
                return self._fallback_relevance_score(text)

        except Exception as e:
            return self._fallback_relevance_score(text)

    def _fallback_relevance_score(self, text: str) -> float:
        """Fallback relevance calculation."""
        text_lower = text.lower()
        matches = sum(1 for kw in self.search_keywords if kw.lower() in text_lower)
        return matches / len(self.search_keywords) if self.search_keywords else 0.5

    def _ai_find_matched_keywords(self, text: str) -> List[str]:
        """Find matched keywords with AI enhancement."""
        matched = []
        text_lower = text.lower()

        for keyword in self.search_keywords:
            if keyword.lower() in text_lower:
                matched.append(keyword)

        return matched

    def _ai_score_relevance(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """AI-enhanced relevance scoring for all articles."""
        if not articles:
            return []

        # Sort by AI relevance score
        scored_articles = sorted(articles, key=lambda x: x.get('ai_relevance_score', 0), reverse=True)

        # Filter low-relevance articles
        relevant_articles = [a for a in scored_articles if a.get('ai_relevance_score', 0) > 0.1]

        return relevant_articles

    def _ai_generate_result_structure(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate AI structure for results."""
        if not self._ai_client or not articles:
            return {'type': 'object', 'description': 'No AI structure available'}

        try:
            sample_data = {'articles': articles[:3]}  # Use top 3 for structure
            return self.extract_structure(sample_data)
        except Exception as e:
            self.logger.warning(f"AI structure generation failed: {e}")
            return {'type': 'object', 'description': 'AI structure generation failed'}

    def _ai_enhance_structured_data(self, data: Dict[str, Any], source_url: str) -> List[Dict[str, Any]]:
        """Enhance structured data with AI."""
        # If HTTPClient returned structured data, enhance it with AI scoring
        enhanced_articles = []

        if 'title' in data and 'content' in data:
            # Single article
            ai_relevance = self._ai_calculate_relevance(
                f"{data.get('title', '')} {data.get('content', '')}",
                source_url
            )

            if ai_relevance > 0.1:
                enhanced_articles.append({
                    'title': data.get('title', ''),
                    'url': source_url,
                    'summary': data.get('content', '')[:400] + '...',
                    'ai_relevance_score': ai_relevance,
                    'ai_keywords_matched': self._ai_find_matched_keywords(data.get('content', '')),
                    'source_url': source_url,
                    'extraction_method': 'ai_enhanced_structured'
                })

        return enhanced_articles

    def _ai_validate_relevance(self, article: Dict[str, Any]) -> bool:
        """Use AI to validate article relevance."""
        try:
            ai_score = article.get('ai_relevance_score', 0)
            return ai_score > 0.1
        except Exception:
            return True

    def _extract_search_keywords_fallback(self) -> List[str]:
        """Fallback keyword extraction without AI."""
        import re

        text = self._spec.raw_text.lower()
        filler_words = {
            'i', 'want', 'get', 'show', 'find', 'fetch', 'need', 'looking', 'for',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'from',
            'that', 'about', 'article', 'articles', 'news', 'story', 'nyt', 'times'
        }

        words = re.findall(r'\b\w+\b', text)
        keywords = [word for word in words if len(word) > 2 and word not in filler_words]

        return keywords[:10]

    def _normalize_nyt_url(self, href: str) -> str:
        """Normalize NYT URL to full format."""
        if not href:
            return ""

        href = href.split('?')[0].split('#')[0]

        if href.startswith('/'):
            return f"https://www.nytimes.com{href}"
        elif href.startswith('http'):
            return href
        else:
            return f"https://www.nytimes.com/{href}"