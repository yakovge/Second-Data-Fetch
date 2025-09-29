import asyncio
import time
import json
from typing import Dict, List, Optional, Any, AsyncGenerator, Generator
from datetime import datetime
from dataclasses import dataclass
import logging
from pathlib import Path

from playwright.async_api import async_playwright, Browser, BrowserContext, Page
from playwright.sync_api import sync_playwright, Browser as SyncBrowser, BrowserContext as SyncBrowserContext, Page as SyncPage

from ..core.datafetch import DataFetch, FetchResult, FetchSpec, DataFormat, FetchMethod, FetchError, ValidationError


@dataclass
class BrowserConfig:
    headless: bool
    viewport_width: int
    viewport_height: int
    user_agent: str
    timeout: int
    wait_for_load_state: str
    wait_for_selector: Optional[str]
    javascript_enabled: bool
    images_enabled: bool
    block_resources: List[str]


class BrowserClient(DataFetch):
    """
    Browser client implementation using Playwright.
    
    Designed for dynamic content and JavaScript-heavy news websites.
    Includes HAR file generation, screenshot capture, and advanced waiting strategies.
    """
    
    DEFAULT_USER_AGENTS = {
        'chrome': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'firefox': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0',
        'safari': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2.1 Safari/605.1.15'
    }
    
    BLOCK_RESOURCES = [
        'image',
        'font',
        'media',
        'websocket'
    ]
    
    def __init__(self, 
                 spec: FetchSpec,
                 cache_client: Optional[Any] = None,
                 storage_client: Optional[Any] = None,
                 ai_client: Optional[Any] = None,
                 browser_type: str = 'chromium'):
        super().__init__(spec, cache_client, storage_client, ai_client)
        
        self.browser_type = browser_type
        self.config = self._create_browser_config()
        self.browser = None
        self.context = None
        self._setup_logging()
        self._har_data = {}
    
    def _create_browser_config(self) -> BrowserConfig:
        """Create browser configuration from spec."""
        # Determine optimal settings based on URLs
        is_news_site = any(
            news_domain in url.lower() 
            for url in self._spec.urls 
            for news_domain in ['news', 'reuters', 'nytimes', 'bbc', 'cnn', 'guardian']
        )
        
        return BrowserConfig(
            headless=True,
            viewport_width=1920,
            viewport_height=1080,
            user_agent=self.DEFAULT_USER_AGENTS['chrome'],
            timeout=max(45000, self._spec.timeout * 1000),  # Increase timeout for news sites
            wait_for_load_state='domcontentloaded',  # Less strict than networkidle
            wait_for_selector=None,
            javascript_enabled=True,
            images_enabled=not is_news_site,  # Disable images for news sites to speed up
            block_resources=self.BLOCK_RESOURCES if is_news_site else []
        )
    
    def _setup_logging(self):
        """Setup logging for browser client."""
        self.logger = logging.getLogger(f"datafetch.browser.{self._session_id[:8]}")
        self.logger.setLevel(logging.INFO)
    
    def fetch(self) -> FetchResult:
        """
        Synchronous fetch operation using Playwright.
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
            self.logger.info(f"Browser fetching URL: {url}")
            
            result = self._fetch_single_url_sync(url)
            
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
            self.logger.error(f"Browser fetch failed: {str(e)}")
            execution_time = time.time() - start_time
            return FetchResult(
                url=self._spec.urls[0],
                data=None,
                timestamp=datetime.now(),
                format=self._spec.expected_format,
                method=FetchMethod.PLAYWRIGHT,
                metadata={'error': str(e)},
                error=str(e),
                execution_time=execution_time
            )

    def fetch_all(self) -> FetchResult:
        """
        Fetch from ALL URLs in the spec using browser automation and combine results.
        This addresses the multi-URL architecture issue.
        """
        start_time = time.time()
        all_data = []
        successful_urls = []
        failed_urls = []
        total_fetch_time = 0

        self.logger.info(f"Browser fetching from {len(self._spec.urls)} URLs")

        for i, url in enumerate(self._spec.urls, 1):
            try:
                self.logger.info(f"Browser processing URL {i}/{len(self._spec.urls)}: {url}")
                print(f"   [BROWSER] Processing URL {i}/{len(self._spec.urls)}: {url}")

                # Try cache first
                cached_result = None
                if self._cache_client:
                    cached_result = self._try_cache(url)

                if cached_result:
                    result = cached_result
                    self.logger.info(f"Cache hit for {url}")
                    print(f"   [BROWSER] Cache hit for {url}")
                else:
                    # Fetch individual URL
                    result = self._fetch_single_url_sync(url)

                    # Store in cache if successful
                    if self._cache_client and result.error is None:
                        self._store_in_cache(url, result)

                if result.error is None:
                    # Combine data from this URL
                    data_size = len(result.data) if isinstance(result.data, (list, tuple)) else 1
                    if isinstance(result.data, list):
                        all_data.extend(result.data)
                    else:
                        all_data.append(result.data)
                    successful_urls.append(url)
                    self.logger.info(f"Successfully browser-fetched from {url}")
                    print(f"   [BROWSER] SUCCESS: {url} -> {data_size} items")
                else:
                    failed_urls.append({'url': url, 'error': result.error})
                    self.logger.warning(f"Failed to browser-fetch from {url}: {result.error}")
                    print(f"   [BROWSER] FAILED: {url} -> {result.error}")

                total_fetch_time += result.execution_time

            except Exception as e:
                self.logger.error(f"Browser exception fetching {url}: {str(e)}")
                print(f"   [BROWSER] EXCEPTION: {url} -> {str(e)}")
                failed_urls.append({'url': url, 'error': str(e)})

        # Record metrics
        execution_time = time.time() - start_time
        self._record_metric('total_execution_time', execution_time)
        self._record_metric('urls_fetched', len(successful_urls))
        self._record_metric('urls_failed', len(failed_urls))

        # Create combined result
        primary_url = successful_urls[0] if successful_urls else self._spec.urls[0]

        return FetchResult(
            url=primary_url,
            data=all_data,
            timestamp=datetime.now(),
            format=self._spec.expected_format,
            method=FetchMethod.PLAYWRIGHT,
            metadata={
                'total_urls': len(self._spec.urls),
                'successful_urls': successful_urls,
                'failed_urls': failed_urls,
                'combined_results': True
            },
            error=None if successful_urls else f"All {len(failed_urls)} URLs failed",
            execution_time=execution_time
        )

    async def afetch(self) -> FetchResult:
        """
        Asynchronous fetch operation using Playwright.
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
            self.logger.info(f"Async browser fetching URL: {url}")
            
            result = await self._fetch_single_url_async(url)
            
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
            self.logger.error(f"Async browser fetch failed: {str(e)}")
            execution_time = time.time() - start_time
            return FetchResult(
                url=self._spec.urls[0],
                data=None,
                timestamp=datetime.now(),
                format=self._spec.expected_format,
                method=FetchMethod.PLAYWRIGHT,
                metadata={'error': str(e)},
                error=str(e),
                execution_time=execution_time
            )
    
    def fetch_stream(self) -> Generator[FetchResult, None, None]:
        """
        Streaming fetch operation for multiple URLs (sync).
        """
        with sync_playwright() as playwright:
            browser = self._launch_sync_browser(playwright)
            context = self._create_sync_context(browser)
            
            try:
                for url in self._spec.urls:
                    try:
                        self.logger.info(f"Streaming browser fetch URL: {url}")
                        result = self._fetch_with_sync_context(context, url)
                        yield result
                    except Exception as e:
                        self.logger.error(f"Stream fetch failed for {url}: {str(e)}")
                        yield FetchResult(
                            url=url,
                            data=None,
                            timestamp=datetime.now(),
                            format=self._spec.expected_format,
                            method=FetchMethod.PLAYWRIGHT,
                            metadata={'error': str(e)},
                            error=str(e)
                        )
            finally:
                context.close()
                browser.close()
    
    async def afetch_stream(self) -> AsyncGenerator[FetchResult, None]:
        """
        Asynchronous streaming fetch operation.
        """
        async with async_playwright() as playwright:
            browser = await self._launch_async_browser(playwright)
            context = await self._create_async_context(browser)
            
            try:
                # Process URLs concurrently with limited concurrency
                semaphore = asyncio.Semaphore(3)  # Limit to 3 concurrent pages
                
                async def fetch_with_semaphore(url: str):
                    async with semaphore:
                        return await self._fetch_with_async_context(context, url)
                
                tasks = [fetch_with_semaphore(url) for url in self._spec.urls]
                
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
                            method=FetchMethod.PLAYWRIGHT,
                            metadata={'error': str(e)},
                            error=str(e)
                        )
            finally:
                await context.close()
                await browser.close()
    
    def _fetch_single_url_sync(self, url: str) -> FetchResult:
        """Sync fetch from single URL with Playwright."""
        with sync_playwright() as playwright:
            browser = self._launch_sync_browser(playwright)
            context = self._create_sync_context(browser)
            
            try:
                return self._fetch_with_sync_context(context, url)
            finally:
                context.close()
                browser.close()
    
    async def _fetch_single_url_async(self, url: str) -> FetchResult:
        """Async fetch from single URL with Playwright."""
        async with async_playwright() as playwright:
            browser = await self._launch_async_browser(playwright)
            context = await self._create_async_context(browser)
            
            try:
                return await self._fetch_with_async_context(context, url)
            finally:
                await context.close()
                await browser.close()
    
    def _launch_sync_browser(self, playwright) -> SyncBrowser:
        """Launch synchronous browser with stealth settings."""
        if self.browser_type == 'chromium':
            # Stealth args to avoid detection
            stealth_args = [
                '--no-sandbox',
                '--disable-dev-shm-usage',
                '--disable-blink-features=AutomationControlled',
                '--disable-web-security',
                '--disable-features=VizDisplayCompositor',
                '--user-agent=' + self.config.user_agent
            ]
            return playwright.chromium.launch(
                headless=self.config.headless,
                args=stealth_args
            )
        elif self.browser_type == 'firefox':
            return playwright.firefox.launch(headless=self.config.headless)
        elif self.browser_type == 'webkit':
            return playwright.webkit.launch(headless=self.config.headless)
        else:
            raise ValueError(f"Unsupported browser type: {self.browser_type}")
    
    async def _launch_async_browser(self, playwright) -> Browser:
        """Launch asynchronous browser with stealth settings."""
        if self.browser_type == 'chromium':
            # Stealth args to avoid detection
            stealth_args = [
                '--no-sandbox',
                '--disable-dev-shm-usage',
                '--disable-blink-features=AutomationControlled',
                '--disable-web-security',
                '--disable-features=VizDisplayCompositor',
                '--user-agent=' + self.config.user_agent
            ]
            return await playwright.chromium.launch(
                headless=self.config.headless,
                args=stealth_args
            )
        elif self.browser_type == 'firefox':
            return await playwright.firefox.launch(headless=self.config.headless)
        elif self.browser_type == 'webkit':
            return await playwright.webkit.launch(headless=self.config.headless)
        else:
            raise ValueError(f"Unsupported browser type: {self.browser_type}")
    
    def _create_sync_context(self, browser: SyncBrowser) -> SyncBrowserContext:
        """Create synchronous browser context."""
        context = browser.new_context(
            viewport={'width': self.config.viewport_width, 'height': self.config.viewport_height},
            user_agent=self.config.user_agent,
            ignore_https_errors=True,
            record_har_path=None  # HAR recording disabled by default
        )
        
        # Block resources if configured
        if self.config.block_resources:
            context.route("**/*", lambda route: self._handle_route_sync(route))
        
        return context
    
    async def _create_async_context(self, browser: Browser) -> BrowserContext:
        """Create asynchronous browser context."""
        context = await browser.new_context(
            viewport={'width': self.config.viewport_width, 'height': self.config.viewport_height},
            user_agent=self.config.user_agent,
            ignore_https_errors=True,
            record_har_path=None  # HAR recording disabled by default
        )
        
        # Block resources if configured
        if self.config.block_resources:
            await context.route("**/*", self._handle_route_async)
        
        return context
    
    def _handle_route_sync(self, route):
        """Handle route for resource blocking (sync)."""
        if route.request.resource_type in self.config.block_resources:
            route.abort()
        else:
            route.continue_()
    
    async def _handle_route_async(self, route):
        """Handle route for resource blocking (async)."""
        if route.request.resource_type in self.config.block_resources:
            await route.abort()
        else:
            await route.continue_()
    
    def _fetch_with_sync_context(self, context: SyncBrowserContext, url: str) -> FetchResult:
        """Fetch data using sync context."""
        page = context.new_page()
        
        try:
            # Navigate to page
            response = page.goto(url, timeout=self.config.timeout, wait_until=self.config.wait_for_load_state)
            
            if not response or response.status >= 400:
                raise FetchError(f"Failed to load page: {response.status if response else 'No response'}")
            
            # Wait for additional selector if specified
            if self.config.wait_for_selector:
                page.wait_for_selector(self.config.wait_for_selector, timeout=self.config.timeout)
            
            # Extract data based on expected format
            data = self._extract_data_sync(page, url)
            
            # Validate data
            if not self.validate_data(data):
                raise ValidationError("Data validation failed")
            
            # Capture metadata
            metadata = self._capture_metadata_sync(page, response)
            
            return FetchResult(
                url=url,
                data=data,
                timestamp=datetime.now(),
                format=self._spec.expected_format,
                method=FetchMethod.PLAYWRIGHT,
                metadata=metadata,
                cache_hit=False
            )
            
        finally:
            page.close()
    
    async def _fetch_with_async_context(self, context: BrowserContext, url: str) -> FetchResult:
        """Fetch data using async context."""
        page = await context.new_page()
        
        try:
            # Navigate to page
            response = await page.goto(url, timeout=self.config.timeout, wait_until=self.config.wait_for_load_state)
            
            if not response or response.status >= 400:
                raise FetchError(f"Failed to load page: {response.status if response else 'No response'}")
            
            # Wait for additional selector if specified
            if self.config.wait_for_selector:
                await page.wait_for_selector(self.config.wait_for_selector, timeout=self.config.timeout)
            
            # Extract data based on expected format
            data = await self._extract_data_async(page, url)
            
            # Validate data
            if not self.validate_data(data):
                raise ValidationError("Data validation failed")
            
            # Capture metadata
            metadata = await self._capture_metadata_async(page, response)
            
            return FetchResult(
                url=url,
                data=data,
                timestamp=datetime.now(),
                format=self._spec.expected_format,
                method=FetchMethod.PLAYWRIGHT,
                metadata=metadata,
                cache_hit=False
            )
            
        finally:
            await page.close()
    
    def _extract_data_sync(self, page: SyncPage, url: str) -> Any:
        """Extract data from page (sync)."""
        if self._spec.expected_format == DataFormat.JSON:
            return self._extract_json_sync(page)
        elif self._spec.expected_format == DataFormat.HTML:
            return self._extract_html_sync(page)
        else:
            return page.content()
    
    async def _extract_data_async(self, page: Page, url: str) -> Any:
        """Extract data from page (async)."""
        if self._spec.expected_format == DataFormat.JSON:
            return await self._extract_json_async(page)
        elif self._spec.expected_format == DataFormat.HTML:
            return await self._extract_html_async(page)
        else:
            return await page.content()
    
    def _extract_json_sync(self, page: SyncPage) -> Any:
        """Extract JSON data from page (sync)."""
        # Try to extract structured data
        structured_data = page.evaluate("""
            () => {
                // Extract JSON-LD data
                const jsonLdScripts = document.querySelectorAll('script[type="application/ld+json"]');
                const jsonLdData = [];
                
                for (let script of jsonLdScripts) {
                    try {
                        jsonLdData.push(JSON.parse(script.textContent));
                    } catch (e) {
                        // Skip invalid JSON
                    }
                }
                
                // Extract OpenGraph data
                const ogData = {};
                const ogMetas = document.querySelectorAll('meta[property^="og:"]');
                for (let meta of ogMetas) {
                    const property = meta.getAttribute('property').replace('og:', '');
                    ogData[property] = meta.getAttribute('content');
                }
                
                // Extract article data
                const articleData = {};
                const articleMetas = document.querySelectorAll('meta[property^="article:"]');
                for (let meta of articleMetas) {
                    const property = meta.getAttribute('property').replace('article:', '');
                    articleData[property] = meta.getAttribute('content');
                }
                
                return {
                    jsonLd: jsonLdData,
                    openGraph: ogData,
                    article: articleData,
                    url: window.location.href,
                    title: document.title
                };
            }
        """)
        
        # Check if this is a section/listing/search page (prioritize article lists over individual articles)
        if ('section/' in page.url or 'category/' in page.url or 'topic/' in page.url or
            '/search' in page.url or '?q=' in page.url or '?query=' in page.url or
            any(domain in page.url for domain in ['nytimes.com/section', 'bbc.com/news', 'reuters.com/', 'cnn.com/politics', 'cnn.com/world', 'cnn.com/us'])):
            return self._extract_article_links_sync(page)

        # If we have structured data for individual articles, return it
        if structured_data and (structured_data.get('jsonLd') or structured_data.get('openGraph')):
            return structured_data

        # Otherwise extract single article content
        return self._extract_article_content_sync(page)
    
    async def _extract_json_async(self, page: Page) -> Any:
        """Extract JSON data from page (async)."""
        # Try to extract structured data
        structured_data = await page.evaluate("""
            () => {
                // Extract JSON-LD data
                const jsonLdScripts = document.querySelectorAll('script[type="application/ld+json"]');
                const jsonLdData = [];
                
                for (let script of jsonLdScripts) {
                    try {
                        jsonLdData.push(JSON.parse(script.textContent));
                    } catch (e) {
                        // Skip invalid JSON
                    }
                }
                
                // Extract OpenGraph data
                const ogData = {};
                const ogMetas = document.querySelectorAll('meta[property^="og:"]');
                for (let meta of ogMetas) {
                    const property = meta.getAttribute('property').replace('og:', '');
                    ogData[property] = meta.getAttribute('content');
                }
                
                // Extract article data
                const articleData = {};
                const articleMetas = document.querySelectorAll('meta[property^="article:"]');
                for (let meta of articleMetas) {
                    const property = meta.getAttribute('property').replace('article:', '');
                    articleData[property] = meta.getAttribute('content');
                }
                
                return {
                    jsonLd: jsonLdData,
                    openGraph: ogData,
                    article: articleData,
                    url: window.location.href,
                    title: document.title
                };
            }
        """)
        
        # Check if this is a section/listing/search page (prioritize article lists over individual articles)
        if ('section/' in page.url or 'category/' in page.url or 'topic/' in page.url or
            '/search' in page.url or '?q=' in page.url or '?query=' in page.url or
            any(domain in page.url for domain in ['nytimes.com/section', 'bbc.com/news', 'reuters.com/', 'cnn.com/politics', 'cnn.com/world', 'cnn.com/us'])):
            return await self._extract_article_links_async(page)

        # If we have structured data for individual articles, return it
        if structured_data and (structured_data.get('jsonLd') or structured_data.get('openGraph')):
            return structured_data

        # Otherwise extract single article content
        return await self._extract_article_content_async(page)
    
    def _extract_html_sync(self, page: SyncPage) -> Dict[str, Any]:
        """Extract HTML data from page (sync)."""
        return self._extract_article_content_sync(page)
    
    async def _extract_html_async(self, page: Page) -> Dict[str, Any]:
        """Extract HTML data from page (async)."""
        return await self._extract_article_content_async(page)
    
    def _extract_article_links_sync(self, page: SyncPage) -> List[Dict[str, Any]]:
        """Extract article links from section/listing pages with intelligent scrolling."""
        self.logger.info("Starting article extraction with progressive scrolling...")

        # First, get initial articles count
        initial_count = page.evaluate("() => document.querySelectorAll('a[href*=\"/2024/\"], a[href*=\"/2025/\"]').length")
        self.logger.info(f"Initial articles found on page load: {initial_count}")

        return page.evaluate("""
            () => {
                const articles = [];
                const scrollMetrics = {
                    totalScrolls: 0,
                    articlesBeforeScroll: 0,
                    articlesAfterScroll: 0,
                    pageHeight: document.body.scrollHeight
                };

                // Get current page URL and search query for content filtering
                const currentUrl = window.location.href.toLowerCase();
                const isSearchPage = currentUrl.includes('/search') || currentUrl.includes('?q=') || currentUrl.includes('?query=');

                // Extract search query for content filtering
                let searchQuery = '';
                if (isSearchPage) {
                    const urlParams = new URLSearchParams(window.location.search);
                    searchQuery = (urlParams.get('q') || urlParams.get('query') || '').toLowerCase();
                }

                // Enhanced article validation function with content filtering
                function isValidArticleLink(href, text) {
                    if (!href || !text || text.length < 10) return false;

                    const url = href.toLowerCase();
                    const title = text.toLowerCase();

                    // Exclude navigation, social, and non-article links
                    const excludePatterns = [
                        '/search/', '/about/', '/contact/', '/privacy/', '/terms/',
                        '/subscribe/', '/newsletter/', '/login/', '/register/',
                        'facebook.com', 'twitter.com', 'instagram.com', 'youtube.com',
                        'mailto:', 'tel:', '#', 'javascript:'
                    ];

                    for (const pattern of excludePatterns) {
                        if (url.includes(pattern)) return false;
                    }

                    // Content filtering: if we have a search query, check if title is relevant
                    if (searchQuery && searchQuery.length > 2) {
                        const searchTerms = searchQuery.split(' ').filter(term => term.length > 2);
                        const titleRelevant = searchTerms.some(term => title.includes(term));
                        if (!titleRelevant) {
                            // Skip articles that don't mention the search terms
                            console.log(`Skipping irrelevant article: ${title}`);
                            return false;
                        }
                    }

                    // Include if it has date patterns (2024/2025)
                    if (url.includes('/2024/') || url.includes('/2025/')) return true;

                    // Include BBC news URLs
                    if (url.includes('bbc.com') && url.includes('/news/')) return true;

                    // Include CNN article patterns
                    if (url.includes('cnn.com') && (
                        url.includes('/2024/') || url.includes('/2025/') ||
                        url.includes('index.html') || url.includes('/politics/') ||
                        url.includes('/world/') || url.includes('/us/')
                    )) return true;

                    // Include Guardian articles
                    if (url.includes('guardian.com') && (
                        url.includes('/2024/') || url.includes('/2025/') ||
                        url.match(/\/\d{4}\/[a-z]{3}\/\d{2}\//) // Guardian date pattern
                    )) return true;

                    // Include NYT articles
                    if (url.includes('nytimes.com') && (
                        url.includes('/2024/') || url.includes('/2025/') ||
                        url.includes('html') // NYT often ends in .html
                    )) return true;

                    // Include Reuters articles
                    if (url.includes('reuters.com') && (
                        url.includes('/2024/') || url.includes('/2025/') ||
                        url.match(/\/\d{4}\/\d{2}\/\d{2}\//)
                    )) return true;

                    // Generic article patterns
                    if (url.includes('/article/') || url.includes('/story/') || url.includes('/post/')) return true;

                    return false;
                }

                // Enhanced multi-site selectors for article links (search result priority)
                const articleSelectors = [
                    // Search result specific patterns (priority for search pages)
                    '.search-result a', '.search-results a',
                    '.search-item a', '.search-entry a',
                    '.result a', '.results a',

                    // BBC search result patterns
                    '.ssrcss-1pie6c4-PromoContent a',  // BBC search results
                    '.ssrcss-atcud-PromoLink a',       // BBC promo links
                    '.gel-layout__item a',             // BBC layout items

                    // CNN search result patterns
                    '.cnn-search__result a',           // CNN search results
                    '.cnn-search__result-headline a', // CNN search headlines
                    '.zn-body__read-all a',           // CNN read all links

                    // Guardian search patterns
                    '.fc-item__content a',            // Guardian front content
                    '.fc-sublink a',                  // Guardian sublinks

                    // Generic article patterns
                    'article a',
                    'h1 a', 'h2 a', 'h3 a',
                    '[data-testid="headline"] a',

                    // BBC-specific patterns
                    '[data-component="headline"] a',
                    '.gs-c-promo-heading a',
                    '.media__link a',
                    '.gel-layout a',
                    'a[href*="/news/"]',

                    // CNN-specific patterns
                    '.cd__headline a',
                    '.container__headline a',
                    '.card-media a',
                    '.zn-body__paragraph a',
                    '.headline a',

                    // Guardian patterns
                    '[data-link-name*="article"] a',
                    '.fc-item__link a',
                    '.u-faux-block-link a',

                    // NYT patterns
                    '.story-wrapper a',
                    '.css-1l4spti a',

                    // Reuters patterns
                    '[class*="story"] a',
                    '[class*="headline"] a',

                    // Date-based and generic patterns
                    'a[href*="/2024/"]', 'a[href*="/2025/"]',
                    'a[href*="/article/"]',
                    'a[href*="/story/"]'
                ];

                function extractCurrentArticles() {
                    const foundLinks = new Set();
                    const currentArticles = [];

                    for (let selector of articleSelectors) {
                        const links = document.querySelectorAll(selector);

                        for (let link of links) {
                            const href = link.href;
                            const text = link.textContent?.trim();

                            // Enhanced filtering for actual article URLs
                            if (isValidArticleLink(href, text)) {
                                if (!foundLinks.has(href) && text.length > 10) {
                                    foundLinks.add(href);

                                    // Get additional info from parent elements
                                    let summary = '';
                                    const parent = link.closest('article, .story-wrapper, .css-1l4spti');
                                    if (parent) {
                                        const summaryEl = parent.querySelector('p, .summary, .css-1pga48a');
                                        if (summaryEl) {
                                            summary = summaryEl.textContent?.trim() || '';
                                        }
                                    }

                                    currentArticles.push({
                                        title: text,
                                        url: href,
                                        summary: summary.substring(0, 200)
                                    });
                                }
                            }
                        }
                    }
                    return currentArticles;
                }

                // Extract initial articles
                let currentArticles = extractCurrentArticles();
                scrollMetrics.articlesBeforeScroll = currentArticles.length;
                console.log(`📄 Initial extraction: ${currentArticles.length} articles found`);

                // Progressive scrolling to load more content
                let lastHeight = document.body.scrollHeight;
                let scrollAttempts = 0;
                const maxScrolls = 5; // Limit scrolling to prevent infinite loops

                while (scrollAttempts < maxScrolls && currentArticles.length < 50) {
                    // Scroll to bottom
                    window.scrollTo(0, document.body.scrollHeight);
                    scrollMetrics.totalScrolls++;

                    // Wait for potential new content to load (optimized for sync)
                    let waited = 0;
                    const checkInterval = 50; // Check every 50ms for faster detection
                    const maxWait = 600; // Reduced from 2000ms to 600ms

                    while (waited < maxWait) {
                        if (document.body.scrollHeight > lastHeight) {
                            console.log(`✅ Content loaded after ${waited}ms`);
                            break;
                        }
                        // Efficient wait - reduced interval for faster detection
                        const start = Date.now();
                        while (Date.now() - start < checkInterval) {}
                        waited += checkInterval;
                    }

                    if (waited >= maxWait) {
                        console.log(`⏱️  Timeout after ${maxWait}ms - no new content detected`);
                    }

                    lastHeight = document.body.scrollHeight;

                    // Extract articles again
                    const newArticles = extractCurrentArticles();
                    console.log(`🔄 Scroll ${scrollAttempts + 1}: Found ${newArticles.length} total articles (was ${currentArticles.length})`);

                    // If no new articles found, break
                    if (newArticles.length <= currentArticles.length) {
                        console.log(`⏹️  No new articles after scroll ${scrollAttempts + 1}, stopping`);
                        break;
                    }

                    currentArticles = newArticles;
                    scrollAttempts++;
                }

                scrollMetrics.articlesAfterScroll = currentArticles.length;
                scrollMetrics.finalPageHeight = document.body.scrollHeight;

                console.log(`📊 Scrolling Summary:`);
                console.log(`   • Total scrolls performed: ${scrollMetrics.totalScrolls}`);
                console.log(`   • Articles before scrolling: ${scrollMetrics.articlesBeforeScroll}`);
                console.log(`   • Articles after scrolling: ${scrollMetrics.articlesAfterScroll}`);
                console.log(`   • Additional articles found: ${scrollMetrics.articlesAfterScroll - scrollMetrics.articlesBeforeScroll}`);
                console.log(`   • Page height grew: ${scrollMetrics.finalPageHeight - scrollMetrics.pageHeight}px`);

                // Return articles with scroll metrics
                const result = currentArticles.slice(0, 20); // Return top 20 articles
                result._scrollMetrics = scrollMetrics;
                return result;
            }
        """)

    async def _extract_article_links_async(self, page: Page) -> List[Dict[str, Any]]:
        """Extract article links from section/listing pages with intelligent scrolling (async)."""
        self.logger.info("Starting async article extraction with progressive scrolling...")

        # First, get initial articles count
        initial_count = await page.evaluate("() => document.querySelectorAll('a[href*=\"/2024/\"], a[href*=\"/2025/\"]').length")
        self.logger.info(f"Initial articles found on page load: {initial_count}")

        # Use the same scrolling logic as sync version
        return await page.evaluate("""
            async () => {
                const articles = [];
                const scrollMetrics = {
                    totalScrolls: 0,
                    articlesBeforeScroll: 0,
                    articlesAfterScroll: 0,
                    pageHeight: document.body.scrollHeight
                };

                // Get current page URL and search query for content filtering
                const currentUrl = window.location.href.toLowerCase();
                const isSearchPage = currentUrl.includes('/search') || currentUrl.includes('?q=') || currentUrl.includes('?query=');

                // Extract search query for content filtering
                let searchQuery = '';
                if (isSearchPage) {
                    const urlParams = new URLSearchParams(window.location.search);
                    searchQuery = (urlParams.get('q') || urlParams.get('query') || '').toLowerCase();
                }

                // Enhanced article validation function with content filtering
                function isValidArticleLink(href, text) {
                    if (!href || !text || text.length < 10) return false;

                    const url = href.toLowerCase();
                    const title = text.toLowerCase();

                    // Exclude navigation, social, and non-article links
                    const excludePatterns = [
                        '/search/', '/about/', '/contact/', '/privacy/', '/terms/',
                        '/subscribe/', '/newsletter/', '/login/', '/register/',
                        'facebook.com', 'twitter.com', 'instagram.com', 'youtube.com',
                        'mailto:', 'tel:', '#', 'javascript:'
                    ];

                    for (const pattern of excludePatterns) {
                        if (url.includes(pattern)) return false;
                    }

                    // Content filtering: if we have a search query, check if title is relevant
                    if (searchQuery && searchQuery.length > 2) {
                        const searchTerms = searchQuery.split(' ').filter(term => term.length > 2);
                        const titleRelevant = searchTerms.some(term => title.includes(term));
                        if (!titleRelevant) {
                            // Skip articles that don't mention the search terms
                            console.log(`Skipping irrelevant article: ${title}`);
                            return false;
                        }
                    }

                    // Include if it has date patterns (2024/2025)
                    if (url.includes('/2024/') || url.includes('/2025/')) return true;

                    // Include BBC news URLs
                    if (url.includes('bbc.com') && url.includes('/news/')) return true;

                    // Include CNN article patterns
                    if (url.includes('cnn.com') && (
                        url.includes('/2024/') || url.includes('/2025/') ||
                        url.includes('index.html') || url.includes('/politics/') ||
                        url.includes('/world/') || url.includes('/us/')
                    )) return true;

                    // Include Guardian articles
                    if (url.includes('guardian.com') && (
                        url.includes('/2024/') || url.includes('/2025/') ||
                        url.match(/\/\d{4}\/[a-z]{3}\/\d{2}\//) // Guardian date pattern
                    )) return true;

                    // Include NYT articles
                    if (url.includes('nytimes.com') && (
                        url.includes('/2024/') || url.includes('/2025/') ||
                        url.includes('html') // NYT often ends in .html
                    )) return true;

                    // Include Reuters articles
                    if (url.includes('reuters.com') && (
                        url.includes('/2024/') || url.includes('/2025/') ||
                        url.match(/\/\d{4}\/\d{2}\/\d{2}\//)
                    )) return true;

                    // Generic article patterns
                    if (url.includes('/article/') || url.includes('/story/') || url.includes('/post/')) return true;

                    return false;
                }

                // Enhanced multi-site selectors for article links (search result priority)
                const articleSelectors = [
                    // Search result specific patterns (priority for search pages)
                    '.search-result a', '.search-results a',
                    '.search-item a', '.search-entry a',
                    '.result a', '.results a',

                    // BBC search result patterns
                    '.ssrcss-1pie6c4-PromoContent a',  // BBC search results
                    '.ssrcss-atcud-PromoLink a',       // BBC promo links
                    '.gel-layout__item a',             // BBC layout items

                    // CNN search result patterns
                    '.cnn-search__result a',           // CNN search results
                    '.cnn-search__result-headline a', // CNN search headlines
                    '.zn-body__read-all a',           // CNN read all links

                    // Guardian search patterns
                    '.fc-item__content a',            // Guardian front content
                    '.fc-sublink a',                  // Guardian sublinks

                    // Generic article patterns
                    'article a',
                    'h1 a', 'h2 a', 'h3 a',
                    '[data-testid="headline"] a',

                    // BBC-specific patterns
                    '[data-component="headline"] a',
                    '.gs-c-promo-heading a',
                    '.media__link a',
                    '.gel-layout a',
                    'a[href*="/news/"]',

                    // CNN-specific patterns
                    '.cd__headline a',
                    '.container__headline a',
                    '.card-media a',
                    '.zn-body__paragraph a',
                    '.headline a',

                    // Guardian patterns
                    '[data-link-name*="article"] a',
                    '.fc-item__link a',
                    '.u-faux-block-link a',

                    // NYT patterns
                    '.story-wrapper a',
                    '.css-1l4spti a',

                    // Reuters patterns
                    '[class*="story"] a',
                    '[class*="headline"] a',

                    // Date-based and generic patterns
                    'a[href*="/2024/"]', 'a[href*="/2025/"]',
                    'a[href*="/article/"]',
                    'a[href*="/story/"]'
                ];

                function extractCurrentArticles() {
                    const foundLinks = new Set();
                    const currentArticles = [];

                    for (let selector of articleSelectors) {
                        const links = document.querySelectorAll(selector);

                        for (let link of links) {
                            const href = link.href;
                            const text = link.textContent?.trim();

                            // Enhanced filtering for actual article URLs
                            if (isValidArticleLink(href, text)) {
                                if (!foundLinks.has(href) && text.length > 10) {
                                    foundLinks.add(href);

                                    // Get additional info from parent elements
                                    let summary = '';
                                    const parent = link.closest('article, .story-wrapper, .css-1l4spti');
                                    if (parent) {
                                        const summaryEl = parent.querySelector('p, .summary, .css-1pga48a');
                                        if (summaryEl) {
                                            summary = summaryEl.textContent?.trim() || '';
                                        }
                                    }

                                    currentArticles.push({
                                        title: text,
                                        url: href,
                                        summary: summary.substring(0, 200)
                                    });
                                }
                            }
                        }
                    }
                    return currentArticles;
                }

                // Extract initial articles
                let currentArticles = extractCurrentArticles();
                scrollMetrics.articlesBeforeScroll = currentArticles.length;
                console.log(`📄 Initial extraction: ${currentArticles.length} articles found`);

                // Progressive scrolling to load more content
                let lastHeight = document.body.scrollHeight;
                let scrollAttempts = 0;
                const maxScrolls = 5; // Limit scrolling to prevent infinite loops

                while (scrollAttempts < maxScrolls && currentArticles.length < 50) {
                    // Scroll to bottom
                    window.scrollTo(0, document.body.scrollHeight);
                    scrollMetrics.totalScrolls++;

                    // Wait for potential new content to load (optimized async version)
                    await new Promise(resolve => {
                        let waited = 0;
                        const checkInterval = 50; // Check every 50ms for faster detection
                        const maxWait = 600; // Reduced from 2000ms to 600ms

                        const interval = setInterval(() => {
                            waited += checkInterval;
                            if (document.body.scrollHeight > lastHeight) {
                                console.log(`✅ Content loaded after ${waited}ms`);
                                clearInterval(interval);
                                resolve();
                            } else if (waited >= maxWait) {
                                console.log(`⏱️  Timeout after ${maxWait}ms - no new content detected`);
                                clearInterval(interval);
                                resolve();
                            }
                        }, checkInterval);
                    });

                    lastHeight = document.body.scrollHeight;

                    // Extract articles again
                    const newArticles = extractCurrentArticles();
                    console.log(`🔄 Scroll ${scrollAttempts + 1}: Found ${newArticles.length} total articles (was ${currentArticles.length})`);

                    // If no new articles found, break
                    if (newArticles.length <= currentArticles.length) {
                        console.log(`⏹️  No new articles after scroll ${scrollAttempts + 1}, stopping`);
                        break;
                    }

                    currentArticles = newArticles;
                    scrollAttempts++;
                }

                scrollMetrics.articlesAfterScroll = currentArticles.length;
                scrollMetrics.finalPageHeight = document.body.scrollHeight;

                console.log(`📊 Scrolling Summary:`);
                console.log(`   • Total scrolls performed: ${scrollMetrics.totalScrolls}`);
                console.log(`   • Articles before scrolling: ${scrollMetrics.articlesBeforeScroll}`);
                console.log(`   • Articles after scrolling: ${scrollMetrics.articlesAfterScroll}`);
                console.log(`   • Additional articles found: ${scrollMetrics.articlesAfterScroll - scrollMetrics.articlesBeforeScroll}`);
                console.log(`   • Page height grew: ${scrollMetrics.finalPageHeight - scrollMetrics.pageHeight}px`);

                // Return articles with scroll metrics
                const result = currentArticles.slice(0, 20); // Return top 20 articles
                result._scrollMetrics = scrollMetrics;
                return result;
            }
        """)

    def _extract_article_content_sync(self, page: SyncPage) -> Dict[str, Any]:
        """Extract article content using common patterns (sync)."""
        return page.evaluate("""
            () => {
                // Common selectors for news articles
                const titleSelectors = [
                    'h1',
                    '.headline',
                    '.title',
                    '[data-testid="headline"]',
                    'h1.entry-title',
                    '.post-title'
                ];
                
                const contentSelectors = [
                    '.article-body',
                    '.content',
                    '.post-content',
                    '[data-testid="article-body"]',
                    '.entry-content',
                    'main p',
                    '.story-body'
                ];
                
                const authorSelectors = [
                    '.author',
                    '.byline',
                    '[data-testid="author"]',
                    '.post-author',
                    '.entry-author'
                ];
                
                const dateSelectors = [
                    'time',
                    '.date',
                    '.published',
                    '[data-testid="date"]',
                    '.entry-date'
                ];
                
                const data = {
                    url: window.location.href,
                    timestamp: new Date().toISOString()
                };
                
                // Extract title
                for (let selector of titleSelectors) {
                    const elem = document.querySelector(selector);
                    if (elem && elem.textContent.trim()) {
                        data.title = elem.textContent.trim();
                        break;
                    }
                }
                
                // Extract content
                for (let selector of contentSelectors) {
                    const elems = document.querySelectorAll(selector);
                    if (elems.length > 0) {
                        data.content = Array.from(elems)
                            .map(el => el.textContent.trim())
                            .filter(text => text.length > 0)
                            .join('\\n');
                        break;
                    }
                }
                
                // Extract author
                for (let selector of authorSelectors) {
                    const elem = document.querySelector(selector);
                    if (elem && elem.textContent.trim()) {
                        data.author = elem.textContent.trim();
                        break;
                    }
                }
                
                // Extract date
                for (let selector of dateSelectors) {
                    const elem = document.querySelector(selector);
                    if (elem) {
                        data.published_date = elem.getAttribute('datetime') || 
                                             elem.getAttribute('content') || 
                                             elem.textContent.trim();
                        break;
                    }
                }
                
                // Extract images
                const images = Array.from(document.querySelectorAll('img'))
                    .filter(img => img.src && !img.src.includes('data:'))
                    .map(img => ({
                        url: img.src,
                        alt: img.alt || '',
                        caption: img.getAttribute('title') || ''
                    }))
                    .slice(0, 10); // Limit to first 10 images
                
                if (images.length > 0) {
                    data.images = images;
                }
                
                return data;
            }
        """)
    
    async def _extract_article_content_async(self, page: Page) -> Dict[str, Any]:
        """Extract article content using common patterns (async)."""
        return await page.evaluate("""
            () => {
                // Common selectors for news articles
                const titleSelectors = [
                    'h1',
                    '.headline',
                    '.title',
                    '[data-testid="headline"]',
                    'h1.entry-title',
                    '.post-title'
                ];
                
                const contentSelectors = [
                    '.article-body',
                    '.content',
                    '.post-content',
                    '[data-testid="article-body"]',
                    '.entry-content',
                    'main p',
                    '.story-body'
                ];
                
                const authorSelectors = [
                    '.author',
                    '.byline',
                    '[data-testid="author"]',
                    '.post-author',
                    '.entry-author'
                ];
                
                const dateSelectors = [
                    'time',
                    '.date',
                    '.published',
                    '[data-testid="date"]',
                    '.entry-date'
                ];
                
                const data = {
                    url: window.location.href,
                    timestamp: new Date().toISOString()
                };
                
                // Extract title
                for (let selector of titleSelectors) {
                    const elem = document.querySelector(selector);
                    if (elem && elem.textContent.trim()) {
                        data.title = elem.textContent.trim();
                        break;
                    }
                }
                
                // Extract content
                for (let selector of contentSelectors) {
                    const elems = document.querySelectorAll(selector);
                    if (elems.length > 0) {
                        data.content = Array.from(elems)
                            .map(el => el.textContent.trim())
                            .filter(text => text.length > 0)
                            .join('\\n');
                        break;
                    }
                }
                
                // Extract author
                for (let selector of authorSelectors) {
                    const elem = document.querySelector(selector);
                    if (elem && elem.textContent.trim()) {
                        data.author = elem.textContent.trim();
                        break;
                    }
                }
                
                // Extract date
                for (let selector of dateSelectors) {
                    const elem = document.querySelector(selector);
                    if (elem) {
                        data.published_date = elem.getAttribute('datetime') || 
                                             elem.getAttribute('content') || 
                                             elem.textContent.trim();
                        break;
                    }
                }
                
                // Extract images
                const images = Array.from(document.querySelectorAll('img'))
                    .filter(img => img.src && !img.src.includes('data:'))
                    .map(img => ({
                        url: img.src,
                        alt: img.alt || '',
                        caption: img.getAttribute('title') || ''
                    }))
                    .slice(0, 10); // Limit to first 10 images
                
                if (images.length > 0) {
                    data.images = images;
                }
                
                return data;
            }
        """)
    
    def _capture_metadata_sync(self, page: SyncPage, response) -> Dict[str, Any]:
        """Capture page metadata (sync)."""
        metadata = {
            'status_code': response.status,
            'final_url': page.url,
            'title': page.title(),
            'viewport': page.viewport_size,
            'user_agent': self.config.user_agent,
            'load_state': 'complete'
        }
        
        # Capture performance metrics if available
        try:
            performance = page.evaluate("() => JSON.stringify(performance.getEntriesByType('navigation')[0])")
            if performance:
                metadata['performance'] = json.loads(performance)
        except Exception:
            pass
        
        return metadata
    
    async def _capture_metadata_async(self, page: Page, response) -> Dict[str, Any]:
        """Capture page metadata (async)."""
        metadata = {
            'status_code': response.status,
            'final_url': page.url,
            'title': await page.title(),
            'viewport': page.viewport_size,
            'user_agent': self.config.user_agent,
            'load_state': 'complete'
        }
        
        # Capture performance metrics if available
        try:
            performance = await page.evaluate("() => JSON.stringify(performance.getEntriesByType('navigation')[0])")
            if performance:
                metadata['performance'] = json.loads(performance)
        except Exception:
            pass
        
        return metadata
    
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
            
            if isinstance(data, dict):
                if not data:
                    return False
                # For articles, check if we have title or content
                if 'title' in data or 'content' in data:
                    return True
            
            if isinstance(data, list) and not data:
                return False
            
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
    
    async def capture_har(self, url: str) -> Dict[str, Any]:
        """Capture HAR file for debugging and analysis."""
        async with async_playwright() as playwright:
            browser = await self._launch_async_browser(playwright)
            
            # Create context with HAR recording
            har_path = f"/tmp/datafetch_har_{self._session_id}_{int(time.time())}.har"
            context = await browser.new_context(record_har_path=har_path)
            
            try:
                page = await context.new_page()
                await page.goto(url, timeout=self.config.timeout)
                await page.wait_for_load_state(self.config.wait_for_load_state)
                
                # Read HAR file
                with open(har_path, 'r') as f:
                    har_data = json.load(f)
                
                return har_data
                
            finally:
                await context.close()
                await browser.close()
                # Clean up HAR file
                try:
                    Path(har_path).unlink()
                except Exception:
                    pass
    
    async def capture_screenshot(self, url: str, full_page: bool = True) -> bytes:
        """Capture screenshot of the page."""
        async with async_playwright() as playwright:
            browser = await self._launch_async_browser(playwright)
            context = await self._create_async_context(browser)
            
            try:
                page = await context.new_page()
                await page.goto(url, timeout=self.config.timeout)
                await page.wait_for_load_state(self.config.wait_for_load_state)
                
                screenshot = await page.screenshot(full_page=full_page, type='png')
                return screenshot
                
            finally:
                await context.close()
                await browser.close()