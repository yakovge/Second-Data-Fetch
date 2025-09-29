import asyncio
import json
import logging
import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
import time

import anthropic

try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file if available
except ImportError:
    pass  # dotenv not available, continue without it

from ..core.datafetch import FetchSpec, DataFormat, FetchMethod, SecurityError


@dataclass
class ClaudeConfig:
    model_name: str
    api_key: str
    max_tokens: int
    timeout: int


class ClaudeClient:
    """
    Claude 3 Haiku AI integration for structure generation and implementation.

    Handles:
    - Data structure inference from sample data
    - DataFetch implementation generation
    - URL discovery from text descriptions
    - Input sanitization and prompt injection prevention
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "claude-3-haiku-20240307"):
        """
        Initialize Claude client.

        Args:
            api_key: Anthropic API key (or from environment)
            model_name: Model to use (default: claude-3-haiku-20240307)
        """
        self.config = self._create_config(api_key, model_name)
        self.logger = logging.getLogger("datafetch.ai.claude")
        self.logger.setLevel(logging.INFO)

        # Configure Claude
        self.client = anthropic.Anthropic(api_key=self.config.api_key)

        self._prompt_cache = {}

    def _create_config(self, api_key: Optional[str], model_name: str) -> ClaudeConfig:
        """Create AI configuration."""
        if not api_key:
            api_key = os.getenv('CLAUDE_API_KEY') or os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("Claude API key must be provided or set in CLAUDE_API_KEY or ANTHROPIC_API_KEY environment variable")

        return ClaudeConfig(
            model_name=model_name,
            api_key=api_key,
            max_tokens=4096,  # Claude Haiku maximum
            timeout=60
        )

    def generate_structure_from_sample(self, sample_data: Any, context: str = "") -> Dict[str, Any]:
        """
        Generate data structure definition from sample data.

        Args:
            sample_data: Sample of fetched data
            context: Additional context about the data source

        Returns:
            Dictionary containing structure definition
        """
        try:
            # Sanitize inputs
            sanitized_context = self._sanitize_input(context)

            # Prepare sample data (limit size to prevent token overflow)
            sample_str = self._prepare_sample_data(sample_data)

            # Create prompt
            prompt = self._create_structure_generation_prompt(sample_str, sanitized_context)

            # Generate response
            response = self._generate_with_retry(prompt)

            # Parse and validate response
            structure = self._parse_structure_response(response)

            self.logger.info("Successfully generated structure from sample data")
            return structure

        except Exception as e:
            self.logger.error(f"Structure generation failed: {str(e)}")
            # Return basic structure as fallback
            return self._infer_basic_structure(sample_data)

    def generate_urls_from_text(self, raw_text: str, domain_hints: Optional[List[str]] = None) -> List[str]:
        """
        Generate URLs from text description.

        Args:
            raw_text: User's description of data requirements
            domain_hints: Optional list of domain hints

        Returns:
            List of generated URLs
        """
        try:
            # Sanitize input
            sanitized_text = self._sanitize_input(raw_text)

            # Create prompt
            prompt = self._create_url_generation_prompt(sanitized_text, domain_hints)

            # Generate response
            response = self._generate_with_retry(prompt)

            # Parse URLs from response
            urls = self._parse_url_response(response)

            # Validate URLs
            validated_urls = self._validate_generated_urls(urls)

            self.logger.info(f"Generated {len(validated_urls)} URLs from text")
            return validated_urls

        except Exception as e:
            self.logger.error(f"URL generation failed: {str(e)}")
            return []

    def generate_datafetch_implementation(self,
                                       spec: FetchSpec,
                                       sample_data: Optional[Any] = None,
                                       base_class_path: str = "src.core.datafetch.DataFetch") -> str:
        """
        Generate DataFetch implementation code.

        Args:
            spec: FetchSpec containing requirements
            sample_data: Optional sample data for reference
            base_class_path: Path to base DataFetch class

        Returns:
            Generated Python code as string
        """
        try:
            # Sanitize inputs
            sanitized_raw_text = self._sanitize_input(spec.raw_text)

            # Prepare sample data if provided
            sample_str = ""
            if sample_data:
                sample_str = self._prepare_sample_data(sample_data)

            # Create prompt
            prompt = self._create_implementation_prompt(
                sanitized_raw_text,
                spec.urls,
                spec.expected_format,
                spec.method,
                sample_str,
                base_class_path
            )

            # Generate response
            response = self._generate_with_retry(prompt, max_tokens=4096)

            # Extract and validate code
            code = self._extract_code_from_response(response)

            # Validate generated code for security
            self._validate_generated_code(code)

            self.logger.info("Successfully generated DataFetch implementation")
            return code

        except Exception as e:
            self.logger.error(f"Implementation generation failed: {str(e)}")
            raise

    def _sanitize_input(self, text: str) -> str:
        """
        Sanitize input to prevent prompt injection attacks.

        Args:
            text: Input text to sanitize

        Returns:
            Sanitized text

        Raises:
            SecurityError: If dangerous content is detected
        """
        if not text:
            return ""

        # Convert to string if not already
        text = str(text)

        # Check for prompt injection patterns
        dangerous_patterns = [
            r'ignore\s+previous\s+instructions',
            r'ignore\s+above',
            r'ignore\s+all\s+instructions',
            r'new\s+instructions:',
            r'system\s+prompt:',
            r'assistant\s+prompt:',
            r'</?\s*system\s*>',
            r'</?\s*user\s*>',
            r'</?\s*assistant\s*>',
            r'\\x[0-9a-fA-F]{2}',  # Hex escape sequences
            r'\\u[0-9a-fA-F]{4}',  # Unicode escape sequences
        ]

        import re
        for pattern in dangerous_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                raise SecurityError(f"Potential prompt injection detected: {pattern}")

        # Remove or escape potentially dangerous characters
        # Keep basic punctuation and formatting
        allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:()\'"[]{}/-_@#$%^&*+=|\\`~\n\r\t')
        sanitized = ''.join(char for char in text if char in allowed_chars)

        # Limit length to prevent token overflow
        max_length = 10000
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length] + "... [truncated]"

        return sanitized.strip()

    def _prepare_sample_data(self, sample_data: Any) -> str:
        """Prepare sample data for inclusion in prompts."""
        try:
            if isinstance(sample_data, (dict, list)):
                sample_str = json.dumps(sample_data, indent=2, ensure_ascii=False)
            else:
                sample_str = str(sample_data)

            # Limit size to prevent token overflow
            max_length = 5000
            if len(sample_str) > max_length:
                sample_str = sample_str[:max_length] + "\n... [truncated]"

            return sample_str
        except Exception:
            return str(sample_data)[:1000] + "... [error serializing]"

    def _create_structure_generation_prompt(self, sample_data: str, context: str) -> str:
        """Create prompt for structure generation."""
        return f"""You are a data structure analyst. Analyze the provided sample data and generate a JSON schema that describes its structure.

Context: {context}

Sample Data:
```
{sample_data}
```

Requirements:
1. Generate a valid JSON schema (draft-07)
2. Identify required vs optional fields
3. Use appropriate data types (string, number, boolean, array, object)
4. Include format specifications where applicable (date-time, uri, email)
5. Provide descriptions for fields when the purpose is clear
6. Handle nested objects and arrays appropriately
7. Focus on news article structures if applicable

Return only the JSON schema, no additional commentary:"""

    def _create_url_generation_prompt(self, raw_text: str, domain_hints: Optional[List[str]]) -> str:
        """Create prompt for URL generation with dynamic website-specific guidance."""

        # Generate dynamic content based on target domains
        if domain_hints:
            target_guidance = self._generate_dynamic_url_guidance(domain_hints, raw_text)
            domain_section = f"\nTarget domains: {', '.join(domain_hints)}"
        else:
            target_guidance = self._generate_multi_site_url_guidance(raw_text)
            domain_section = "\nNo specific sites mentioned - generate from diverse sources"

        return f"""You are an adaptive web data specialist. Generate topic-specific URLs for ANY news website.

User Requirements: {raw_text}{domain_section}

{target_guidance}

CRITICAL REQUIREMENTS:
1. Generate section/category URLs that list articles, NEVER search pages (/search?q=)
2. Focus on the specific websites mentioned or provided in domain hints
3. Use website-appropriate URL patterns for each domain
4. Generate 2-3 relevant URLs per target website
5. Ensure URLs lead to article listings, not individual articles

ADAPTIVE APPROACH:
- If specific website mentioned: Focus on that site's URL patterns
- If multiple sites: Use each site's optimal URL structure
- If unknown site: Use common news site patterns
- Always prioritize section/category pages over search results

Return only URLs, one per line, no additional text or explanations:"""

    def _generate_dynamic_url_guidance(self, domain_hints: List[str], raw_text: str) -> str:
        """Generate dynamic URL guidance based on specific target domains."""
        guidance_parts = []

        for domain in domain_hints:
            if 'nytimes.com' in domain:
                guidance_parts.append("""
NYT URL PATTERNS:
- Politics: https://www.nytimes.com/section/politics
- Business: https://www.nytimes.com/section/business
- Technology: https://www.nytimes.com/section/technology
- World: https://www.nytimes.com/section/world
- US: https://www.nytimes.com/section/us
Format: /section/[topic-name]""")

            elif 'bbc.com' in domain or 'bbc.co.uk' in domain:
                guidance_parts.append("""
BBC URL PATTERNS:
- World: https://www.bbc.com/news/world
- Politics: https://www.bbc.com/news/politics
- Business: https://www.bbc.com/news/business
- Technology: https://www.bbc.com/news/technology
- Science: https://www.bbc.com/news/science-environment
Format: /news/[topic-name]""")

            elif 'reuters.com' in domain:
                guidance_parts.append("""
REUTERS URL PATTERNS:
- Technology: https://www.reuters.com/technology/
- Business: https://www.reuters.com/business/
- World: https://www.reuters.com/world/
- Markets: https://www.reuters.com/markets/
- Politics: https://www.reuters.com/world/us/
Format: /[topic]/ or /world/[region]/
NOTE: Reuters may have access restrictions (401 errors)""")

            elif 'cnn.com' in domain:
                guidance_parts.append("""
CNN URL PATTERNS:
- Politics: https://www.cnn.com/politics
- Business: https://www.cnn.com/business
- World: https://www.cnn.com/world
- Technology: https://www.cnn.com/business/tech
- US: https://www.cnn.com/us
Format: /[topic-name]""")

            elif 'guardian.com' in domain or 'theguardian.com' in domain:
                guidance_parts.append("""
GUARDIAN URL PATTERNS:
- World: https://www.theguardian.com/world
- Politics: https://www.theguardian.com/politics
- Business: https://www.theguardian.com/business
- Technology: https://www.theguardian.com/technology
- Environment: https://www.theguardian.com/environment
Format: /[topic-name]""")

            else:
                guidance_parts.append(f"""
UNKNOWN DOMAIN ({domain}) PATTERNS:
- Try: {domain}/news/[topic]
- Try: {domain}/[topic]/
- Try: {domain}/section/[topic]
- Try: {domain}/category/[topic]
Use common news site URL structures""")

        return "\n".join(guidance_parts)

    def _generate_multi_site_url_guidance(self, raw_text: str) -> str:
        """Generate guidance for multi-site URL generation when no specific sites mentioned."""
        return """
MULTI-SITE DIVERSE URL GENERATION:
Generate URLs from 3 different major news sources for maximum diversity.

ROTATION STRATEGY - vary the order to avoid bias:
Set A: BBC → CNN → NYT
Set B: Reuters → Guardian → NYT
Set C: CNN → BBC → Guardian

WEBSITE-SPECIFIC PATTERNS:
- BBC: /news/[topic] (world, politics, business, technology)
- CNN: /[topic] (politics, business, world, tech)
- NYT: /section/[topic] (politics, business, technology, world, us)
- Reuters: /[topic]/ (technology, business, world)
- Guardian: /[topic] (world, politics, business, technology)

TOPIC MAPPING:
- Politics → politics, world/us sections
- Technology → technology, business/tech sections
- Business → business, markets sections
- World → world, international sections
- Climate → science-environment, world sections

Choose 3 different sites and generate 1 relevant URL per site."""

    def _create_implementation_prompt(self,
                                   raw_text: str,
                                   urls: List[str],
                                   expected_format: DataFormat,
                                   method: FetchMethod,
                                   sample_data: str,
                                   base_class_path: str) -> str:
        """Create prompt for DataFetch implementation generation."""

        # Detect target websites for dynamic adaptation
        website_types = set()
        for url in urls:
            if 'nytimes.com' in url:
                website_types.add('nyt')
            elif 'bbc.com' in url or 'bbc.co.uk' in url:
                website_types.add('bbc')
            elif 'reuters.com' in url:
                website_types.add('reuters')
            elif 'cnn.com' in url:
                website_types.add('cnn')
            elif 'guardian.com' in url or 'theguardian.com' in url:
                website_types.add('guardian')
            else:
                website_types.add('unknown')

        # Create website-specific guidance
        website_guidance = self._generate_website_specific_guidance(website_types)

        sample_section = ""
        if sample_data:
            sample_section = f"""

Sample Data Reference:
```
{sample_data}
```"""

        return f"""You are an adaptive Python code generator that creates DataFetch implementations for ANY news website.

Target websites detected: {', '.join(website_types)}
Requirements: {raw_text}
URLs: {', '.join(urls)}
Expected Format: {expected_format.value}
Fetch Method: {method.value}
Base Class: {base_class_path}{sample_section}

{website_guidance}

CRITICAL REQUIREMENTS:
1. Inherit from DataFetch and implement ALL required methods
2. Use HTTPClient or BrowserClient based on method parameter
3. Process ALL URLs in the spec, not just the first one
4. Handle website failures gracefully - continue with other URLs
5. Combine results from all successful URLs into unified list
6. Use website-specific extraction logic for each domain

TEMPLATE (use this exact structure):
```python
from datetime import datetime
from typing import Dict, List, Optional, Any, Generator, AsyncGenerator
import time
import asyncio

from src.core.datafetch import DataFetch, FetchResult, DataFormat, FetchMethod, FetchError, ValidationError
from src.collectors.http_client import HTTPClient
from src.collectors.browser_client import BrowserClient

class AdaptiveNewsFetch(DataFetch):
    def __init__(self, spec, cache_client=None, storage_client=None, ai_client=None):
        super().__init__(spec, cache_client, storage_client, ai_client)

        # Set up logging
        import logging
        self.logger = logging.getLogger(f"datafetch.adaptive.{{self._session_id[:8]}}")
        self.logger.setLevel(logging.INFO)

        # Choose client based on method
        if spec.method == FetchMethod.PLAYWRIGHT:
            self.client = BrowserClient(spec, cache_client, storage_client, ai_client)
        else:
            self.client = HTTPClient(spec, cache_client, storage_client, ai_client)

    def fetch(self) -> FetchResult:
        \"\"\"Fetch from ALL URLs and combine results.\"\"\"
        start_time = time.time()

        # Use the client's fetch_all method if available for proper multi-URL handling
        if hasattr(self.client, 'fetch_all'):
            result = self.client.fetch_all()
        else:
            result = self.client.fetch()

        # Apply website-specific processing to the combined results
        if result.data and not result.error:
            processed_data = []

            # Process each item with website-specific logic
            if isinstance(result.data, list):
                for item in result.data:
                    # Try to determine source website from item data
                    source_url = item.get('source_url', '') if isinstance(item, dict) else ''
                    website_type = self._detect_website_type(source_url) if source_url else 'unknown'

                    standardized_item = self._standardize_article(item, source_url)
                    processed_data.append(standardized_item)
            else:
                # Single item result
                source_url = result.url if hasattr(result, 'url') else ''
                website_type = self._detect_website_type(source_url)
                standardized_item = self._standardize_article(result.data, source_url)
                processed_data.append(standardized_item)

            # Update result with processed data
            result.data = processed_data
            result.execution_time = time.time() - start_time

        return result

    async def afetch(self) -> FetchResult:
        \"\"\"Async version of fetch.\"\"\"
        return await asyncio.get_event_loop().run_in_executor(None, self.fetch)

    def fetch_stream(self) -> Generator[FetchResult, None, None]:
        \"\"\"Stream individual articles.\"\"\"
        # Get all results first
        main_result = self.fetch()

        if main_result.data and isinstance(main_result.data, list):
            # Yield each article as individual result
            for article in main_result.data:
                yield FetchResult(
                    url=article.get('url', main_result.url) if isinstance(article, dict) else main_result.url,
                    data=article,
                    timestamp=datetime.now(),
                    format=self.spec.expected_format,
                    method=self.spec.method
                )
        elif main_result.data:
            # Single result
            yield main_result
        else:
            # No data or error
            yield main_result

    async def afetch_stream(self) -> AsyncGenerator[FetchResult, None]:
        \"\"\"Async stream version.\"\"\"
        for result in self.fetch_stream():
            yield result

    def _detect_website_type(self, url: str) -> str:
        \"\"\"Detect website type from URL.\"\"\"
        url_lower = url.lower()
        if 'nytimes.com' in url_lower:
            return 'nyt'
        elif 'bbc.com' in url_lower or 'bbc.co.uk' in url_lower:
            return 'bbc'
        elif 'reuters.com' in url_lower:
            return 'reuters'
        elif 'cnn.com' in url_lower:
            return 'cnn'
        elif 'guardian.com' in url_lower or 'theguardian.com' in url_lower:
            return 'guardian'
        else:
            return 'unknown'


    def _standardize_article(self, article: Any, source_url: str) -> Dict[str, Any]:
        \"\"\"Standardize article format across websites.\"\"\"
        if isinstance(article, dict):
            return {{
                'title': article.get('title', 'No title'),
                'summary': article.get('summary', article.get('content', '')[:200] + '...' if article.get('content') else 'No summary'),
                'url': article.get('url', source_url),
                'author': article.get('author', 'Unknown'),
                'published_date': article.get('published_date', article.get('timestamp', 'Unknown')),
                'source_url': source_url
            }}
        else:
            return {{
                'title': str(article)[:100] if article else 'No title',
                'summary': 'No summary available',
                'url': source_url,
                'author': 'Unknown',
                'published_date': 'Unknown',
                'source_url': source_url
            }}

    def validate_data(self, data: Any) -> bool:
        \"\"\"Validate fetched data.\"\"\"
        return data is not None and (isinstance(data, (list, dict)) and len(data) > 0 if data else False)

    def extract_structure(self, sample_data: Any) -> Dict[str, Any]:
        \"\"\"Extract structure from sample data.\"\"\"
        return {{'type': 'array', 'items': {{'type': 'object'}}}}
```

Generate the complete implementation with proper website-specific extraction logic."""

    def _generate_website_specific_guidance(self, website_types: set) -> str:
        """Generate dynamic website-specific guidance for AI prompts."""
        guidance_parts = []

        if 'nyt' in website_types:
            guidance_parts.append("""
NYT-SPECIFIC GUIDANCE:
- URL patterns: /section/[topic] (politics, business, technology, world)
- Key selectors: h1[data-testid="headline"], section[name="articleBody"], .css-* classes
- Article links: look for date-based URLs (/2024/, /2025/)
- Content structure: typically has structured data, rich metadata""")

        if 'bbc' in website_types:
            guidance_parts.append("""
BBC-SPECIFIC GUIDANCE:
- URL patterns: /news/[topic] (world, business, technology, politics)
- Key selectors: h1, [data-component="text-block"], .media-caption
- Article links: look for /news/ URLs with topic categories
- Content structure: clean HTML structure, good semantic markup""")

        if 'reuters' in website_types:
            guidance_parts.append("""
REUTERS-SPECIFIC GUIDANCE:
- URL patterns: /[topic]/ (technology, business, world)
- Key selectors: h1, [data-module="ArticleBody"], .article-body
- Article links: domain-relative URLs, date-based structure
- ACCESS ISSUES: Reuters may return 401 errors - handle gracefully
- Fallback: if 401 error, return empty results and continue with other sites""")

        if 'cnn' in website_types:
            guidance_parts.append("""
CNN-SPECIFIC GUIDANCE:
- URL patterns: /politics/, /business/, /world/ sections
- Key selectors: h1, .zn-body__paragraph, .headline
- Article links: full URLs with date structure
- Content structure: lots of dynamic content, may need browser rendering""")

        if 'guardian' in website_types:
            guidance_parts.append("""
GUARDIAN-SPECIFIC GUIDANCE:
- URL patterns: /[section]/[topic] (world, business, technology)
- Key selectors: [data-gu-name="headline"], .content__article-body
- Article links: guardian.com or theguardian.com domains
- Content structure: semantic HTML, good article metadata""")

        if 'unknown' in website_types:
            guidance_parts.append("""
UNKNOWN SITE GUIDANCE:
- Use generic selectors: h1, .headline, .title, [class*="title"]
- Content selectors: .content, .article-body, [class*="content"], [class*="body"]
- Try multiple selector strategies
- Graceful degradation if specific patterns fail""")

        # Add general multi-site guidance
        guidance_parts.append("""
MULTI-SITE PROCESSING:
- Handle each website type differently in the same implementation
- If one site fails (like Reuters 401), continue with others
- Combine results from all successful sites
- Standardize output format across all sites
- Use website detection to choose appropriate parsing strategy""")

        return "\n".join(guidance_parts)

    def _generate_with_retry(self, prompt: str, max_retries: int = 3, max_tokens: Optional[int] = None) -> str:
        """Generate response with retry logic."""
        tokens = max_tokens or self.config.max_tokens
        last_exception = None

        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    self.logger.info(f"Retry attempt {attempt + 1} for generation")
                    time.sleep(2 ** attempt)  # Exponential backoff

                response = self.client.messages.create(
                    model=self.config.model_name,
                    max_tokens=tokens,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )

                if response.content and len(response.content) > 0:
                    # Extract text from content blocks
                    text_content = ""
                    for content_block in response.content:
                        if hasattr(content_block, 'text'):
                            text_content += content_block.text

                    if text_content:
                        return text_content
                    else:
                        raise Exception("Empty response from model")
                else:
                    raise Exception("No content in response")

            except Exception as e:
                last_exception = e
                self.logger.warning(f"Generation attempt {attempt + 1} failed: {str(e)}")
                continue

        raise Exception(f"Generation failed after {max_retries} attempts: {str(last_exception)}")

    def _parse_structure_response(self, response: str) -> Dict[str, Any]:
        """Parse structure generation response."""
        try:
            # Try to extract JSON from response
            import re

            # Look for JSON block
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                structure = json.loads(json_str)

                # Validate it looks like a JSON schema
                if isinstance(structure, dict) and 'type' in structure:
                    return structure

            # If no valid JSON schema found, create basic one
            return {"type": "object", "properties": {}}

        except Exception as e:
            self.logger.warning(f"Failed to parse structure response: {str(e)}")
            return {"type": "object", "properties": {}}

    def _parse_url_response(self, response: str) -> List[str]:
        """Parse URL generation response."""
        import re

        # Extract URLs using regex
        url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
        urls = re.findall(url_pattern, response)

        # Clean up URLs
        cleaned_urls = []
        for url in urls:
            url = url.strip('.,!?;:()[]{}')
            if not url.startswith('http'):
                url = 'https://' + url
            cleaned_urls.append(url)

        return cleaned_urls

    def _validate_generated_urls(self, urls: List[str]) -> List[str]:
        """Validate generated URLs."""
        from urllib.parse import urlparse

        validated = []

        for url in urls:
            try:
                parsed = urlparse(url)

                # Basic validation
                if (parsed.scheme in ['http', 'https'] and
                    parsed.netloc and
                    '.' in parsed.netloc and
                    not any(blocked in parsed.netloc.lower() for blocked in
                           ['localhost', '127.0.0.1', '0.0.0.0', '::1'])):
                    validated.append(url)

            except Exception:
                continue

        return validated

    def _extract_code_from_response(self, response: str) -> str:
        """Extract Python code from response."""
        import re

        # Look for code block
        code_match = re.search(r'```python\n?(.*?)\n?```', response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        # Look for class definition
        class_match = re.search(r'(class\s+\w+.*?(?=\n\nclass|\n\n\w+|\Z))', response, re.DOTALL)
        if class_match:
            return class_match.group(1).strip()

        # Return entire response if no code block found
        return response.strip()

    def _validate_generated_code(self, code: str) -> None:
        """Validate generated code for security issues."""
        dangerous_patterns = [
            r'import\s+os\s*;.*os\.system',
            r'import\s+subprocess',
            r'eval\s*\(',
            r'exec\s*\(',
            r'__import__\s*\(',
            r'compile\s*\(',
            r'open\s*\([^)]*[\'"]w[\'"]',  # File writes
            r'pickle\.loads',
            r'marshal\.loads',
            r'shelve\.',
            r'socket\.',
            r'urllib\.request\.urlopen\s*\([^)]*file://',
        ]

        import re
        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                raise SecurityError(f"Potentially dangerous code pattern detected: {pattern}")

        # Check for suspicious string literals
        if re.search(r'[\'"][^\'"]*(rm\s+-rf|format\s+c:|del\s+/)[^\'"\s]*[\'"]', code, re.IGNORECASE):
            raise SecurityError("Potentially destructive command detected in code")

        # Basic Python syntax validation
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            raise SecurityError(f"Generated code has syntax errors: {str(e)}")

    def _infer_basic_structure(self, sample_data: Any) -> Dict[str, Any]:
        """Fallback basic structure inference."""
        if isinstance(sample_data, dict):
            properties = {}
            for key, value in sample_data.items():
                if isinstance(value, str):
                    properties[key] = {"type": "string"}
                elif isinstance(value, int):
                    properties[key] = {"type": "integer"}
                elif isinstance(value, float):
                    properties[key] = {"type": "number"}
                elif isinstance(value, bool):
                    properties[key] = {"type": "boolean"}
                elif isinstance(value, list):
                    properties[key] = {"type": "array", "items": {"type": "string"}}
                else:
                    properties[key] = {"type": "object"}

            return {
                "type": "object",
                "properties": properties
            }

        elif isinstance(sample_data, list):
            return {
                "type": "array",
                "items": {"type": "object"}
            }

        else:
            return {"type": "string"}

    async def agenerate_structure_from_sample(self, sample_data: Any, context: str = "") -> Dict[str, Any]:
        """Async version of structure generation."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.generate_structure_from_sample, sample_data, context
        )

    async def agenerate_urls_from_text(self, raw_text: str, domain_hints: Optional[List[str]] = None) -> List[str]:
        """Async version of URL generation."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.generate_urls_from_text, raw_text, domain_hints
        )

    async def agenerate_datafetch_implementation(self,
                                              spec: FetchSpec,
                                              sample_data: Optional[Any] = None,
                                              base_class_path: str = "src.core.datafetch.DataFetch") -> str:
        """Async version of implementation generation."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.generate_datafetch_implementation, spec, sample_data, base_class_path
        )

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics (placeholder for future implementation)."""
        return {
            "model_name": self.config.model_name,
            "requests_made": 0,  # Would track actual requests
            "tokens_used": 0,    # Would track actual token usage
            "cache_hits": len(self._prompt_cache)
        }