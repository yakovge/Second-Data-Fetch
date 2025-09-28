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
        """Create prompt for URL generation."""
        domain_section = ""
        if domain_hints:
            domain_section = f"\nPreferred domains: {', '.join(domain_hints)}"

        return f"""You are a web data specialist. Generate relevant, topic-specific URLs based on the user's data requirements.

User Requirements: {raw_text}{domain_section}

Instructions:
1. Generate exactly 2 relevant URLs that are SPECIFIC to the topic mentioned
2. Analyze the query for geographic regions, topics, or subjects
3. For different countries/regions, use different geographic sections:
   - Germany/Europe: world/europe, business (for German companies)
   - Russia/Eastern Europe: world/europe, world/asia (Russia spans both)
   - China/Asia: world/asia, business (Chinese economy)
   - Middle East: world/middleeast
   - US topics: us/politics, us (domestic)
   - Technology: technology, business
   - Climate: climate, science
4. Use section URLs like /section/[topic] (verified working sections only)
5. Prioritize geographic specificity over generic world section
6. Use HTTPS URLs only
7. Return only valid NYT section URLs that actually exist

Examples of TOPIC-SPECIFIC URL generation:
Query "articles about Germany" → https://www.nytimes.com/section/world/europe, https://www.nytimes.com/section/business
Query "articles about Russia" → https://www.nytimes.com/section/world/europe, https://www.nytimes.com/section/world/asia
Query "articles about China" → https://www.nytimes.com/section/world/asia, https://www.nytimes.com/section/business
Query "articles about technology" → https://www.nytimes.com/section/technology, https://www.nytimes.com/section/business
Query "articles about climate" → https://www.nytimes.com/section/climate, https://www.nytimes.com/section/science

IMPORTANT: Make URLs specific to the topic/geography mentioned, not generic world sections.

Return URLs one per line, no additional text:"""

    def _create_implementation_prompt(self,
                                   raw_text: str,
                                   urls: List[str],
                                   expected_format: DataFormat,
                                   method: FetchMethod,
                                   sample_data: str,
                                   base_class_path: str) -> str:
        """Create prompt for DataFetch implementation generation."""
        sample_section = ""
        if sample_data:
            sample_section = f"""

Sample Data Reference:
```
{sample_data}
```"""

        return f"""You are a Python code generator specializing in web scraping and data fetching. Generate a complete DataFetch implementation class.

Requirements: {raw_text}
URLs: {', '.join(urls)}
Expected Format: {expected_format.value}
Fetch Method: {method.value}
Base Class: {base_class_path}{sample_section}

Instructions:
1. Create a class that inherits from DataFetch
2. Implement all required abstract methods
3. Use appropriate HTTP/browser client based on method
4. Include proper error handling and validation
5. Follow the existing pattern from the base class
6. Add specific parsing logic for the expected data format
7. Include comprehensive docstrings
8. Handle news website patterns if applicable
9. No imports beyond standard library and project modules

Generate only the class implementation code:

```python"""

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