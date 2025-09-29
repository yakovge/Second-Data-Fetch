"""
AI Orchestration Workflow - The missing central orchestrator.

This module implements the complete AI-driven DataFetch workflow described in README.md:
1. URL Discovery: Generate URLs from raw text if missing
2. Data Collection: Fetch raw data
3. Structure Generation: Use AI to define structure if missing
4. Implementation Generation: Create DataFetch class implementation via AI
5. Dynamic Execution: Compile and execute the generated code

This is the core component that makes the system truly AI-driven.
"""

import asyncio
import logging
import os
import sys
import tempfile
import importlib.util
from typing import Dict, List, Optional, Any, Type, Generator, AsyncGenerator
from datetime import datetime
import json

from ..core.datafetch import DataFetch, FetchSpec, FetchResult, DataFormat, FetchMethod
from ..spec.parser import RawTextParser, URLManager, StructureDefinition
from ..ai.gemini_client import GeminiClient
from ..ai.claude_client import ClaudeClient
from ..collectors.http_client import HTTPClient
from ..collectors.browser_client import BrowserClient


class AIOrchestrator:
    """
    Central AI orchestration system that implements the complete workflow
    from raw text to working DataFetch implementation.

    This is the missing piece that makes the system truly AI-driven.
    """

    def __init__(self,
                 ai_client: Optional[Any] = None,
                 cache_client: Optional[Any] = None,
                 storage_client: Optional[Any] = None):
        """
        Initialize AI orchestrator.

        Args:
            ai_client: Claude or Gemini client for AI operations
            cache_client: Redis or other caching client
            storage_client: S3 or other storage client
        """
        # Initialize logger first
        self.logger = logging.getLogger("datafetch.orchestrator")
        self.logger.setLevel(logging.INFO)

        self.ai_client = ai_client or self._create_default_ai_client()
        self.cache_client = cache_client
        self.storage_client = storage_client

        self.parser = RawTextParser()
        self.url_manager = URLManager()
        self.structure_definition = StructureDefinition()

        # Track generated classes for cleanup
        self._generated_modules = []

    def _create_default_ai_client(self):
        """Create default AI client if none provided. Tries Claude first, then Gemini."""
        # Try Claude first (preferred)
        try:
            self.logger.info("Attempting to create Claude client")
            return ClaudeClient()
        except ValueError as e:
            self.logger.warning(f"Could not create Claude client: {e}")

        # Fall back to Gemini
        try:
            self.logger.info("Attempting to create Gemini client")
            return GeminiClient()
        except ValueError as e:
            self.logger.warning(f"Could not create Gemini client: {e}")

        self.logger.warning("No AI client could be created - API keys missing")
        return None

    def orchestrate_fetch(self, raw_text: str) -> FetchResult:
        """
        Complete AI orchestration workflow - the main entry point.

        Args:
            raw_text: User's description of data requirements

        Returns:
            FetchResult containing fetched data

        This implements the complete workflow:
        Raw Text → AI URL Generation → AI Structure Inference →
        AI Implementation Generation → Dynamic Execution → Results
        """
        self.logger.info(f"Starting AI orchestration for: {raw_text[:100]}...")

        try:
            # Step 1: Parse raw text and discover URLs
            urls = self._discover_urls(raw_text)

            # Step 2: Collect sample data to understand structure
            sample_data = self._collect_sample_data(urls, raw_text)

            # Step 3: Generate data structure using AI
            structure_definition = self._generate_structure(sample_data, raw_text)

            # Step 4: Create FetchSpec
            spec = self._create_fetch_spec(raw_text, urls, structure_definition)

            # Step 5: Generate DataFetch implementation using AI
            implementation_class = self._generate_implementation(spec, sample_data)

            # Step 6: Execute the generated implementation
            result = self._execute_implementation(implementation_class, spec)

            self.logger.info("AI orchestration completed successfully")
            return result

        except Exception as e:
            self.logger.error(f"AI orchestration failed: {str(e)}")
            # Return error result
            return FetchResult(
                url=urls[0] if urls else "unknown",
                data={},
                timestamp=datetime.now(),
                format=DataFormat.JSON,
                method=FetchMethod.REQUESTS,
                metadata={"orchestrator_error": str(e)},
                error=f"AI orchestration failed: {str(e)}"
            )

    async def aorchestrate_fetch(self, raw_text: str) -> FetchResult:
        """Async version of orchestrate_fetch."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.orchestrate_fetch, raw_text
        )

    def _detect_target_websites(self, raw_text: str) -> List[str]:
        """
        Detect target websites mentioned in user query.

        Returns list of website domains, or default multi-site list if none specified.
        """
        text_lower = raw_text.lower()

        # Website detection patterns
        known_sites = {
            'bbc': ['bbc.com', 'bbc.co.uk'],
            'reuters': ['reuters.com'],
            'nytimes': ['nytimes.com'],
            'cnn': ['cnn.com'],
            'guardian': ['theguardian.com', 'guardian.com'],
            'wsj': ['wsj.com'],
            'washingtonpost': ['washingtonpost.com'],
            'apnews': ['apnews.com'],
            'bloomberg': ['bloomberg.com'],
            'ft': ['ft.com'],
            'economist': ['economist.com']
        }

        detected_sites = []
        for site_name, domains in known_sites.items():
            if site_name in text_lower or any(domain in text_lower for domain in domains):
                detected_sites.extend(domains)

        # If no specific sites mentioned, use diverse multi-site approach
        if not detected_sites:
            return ['nytimes.com', 'bbc.com', 'reuters.com']  # Diverse defaults

        return detected_sites

    def _discover_urls(self, raw_text: str) -> List[str]:
        """
        Discover URLs from raw text using adaptive website detection + AI.

        Args:
            raw_text: User's description

        Returns:
            List of URLs to fetch from
        """
        self.logger.info("Step 1: Discovering URLs with adaptive website detection")
        print(f"[URL DISCOVERY] Analyzing query: '{raw_text}'")

        # Detect target websites first
        target_sites = self._detect_target_websites(raw_text)
        self.logger.info(f"Detected target websites: {target_sites}")
        print(f"[URL DISCOVERY] Target websites detected: {target_sites}")

        # First, try to extract URLs from text using parser
        parsed = self.parser.parse(raw_text)
        urls = parsed.extracted_urls

        if urls:
            self.logger.info(f"Found {len(urls)} URLs in text")
            print(f"[URL DISCOVERY] Found {len(urls)} URLs directly in text:")
            for i, url in enumerate(urls, 1):
                print(f"  {i}. {url}")
            return urls

        # If no URLs found, use AI with website-aware prompting
        if self.ai_client:
            self.logger.info(f"No URLs found, using AI with adaptive generation for {len(target_sites)} websites")
            print(f"[URL DISCOVERY] No direct URLs found. Using AI to generate URLs for {len(target_sites)} websites...")
            ai_urls = self.ai_client.generate_urls_from_text(raw_text, domain_hints=target_sites)
            if ai_urls:
                self.logger.info(f"AI generated {len(ai_urls)} adaptive URLs: {ai_urls}")
                print(f"[URL DISCOVERY] AI generated {len(ai_urls)} URLs:")
                for i, url in enumerate(ai_urls, 1):
                    print(f"  {i}. {url}")
                return ai_urls

        # Fallback to hardcoded mapping (existing behavior)
        self.logger.info("Falling back to hardcoded URL mapping")
        print("[URL DISCOVERY] AI generation failed. Using fallback URL mapping...")
        fallback_urls = self._fallback_url_mapping(raw_text)
        print(f"[URL DISCOVERY] Fallback generated {len(fallback_urls)} URLs:")
        for i, url in enumerate(fallback_urls, 1):
            print(f"  {i}. {url}")
        return fallback_urls

    def _collect_sample_data(self, urls: List[str], raw_text: str) -> Optional[Any]:
        """
        Collect sample data from URLs to understand structure.

        Args:
            urls: URLs to sample from
            raw_text: Original request text

        Returns:
            Sample data or None if collection fails
        """
        self.logger.info("Step 2: Collecting sample data")
        print(f"[SAMPLE COLLECTION] Collecting sample data to understand website structure...")

        if not urls:
            print("[SAMPLE COLLECTION] No URLs available for sampling")
            return None

        sample_url = urls[0]
        print(f"[SAMPLE COLLECTION] Sampling from first URL: {sample_url}")

        # Create a simple spec for sampling
        sample_spec = FetchSpec(
            raw_text=raw_text,
            urls=[sample_url],  # Just sample first URL
            expected_format=DataFormat.HTML,
            method=FetchMethod.REQUESTS,
            timeout=15,
            retry_count=1
        )

        try:
            # Use HTTPClient for quick sampling
            client = HTTPClient(sample_spec, self.cache_client)
            result = client.fetch()

            if result.error:
                self.logger.warning(f"Sample collection failed: {result.error}")
                print(f"[SAMPLE COLLECTION] Failed to collect sample data: {result.error}")
                return None

            self.logger.info("Sample data collected successfully")
            print(f"[SAMPLE COLLECTION] Successfully collected sample data ({len(str(result.data))} characters)")
            return result.data

        except Exception as e:
            self.logger.warning(f"Sample collection error: {str(e)}")
            print(f"[SAMPLE COLLECTION] Error during sampling: {str(e)}")
            return None

    def _generate_structure(self, sample_data: Optional[Any], raw_text: str) -> Dict[str, Any]:
        """
        Generate structure definition using AI.

        Args:
            sample_data: Sample data from URLs
            raw_text: Original request text

        Returns:
            Structure definition dictionary
        """
        self.logger.info("Step 3: Generating data structure")

        if self.ai_client and sample_data:
            try:
                # Add website context to structure generation
                target_sites = self._detect_target_websites(raw_text)
                context = f"Target websites: {target_sites}. Query: {raw_text}. Adapt structure to common news article patterns."

                structure = self.ai_client.generate_structure_from_sample(
                    sample_data, context=context
                )
                self.logger.info(f"AI generated adaptive structure for {len(target_sites)} websites")
                return structure
            except Exception as e:
                self.logger.warning(f"AI structure generation failed: {str(e)}")

        # Fallback to parser inference
        parsed = self.parser.parse(raw_text)
        if parsed.inferred_structure:
            self.logger.info("Using parser-inferred structure")
            return parsed.inferred_structure

        # Final fallback - basic structure
        self.logger.info("Using basic fallback structure")
        return {"type": "object", "properties": {}}

    def _create_fetch_spec(self, raw_text: str, urls: List[str],
                          structure: Dict[str, Any]) -> FetchSpec:
        """Create FetchSpec from discovered information."""
        self.logger.info("Step 4: Creating FetchSpec")

        # Use parser suggestions for format and method
        parsed = self.parser.parse(raw_text)

        return FetchSpec(
            raw_text=raw_text,
            urls=urls,
            structure_definition=structure,
            expected_format=parsed.suggested_format,
            method=parsed.suggested_method,
            timeout=30,
            retry_count=3
        )

    def _generate_implementation(self, spec: FetchSpec,
                               sample_data: Optional[Any]) -> Type[DataFetch]:
        """
        Generate DataFetch implementation using AI.

        Args:
            spec: Fetch specification
            sample_data: Sample data for reference

        Returns:
            Generated DataFetch class
        """
        self.logger.info("Step 5: Generating DataFetch implementation")
        print(f"[IMPLEMENTATION] Generating custom DataFetch implementation...")

        if not self.ai_client:
            # Fallback to generic implementation
            self.logger.info("No AI client, using generic implementation")
            print("[IMPLEMENTATION] No AI client available. Using generic implementation.")
            return self._create_generic_implementation()

        try:
            # Enhance spec with website context for adaptive generation
            target_sites = self._detect_target_websites(spec.raw_text)
            self.logger.info(f"Generating adaptive implementation for websites: {target_sites}")
            print(f"[IMPLEMENTATION] Creating adaptive implementation for websites: {target_sites}")
            print(f"[IMPLEMENTATION] URLs to be handled: {spec.urls}")

            # Generate Python code using AI with website-specific adaptations
            code = self.ai_client.generate_datafetch_implementation(
                spec, sample_data, "src.core.datafetch.DataFetch"
            )

            # Compile and load the generated code
            implementation_class = self._compile_and_load_code(code, spec)

            self.logger.info(f"Adaptive AI implementation generated and compiled successfully for {len(target_sites)} websites")
            print(f"[IMPLEMENTATION] Successfully generated and compiled adaptive implementation")
            return implementation_class

        except Exception as e:
            self.logger.error(f"Adaptive implementation generation failed: {str(e)}")
            print(f"[IMPLEMENTATION] AI generation failed: {str(e)}. Using generic implementation.")
            # Fallback to generic implementation
            return self._create_generic_implementation()

    def _compile_and_load_code(self, code: str, spec: FetchSpec) -> Type[DataFetch]:
        """
        Compile and dynamically load generated Python code.

        Args:
            code: Generated Python code
            spec: Fetch specification

        Returns:
            Loaded DataFetch class
        """
        self.logger.info("Compiling and loading generated code")

        # Create a unique module name
        import uuid
        module_name = f"generated_datafetch_{uuid.uuid4().hex[:8]}"

        # Create full module code with necessary imports
        full_code = f'''
import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Generator, AsyncGenerator

from src.core.datafetch import DataFetch, FetchResult, DataFormat, FetchMethod, ValidationError, FetchError
from src.collectors.http_client import HTTPClient
from src.collectors.browser_client import BrowserClient

{code}
'''

        # Write to temporary file for debugging
        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(full_code)
                temp_file = f.name

            # Load the module
            spec_obj = importlib.util.spec_from_file_location(module_name, temp_file)
            module = importlib.util.module_from_spec(spec_obj)

            # Add to sys.modules for imports to work
            sys.modules[module_name] = module
            self._generated_modules.append(module_name)

            # Execute the module
            spec_obj.loader.exec_module(module)

            # Find the DataFetch class in the module
            datafetch_class = None
            for name in dir(module):
                obj = getattr(module, name)
                if (isinstance(obj, type) and
                    issubclass(obj, DataFetch) and
                    obj != DataFetch):
                    datafetch_class = obj
                    break

            if not datafetch_class:
                raise Exception("No DataFetch subclass found in generated code")

            self.logger.info(f"Successfully loaded class: {datafetch_class.__name__}")
            return datafetch_class

        except Exception as e:
            self.logger.error(f"Code compilation failed: {str(e)}")
            raise
        finally:
            # Clean up temp file
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass

    def _create_generic_implementation(self) -> Type[DataFetch]:
        """Create a generic DataFetch implementation as fallback."""

        class GenericDataFetch(DataFetch):
            """Generic DataFetch implementation used as fallback."""

            def __init__(self, spec, cache_client=None, storage_client=None, ai_client=None):
                super().__init__(spec, cache_client, storage_client, ai_client)

                # Choose client based on method
                if spec.method == FetchMethod.PLAYWRIGHT:
                    self.client = BrowserClient(spec, cache_client)
                else:
                    self.client = HTTPClient(spec, cache_client)

            def fetch(self) -> FetchResult:
                """Fetch data using the appropriate client."""
                return self.client.fetch()

            async def afetch(self) -> FetchResult:
                """Async fetch data."""
                return await self.client.afetch()

            def fetch_stream(self) -> Generator[FetchResult, None, None]:
                """Stream fetch data."""
                yield self.fetch()

            async def afetch_stream(self) -> AsyncGenerator[FetchResult, None]:
                """Async stream fetch data."""
                result = await self.afetch()
                yield result

            def validate_data(self, data: Any) -> bool:
                """Basic data validation."""
                return data is not None

            def extract_structure(self, sample_data: Any) -> Dict[str, Any]:
                """Basic structure extraction."""
                if isinstance(sample_data, dict):
                    return {"type": "object", "properties": {}}
                elif isinstance(sample_data, list):
                    return {"type": "array", "items": {"type": "object"}}
                else:
                    return {"type": "string"}

        return GenericDataFetch

    def _execute_implementation(self, implementation_class: Type[DataFetch],
                              spec: FetchSpec) -> FetchResult:
        """
        Execute the generated DataFetch implementation.

        Args:
            implementation_class: Generated DataFetch class
            spec: Fetch specification

        Returns:
            FetchResult from execution
        """
        self.logger.info("Step 6: Executing generated implementation")
        print(f"[EXECUTION] Running generated implementation on {len(spec.urls)} URLs...")
        for i, url in enumerate(spec.urls, 1):
            print(f"[EXECUTION]   {i}. {url}")

        try:
            # Instantiate the implementation
            instance = implementation_class(
                spec=spec,
                cache_client=self.cache_client,
                storage_client=self.storage_client,
                ai_client=self.ai_client
            )

            # Execute the fetch
            result = instance.fetch()

            self.logger.info("Implementation executed successfully")
            print(f"[EXECUTION] Successfully executed. Found {len(result.data) if isinstance(result.data, (list, tuple)) else 1} items")
            return result

        except Exception as e:
            self.logger.error(f"Implementation execution failed: {str(e)}")
            print(f"[EXECUTION] Execution failed: {str(e)}")
            raise

    def _fallback_url_mapping(self, raw_text: str) -> List[str]:
        """Fallback URL mapping for when AI is not available."""
        text_lower = raw_text.lower()

        if 'reuters' in text_lower:
            if 'business' in text_lower:
                return ["https://www.reuters.com/business/"]
            elif 'tech' in text_lower:
                return ["https://www.reuters.com/technology/"]
            else:
                return ["https://www.reuters.com/"]
        elif 'nyt' in text_lower or 'new york times' in text_lower:
            if 'tech' in text_lower:
                return ["https://www.nytimes.com/section/technology"]
            elif 'business' in text_lower:
                return ["https://www.nytimes.com/section/business"]
            else:
                return ["https://www.nytimes.com/"]
        elif 'bbc' in text_lower:
            return ["https://www.bbc.com/news"]
        else:
            # Test endpoint
            return ["https://httpbin.org/json"]

    def cleanup(self):
        """Clean up generated modules."""
        for module_name in self._generated_modules:
            if module_name in sys.modules:
                del sys.modules[module_name]
        self._generated_modules.clear()

    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.cleanup()
        except:
            pass


class AIDataFetchFactory:
    """
    Factory class that creates DataFetch instances using AI.

    This provides a simple interface for the AI-driven workflow.
    """

    def __init__(self,
                 ai_client: Optional[Any] = None,
                 cache_client: Optional[Any] = None,
                 storage_client: Optional[Any] = None):
        """Initialize the factory."""
        self.orchestrator = AIOrchestrator(ai_client, cache_client, storage_client)

    def create_from_text(self, raw_text: str) -> DataFetch:
        """
        Create a DataFetch instance from raw text description.

        Args:
            raw_text: User's description of data requirements

        Returns:
            Ready-to-use DataFetch instance

        This is the main entry point for creating AI-generated DataFetch instances.
        """
        # Use orchestrator to determine implementation
        # For now, we'll create the spec and return a generic implementation
        # that uses the orchestrator internally

        urls = self.orchestrator._discover_urls(raw_text)
        structure = {"type": "object"}  # Basic structure

        spec = FetchSpec(
            raw_text=raw_text,
            urls=urls,
            structure_definition=structure,
            expected_format=DataFormat.JSON,
            method=FetchMethod.REQUESTS
        )

        # Create wrapper class that uses orchestrator
        class AIGeneratedDataFetch(DataFetch):
            def __init__(self, orchestrator, spec):
                super().__init__(spec)
                self._orchestrator = orchestrator

            def fetch(self) -> FetchResult:
                return self._orchestrator.orchestrate_fetch(self.spec.raw_text)

            async def afetch(self) -> FetchResult:
                return await self._orchestrator.aorchestrate_fetch(self.spec.raw_text)

            def fetch_stream(self) -> Generator[FetchResult, None, None]:
                yield self.fetch()

            async def afetch_stream(self) -> AsyncGenerator[FetchResult, None]:
                result = await self.afetch()
                yield result

            def validate_data(self, data: Any) -> bool:
                return data is not None

            def extract_structure(self, sample_data: Any) -> Dict[str, Any]:
                return {"type": "object"}

        return AIGeneratedDataFetch(self.orchestrator, spec)

    def cleanup(self):
        """Clean up resources."""
        self.orchestrator.cleanup()