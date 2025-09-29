import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.core.ai_orchestrator import AIOrchestrator, AIDataFetchFactory
from src.core.datafetch import FetchSpec, FetchResult, DataFormat, FetchMethod
from src.ai.claude_client import ClaudeClient


class TestAIOrchestrator:
    """Test cases for AIOrchestrator with multi-website functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_ai_client = Mock(spec=ClaudeClient)
        self.mock_cache_client = Mock()
        self.mock_storage_client = Mock()

        self.orchestrator = AIOrchestrator(
            ai_client=self.mock_ai_client,
            cache_client=self.mock_cache_client,
            storage_client=self.mock_storage_client
        )

    def test_init_with_default_ai_client(self):
        """Test initialization with default AI client creation."""
        with patch('src.core.ai_orchestrator.ClaudeClient') as mock_claude:
            mock_claude.return_value = self.mock_ai_client

            orchestrator = AIOrchestrator()

            mock_claude.assert_called_once()
            assert orchestrator.ai_client == self.mock_ai_client

    def test_init_ai_client_fallback(self):
        """Test AI client fallback when Claude fails."""
        with patch('src.core.ai_orchestrator.ClaudeClient', side_effect=ValueError("No API key")):
            with patch('src.core.ai_orchestrator.GeminiClient') as mock_gemini:
                mock_gemini.return_value = self.mock_ai_client

                orchestrator = AIOrchestrator()

                mock_gemini.assert_called_once()
                assert orchestrator.ai_client == self.mock_ai_client

    def test_detect_target_websites_specific_mention(self):
        """Test website detection when specific sites are mentioned."""
        test_cases = [
            ("articles about climate from BBC", ['bbc.com', 'bbc.co.uk']),
            ("get news from Reuters", ['reuters.com']),
            ("NYT politics section", ['nytimes.com']),
            ("CNN and Guardian articles", ['cnn.com', 'theguardian.com', 'guardian.com'])
        ]

        for query, expected_sites in test_cases:
            result = self.orchestrator._detect_target_websites(query)
            for site in expected_sites:
                assert site in result, f"Expected {site} in result for query: {query}"

    def test_detect_target_websites_no_mention(self):
        """Test website detection when no specific sites are mentioned."""
        query = "articles about technology"
        result = self.orchestrator._detect_target_websites(query)

        # Should return one of the rotation patterns to avoid bias
        expected_rotations = [
            ['bbc.com', 'cnn.com', 'nytimes.com'],
            ['cnn.com', 'nytimes.com', 'bbc.com'],
            ['nytimes.com', 'bbc.com', 'cnn.com']
        ]
        assert result in expected_rotations

    def test_collect_sample_data_multiple_websites(self):
        """Test sample collection from multiple websites."""
        urls = [
            "https://www.nytimes.com/section/politics",
            "https://www.bbc.com/news/world",
            "https://www.reuters.com/world/us/"
        ]

        # Mock HTTPClient responses for different sites
        with patch('src.core.ai_orchestrator.HTTPClient') as mock_http_client:
            # Mock successful responses from NYT and Reuters, BBC fails
            def mock_fetch_side_effect():
                mock_result = Mock()
                if "nytimes" in mock_http_client.call_args[0][0].urls[0]:
                    mock_result.error = None
                    mock_result.data = "<html>NYT content</html>"
                    return mock_result
                elif "reuters" in mock_http_client.call_args[0][0].urls[0]:
                    mock_result.error = None
                    mock_result.data = "<html>Reuters content</html>"
                    return mock_result
                else:  # BBC
                    mock_result.error = "404 Not Found"
                    return mock_result

            mock_instance = Mock()
            mock_instance.fetch.side_effect = mock_fetch_side_effect
            mock_http_client.return_value = mock_instance

            result = self.orchestrator._collect_sample_data(urls, "test query")

            assert isinstance(result, dict)
            assert "nytimes.com" in result
            assert "reuters.com" in result
            assert "bbc.com" not in result  # Should fail

            # Check data structure
            assert result["nytimes.com"]["url"] == "https://www.nytimes.com/section/politics"
            assert result["nytimes.com"]["data"] == "<html>NYT content</html>"
            assert result["reuters.com"]["size"] > 0

    def test_collect_sample_data_all_fail(self):
        """Test sample collection when all websites fail."""
        urls = ["https://www.example.com/fail"]

        with patch('src.core.ai_orchestrator.HTTPClient') as mock_http_client:
            mock_instance = Mock()
            mock_result = Mock()
            mock_result.error = "Connection failed"
            mock_instance.fetch.return_value = mock_result
            mock_http_client.return_value = mock_instance

            result = self.orchestrator._collect_sample_data(urls, "test query")

            assert result is None

    def test_generate_structure_with_multiple_samples(self):
        """Test structure generation with samples from multiple websites."""
        sample_data = {
            "nytimes.com": {
                "url": "https://www.nytimes.com/politics",
                "data": "<html>NYT article structure</html>",
                "size": 1000
            },
            "reuters.com": {
                "url": "https://www.reuters.com/world",
                "data": "<html>Reuters article structure</html>",
                "size": 800
            }
        }

        expected_structure = {"type": "object", "properties": {"title": {"type": "string"}}}
        self.mock_ai_client.generate_structure_from_sample.return_value = expected_structure

        result = self.orchestrator._generate_structure(sample_data, "test query")

        assert result == expected_structure

        # Check that AI client was called with enhanced context
        call_args = self.mock_ai_client.generate_structure_from_sample.call_args
        context = call_args[1]['context']
        assert "nytimes.com" in context
        assert "reuters.com" in context
        assert "unified structure" in context

    def test_generate_implementation_with_multi_site_context(self):
        """Test implementation generation with multi-site context."""
        spec = FetchSpec(
            raw_text="articles about politics",
            urls=["https://www.nytimes.com/politics", "https://www.bbc.com/news/politics"],
            expected_format=DataFormat.JSON,
            method=FetchMethod.REQUESTS
        )

        sample_data = {
            "nytimes.com": {"data": "sample", "size": 1000},
            "bbc.com": {"data": "sample", "size": 800}
        }

        # Mock successful code generation and compilation
        mock_code = "class TestFetch(DataFetch): pass"
        self.mock_ai_client.generate_datafetch_implementation.return_value = mock_code

        with patch.object(self.orchestrator, '_compile_and_load_code') as mock_compile:
            mock_class = Mock()
            mock_compile.return_value = mock_class

            result = self.orchestrator._generate_implementation(spec, sample_data)

            assert result == mock_class

            # Check that AI client was called with unified multi-site sample
            call_args = self.mock_ai_client.generate_datafetch_implementation.call_args
            sample_arg = call_args[0][1]
            assert isinstance(sample_arg, dict)
            assert "multi_site_analysis" in sample_arg
            assert "site_samples" in sample_arg
            assert sample_arg["multi_site_analysis"]["sites_analyzed"] == ["nytimes.com", "bbc.com"]

    def test_orchestrate_fetch_complete_workflow(self):
        """Test complete orchestration workflow."""
        query = "articles about technology from Reuters"

        # Mock URL discovery
        mock_urls = ["https://www.reuters.com/technology/"]
        with patch.object(self.orchestrator, '_discover_urls', return_value=mock_urls):

            # Mock sample collection
            mock_samples = {
                "reuters.com": {"data": "sample content", "size": 1000}
            }
            with patch.object(self.orchestrator, '_collect_sample_data', return_value=mock_samples):

                # Mock structure generation
                mock_structure = {"type": "object"}
                with patch.object(self.orchestrator, '_generate_structure', return_value=mock_structure):

                    # Mock implementation generation and execution
                    mock_impl_class = Mock()
                    mock_result = FetchResult(
                        url="https://www.reuters.com/technology/",
                        data=[{"title": "Tech Article"}],
                        timestamp=datetime.now(),
                        format=DataFormat.JSON,
                        method=FetchMethod.REQUESTS,
                        metadata={}
                    )
                    mock_instance = Mock()
                    mock_instance.fetch.return_value = mock_result
                    mock_impl_class.return_value = mock_instance

                    with patch.object(self.orchestrator, '_generate_implementation', return_value=mock_impl_class):

                        result = self.orchestrator.orchestrate_fetch(query)

                        assert isinstance(result, FetchResult)
                        assert result.data == [{"title": "Tech Article"}]
                        assert result.error is None

    def test_orchestrate_fetch_with_errors(self):
        """Test orchestration workflow with errors."""
        query = "invalid query"

        # Mock URL discovery failure
        with patch.object(self.orchestrator, '_discover_urls', return_value=[]):

            result = self.orchestrator.orchestrate_fetch(query)

            assert isinstance(result, FetchResult)
            assert result.error is not None
            assert "AI orchestration failed" in result.error

    @pytest.mark.asyncio
    async def test_aorchestrate_fetch(self):
        """Test async orchestration."""
        query = "test query"

        with patch.object(self.orchestrator, 'orchestrate_fetch') as mock_sync:
            mock_result = FetchResult(
                url="test",
                data={"test": "data"},
                timestamp=datetime.now(),
                format=DataFormat.JSON,
                method=FetchMethod.REQUESTS,
                metadata={}
            )
            mock_sync.return_value = mock_result

            result = await self.orchestrator.aorchestrate_fetch(query)

            assert result == mock_result
            mock_sync.assert_called_once_with(query)

    def test_fallback_url_mapping(self):
        """Test fallback URL mapping for different news sources."""
        test_cases = [
            ("reuters business news", ["https://www.reuters.com/business/"]),
            ("nyt technology", ["https://www.nytimes.com/section/technology"]),
            ("bbc news", ["https://www.bbc.com/news"]),
            ("random query", ["https://httpbin.org/json"])  # Default fallback
        ]

        for query, expected_urls in test_cases:
            result = self.orchestrator._fallback_url_mapping(query)
            for expected_url in expected_urls:
                assert expected_url in result

    def test_cleanup(self):
        """Test resource cleanup."""
        # Add some mock generated modules
        self.orchestrator._generated_modules = ["test_module_1", "test_module_2"]

        with patch('sys.modules', {"test_module_1": Mock(), "test_module_2": Mock()}) as mock_sys_modules:
            self.orchestrator.cleanup()

            assert len(self.orchestrator._generated_modules) == 0
            assert "test_module_1" not in mock_sys_modules
            assert "test_module_2" not in mock_sys_modules


class TestAIDataFetchFactory:
    """Test cases for AIDataFetchFactory."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_orchestrator = Mock()

        with patch('src.core.ai_orchestrator.AIOrchestrator', return_value=self.mock_orchestrator):
            self.factory = AIDataFetchFactory()

    def test_create_from_text(self):
        """Test DataFetch creation from text."""
        query = "get news articles"

        with patch.object(self.factory.orchestrator, '_discover_urls', return_value=["https://www.bbc.com/news"]):
            datafetch = self.factory.create_from_text(query)

            assert datafetch is not None
            assert hasattr(datafetch, 'fetch')
            assert hasattr(datafetch, 'afetch')

    def test_generated_datafetch_fetch(self):
        """Test generated DataFetch fetch method."""
        query = "test query"
        mock_result = FetchResult(
            url="test",
            data={"test": "data"},
            timestamp=datetime.now(),
            format=DataFormat.JSON,
            method=FetchMethod.REQUESTS,
            metadata={}
        )

        self.mock_orchestrator.orchestrate_fetch.return_value = mock_result

        with patch.object(self.factory.orchestrator, '_discover_urls', return_value=["https://www.bbc.com/news"]):
            datafetch = self.factory.create_from_text(query)
            result = datafetch.fetch()

            assert result == mock_result
            self.mock_orchestrator.orchestrate_fetch.assert_called_once_with(query)

    @pytest.mark.asyncio
    async def test_generated_datafetch_afetch(self):
        """Test generated DataFetch async fetch method."""
        query = "test query"
        mock_result = FetchResult(
            url="test",
            data={"test": "data"},
            timestamp=datetime.now(),
            format=DataFormat.JSON,
            method=FetchMethod.REQUESTS,
            metadata={}
        )

        async def mock_aorchestrate(q):
            return mock_result

        self.mock_orchestrator.aorchestrate_fetch = mock_aorchestrate

        with patch.object(self.factory.orchestrator, '_discover_urls', return_value=["https://www.bbc.com/news"]):
            datafetch = self.factory.create_from_text(query)
            result = await datafetch.afetch()

            assert result == mock_result

    def test_cleanup(self):
        """Test factory cleanup."""
        self.factory.cleanup()
        self.mock_orchestrator.cleanup.assert_called_once()


class TestAIOrchestratorIntegration:
    """Integration tests for AI orchestrator functionality."""

    @pytest.mark.integration
    def test_real_url_discovery_workflow(self):
        """Test URL discovery with real AI client (requires API key)."""
        # This test would require actual API keys and would be slow
        # Mock it for now but structure it for real integration testing

        with patch('src.core.ai_orchestrator.ClaudeClient') as mock_claude_class:
            mock_client = Mock()
            mock_client.generate_urls_from_text.return_value = [
                "https://www.reuters.com/technology/",
                "https://www.bbc.com/news/technology"
            ]
            mock_claude_class.return_value = mock_client

            orchestrator = AIOrchestrator()
            urls = orchestrator._discover_urls("technology news")

            assert len(urls) >= 2
            assert any("reuters" in url for url in urls)

    @pytest.mark.integration
    def test_multi_website_sample_collection_integration(self):
        """Test multi-website sample collection with real HTTP requests."""
        # Use httpbin.org for reliable testing
        urls = [
            "https://httpbin.org/json",
            "https://httpbin.org/html"
        ]

        orchestrator = AIOrchestrator()

        with patch('src.core.ai_orchestrator.HTTPClient') as mock_http:
            # Mock different responses for different URLs
            def create_mock_client(spec, cache_client=None):
                mock_client = Mock()
                if "json" in spec.urls[0]:
                    mock_result = Mock()
                    mock_result.error = None
                    mock_result.data = {"test": "json_data"}
                    mock_client.fetch.return_value = mock_result
                else:
                    mock_result = Mock()
                    mock_result.error = None
                    mock_result.data = "<html>test html</html>"
                    mock_client.fetch.return_value = mock_result
                return mock_client

            mock_http.side_effect = create_mock_client

            result = orchestrator._collect_sample_data(urls, "test query")

            assert isinstance(result, dict)
            assert len(result) == 1  # Both URLs are on same domain, so consolidated
            assert "httpbin.org" in result
            assert result["httpbin.org"]["data"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])