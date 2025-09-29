import pytest
import time
from unittest.mock import Mock, patch
from datetime import datetime

from src.core.ai_orchestrator import AIOrchestrator
from src.ai.claude_client import ClaudeClient
from src.core.datafetch import FetchSpec, DataFormat, FetchMethod
from fetch_articles import fetch_articles


@pytest.mark.integration
class TestMultiWebsiteWorkflow:
    """Integration tests for multi-website data fetching workflow."""

    def setup_method(self):
        """Set up integration test fixtures."""
        self.test_queries = [
            "articles about Trump",
            "technology news from Reuters",
            "climate change articles from BBC",
            "articles about politics"
        ]

    def test_url_discovery_diversity(self):
        """Test that URL discovery generates diverse sources."""
        orchestrator = AIOrchestrator()

        for query in self.test_queries:
            with patch.object(orchestrator.ai_client, 'generate_urls_from_text') as mock_generate:
                # Mock diverse URL responses based on query
                if "Trump" in query or "politics" in query:
                    mock_urls = [
                        "https://www.bbc.com/news/world-us-canada",
                        "https://www.reuters.com/world/us/",
                        "https://www.nytimes.com/section/politics"
                    ]
                elif "Reuters" in query:
                    mock_urls = [
                        "https://www.reuters.com/technology/",
                        "https://www.reuters.com/business/",
                        "https://www.reuters.com/world/"
                    ]
                elif "BBC" in query:
                    mock_urls = [
                        "https://www.bbc.com/news/science-environment",
                        "https://www.bbc.com/news/world",
                        "https://www.bbc.com/news/business"
                    ]
                else:
                    mock_urls = [
                        "https://www.nytimes.com/section/technology",
                        "https://www.bbc.com/news/technology",
                        "https://www.reuters.com/technology/"
                    ]

                mock_generate.return_value = mock_urls

                urls = orchestrator._discover_urls(query)

                assert len(urls) >= 3

                # Check for diversity - not all URLs from same domain
                domains = set()
                for url in urls:
                    if "nytimes" in url:
                        domains.add("nytimes")
                    elif "bbc" in url:
                        domains.add("bbc")
                    elif "reuters" in url:
                        domains.add("reuters")

                # Should have at least 2 different domains for diverse queries
                if "from" not in query:  # Multi-site queries should be diverse
                    assert len(domains) >= 2, f"Query '{query}' should generate diverse URLs, got: {urls}"

    def test_sample_collection_handles_failures_gracefully(self):
        """Test that sample collection handles individual site failures."""
        orchestrator = AIOrchestrator()

        test_urls = [
            "https://www.nytimes.com/section/politics",
            "https://www.bbc.com/news/world-us-canada",  # This will "fail"
            "https://www.reuters.com/world/us/"
        ]

        with patch('src.core.ai_orchestrator.HTTPClient') as mock_http_client:
            def create_mock_client(spec, cache_client=None):
                mock_client = Mock()
                url = spec.urls[0]

                mock_result = Mock()
                if "bbc.com" in url:
                    # Simulate BBC failure
                    mock_result.error = "404 Not Found"
                else:
                    # NYT and Reuters succeed
                    mock_result.error = None
                    if "nytimes" in url:
                        mock_result.data = "<html>NYT content with articles</html>"
                    else:
                        mock_result.data = "<html>Reuters content with news</html>"

                mock_client.fetch.return_value = mock_result
                return mock_client

            mock_http_client.side_effect = create_mock_client

            result = orchestrator._collect_sample_data(test_urls, "test query")

            # Should succeed with 2 out of 3 samples
            assert isinstance(result, dict)
            assert len(result) == 2
            assert "nytimes.com" in result
            assert "reuters.com" in result
            assert "bbc.com" not in result

    def test_implementation_generation_with_mixed_samples(self):
        """Test implementation generation with samples from different sites."""
        orchestrator = AIOrchestrator()

        spec = FetchSpec(
            raw_text="articles about technology",
            urls=[
                "https://www.nytimes.com/section/technology",
                "https://www.reuters.com/technology/"
            ],
            expected_format=DataFormat.JSON,
            method=FetchMethod.REQUESTS
        )

        sample_data = {
            "nytimes.com": {
                "url": "https://www.nytimes.com/section/technology",
                "data": "<html><article><h1>NYT Tech Article</h1></article></html>",
                "size": 1500
            },
            "reuters.com": {
                "url": "https://www.reuters.com/technology/",
                "data": "<html><div class='story'><h2>Reuters Tech News</h2></div></html>",
                "size": 1200
            }
        }

        with patch.object(orchestrator.ai_client, 'generate_datafetch_implementation') as mock_generate:
            mock_code = """
class TechFetch(DataFetch):
    def fetch(self):
        articles = []
        for url in self.spec.urls:
            if 'nytimes' in url:
                articles.extend(self._fetch_nyt_articles(url))
            elif 'reuters' in url:
                articles.extend(self._fetch_reuters_articles(url))
        return FetchResult(
            url=self.spec.urls[0],
            data=articles,
            timestamp=datetime.now(),
            format=self.spec.expected_format,
            method=self.spec.method,
            metadata={}
        )

    def _fetch_nyt_articles(self, url): return []
    def _fetch_reuters_articles(self, url): return []
    async def afetch(self): return self.fetch()
    def fetch_stream(self): yield self.fetch()
    async def afetch_stream(self): yield await self.afetch()
    def validate_data(self, data): return True
    def extract_structure(self, sample): return {}
            """
            mock_generate.return_value = mock_code

            with patch.object(orchestrator, '_compile_and_load_code') as mock_compile:
                mock_class = Mock()
                mock_compile.return_value = mock_class

                result = orchestrator._generate_implementation(spec, sample_data)

                assert result == mock_class

                # Verify AI was called with largest sample (NYT has more characters)
                call_args = mock_generate.call_args
                assert call_args[0][1] == sample_data["nytimes.com"]["data"]

    @pytest.mark.slow
    def test_complete_workflow_with_mocked_responses(self):
        """Test complete workflow from query to results with realistic mocking."""
        query = "articles about climate change"

        with patch('src.core.ai_orchestrator.ClaudeClient') as mock_claude_class:
            # Mock AI client
            mock_ai_client = Mock()

            # Mock URL generation
            mock_ai_client.generate_urls_from_text.return_value = [
                "https://www.nytimes.com/section/climate",
                "https://www.bbc.com/news/science-environment",
                "https://www.reuters.com/business/environment/"
            ]

            # Mock structure generation
            mock_ai_client.generate_structure_from_sample.return_value = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "url": {"type": "string"},
                        "source": {"type": "string"}
                    }
                }
            }

            # Mock implementation generation
            mock_ai_client.generate_datafetch_implementation.return_value = """
class ClimateFetch(DataFetch):
    def fetch(self):
        articles = [
            {"title": "Climate Article 1", "url": "https://nyt.com/1", "source": "NYT"},
            {"title": "Climate Article 2", "url": "https://bbc.com/2", "source": "BBC"}
        ]
        return FetchResult(
            url=self.spec.urls[0],
            data=articles,
            timestamp=datetime.now(),
            format=self.spec.expected_format,
            method=self.spec.method,
            metadata={}
        )
    async def afetch(self): return self.fetch()
    def fetch_stream(self): yield self.fetch()
    async def afetch_stream(self): yield await self.afetch()
    def validate_data(self, data): return True
    def extract_structure(self, sample): return {}
            """

            mock_claude_class.return_value = mock_ai_client

            # Mock HTTP sample collection
            with patch('src.core.ai_orchestrator.HTTPClient') as mock_http:
                mock_http_instance = Mock()
                mock_result = Mock()
                mock_result.error = None
                mock_result.data = "<html>Sample climate article content</html>"
                mock_http_instance.fetch.return_value = mock_result
                mock_http.return_value = mock_http_instance

                # Execute workflow
                orchestrator = AIOrchestrator()
                result = orchestrator.orchestrate_fetch(query)

                # Verify successful execution
                assert result.error is None
                assert isinstance(result.data, list)
                assert len(result.data) == 2
                assert result.data[0]["title"] == "Climate Article 1"
                assert result.data[1]["source"] == "BBC"

    def test_error_handling_complete_failure(self):
        """Test graceful error handling when everything fails."""
        orchestrator = AIOrchestrator()

        with patch.object(orchestrator, '_discover_urls', return_value=[]):
            result = orchestrator.orchestrate_fetch("impossible query")

            assert result.error is not None
            assert "AI orchestration failed" in result.error

    def test_fetch_articles_integration_with_fallback(self):
        """Test fetch_articles function integration with AI failure and fallback."""
        query = "articles about technology"

        # Mock AI orchestrator to fail
        with patch('fetch_articles.AIOrchestrator') as mock_orchestrator_class:
            mock_orchestrator = Mock()
            mock_orchestrator.orchestrate_fetch.side_effect = Exception("AI failed")
            mock_orchestrator_class.return_value = mock_orchestrator

            # Mock static collector to succeed
            with patch('fetch_articles._fetch_articles_fallback') as mock_fallback:
                mock_fallback.return_value = {
                    'articles': [
                        {'title': 'Fallback Article', 'url': 'https://test.com', 'score': 5.0}
                    ],
                    'method_used': 'Static_Collectors',
                    'total_time': 15.0,
                    'status': 'Success'
                }

                result = fetch_articles(query)

                # Should use fallback
                assert result['method_used'] == 'Static_Collectors'
                assert result['status'] == 'Success'
                assert len(result['articles']) == 1
                mock_fallback.assert_called_once()

    @pytest.mark.slow
    def test_performance_multi_website_sampling(self):
        """Test performance of multi-website sampling."""
        orchestrator = AIOrchestrator()

        test_urls = [
            "https://httpbin.org/delay/1",  # 1 second delay
            "https://httpbin.org/delay/2",  # 2 second delay
            "https://httpbin.org/status/404"  # Immediate failure
        ]

        start_time = time.time()

        with patch('src.core.ai_orchestrator.HTTPClient') as mock_http:
            def create_mock_client(spec, cache_client=None):
                mock_client = Mock()
                mock_result = Mock()

                url = spec.urls[0]
                if "delay/1" in url:
                    time.sleep(0.1)  # Simulate short delay
                    mock_result.error = None
                    mock_result.data = "delayed response 1"
                elif "delay/2" in url:
                    time.sleep(0.2)  # Simulate longer delay
                    mock_result.error = None
                    mock_result.data = "delayed response 2"
                else:
                    mock_result.error = "404 Not Found"

                mock_client.fetch.return_value = mock_result
                return mock_client

            mock_http.side_effect = create_mock_client

            result = orchestrator._collect_sample_data(test_urls, "performance test")

            end_time = time.time()
            execution_time = end_time - start_time

            # Should complete in reasonable time (parallel execution)
            assert execution_time < 2.0  # Much faster than sequential (3+ seconds)

            # Should get results from successful URLs
            assert result is not None
            assert len(result) == 2  # Two successful samples


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])