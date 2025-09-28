# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Data Fetch Abstraction system that creates specification-driven data fetching modules. The project builds a `DataFetch` class abstraction that can automatically fetch and structure data from web sources, particularly news websites.

**Central Strategy**: Design a single, robust abstract class (DataFetch) that AI can implement quickly and efficiently. Everything else in this repository supports this goal.

## Architecture & Core Components

### 1. DataFetch Abstract Class (`src/core/datafetch.py`)
- **IMMUTABLE DESIGN**: AI agents can only implement this class, never modify it
- Provides sync/async interfaces: `fetch()`, `afetch()`, `fetch_stream()`, `afetch_stream()`
- Built-in security: input sanitization, URL validation, prompt injection prevention  
- Comprehensive error handling with custom exceptions: `ValidationError`, `FetchError`, `SecurityError`
- Performance metrics collection and caching integration
- Designed specifically for news websites (NYT, Reuters, BBC, CNN, etc.)

### 2. Three-Part Specification System (`src/spec/parser.py`)
- **RawTextParser**: Intelligently parses user descriptions, extracts URLs, infers data structure
- **URLManager**: Validates and normalizes URLs with security checks
- **StructureDefinition**: Generates and validates JSON schemas

### 3. Collectors (`src/collectors/`)
- **HTTPClient**: Requests-based implementation optimized for news sites with retry logic
- **BrowserClient**: Playwright-based for dynamic content and JavaScript-heavy sites
- Both support multiple data formats (JSON, XML, HTML, CSV) with intelligent parsing

### 4. Infrastructure Components
- **AI Integration** (`src/ai/claude_client.py`): Claude 3 Haiku API with security safeguards for URL discovery and structure inference
- **Caching** (`src/cache/redis_client.py`): High-performance Redis client with compression
- **Storage** (`src/storage/s3_manager.py`): AWS S3 with lifecycle management and compression

## Development Commands

```bash
# Environment setup
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v
pytest tests/ -m unit -v      # Unit tests only
pytest tests/ -m integration -v  # Integration tests only

# Linting and formatting
black src/ tests/
flake8 src/ tests/
mypy src/

# Docker development
make docker-build
make docker-test
make docker-dev

# Quick validation
python example_usage.py
```

## Key Implementation Patterns

### Creating DataFetch Implementations
When implementing DataFetch subclasses:

1. **Always inherit from DataFetch**: `class MyFetch(DataFetch):`
2. **Implement all abstract methods**:
   - `fetch()` - synchronous fetch
   - `afetch()` - asynchronous fetch  
   - `fetch_stream()` - streaming generator
   - `afetch_stream()` - async streaming
   - `validate_data()` - data validation
   - `extract_structure()` - structure inference
   - `sanitize_input()` - security sanitization

3. **Use provided clients**: Import HTTPClient or BrowserClient, don't create from scratch
4. **Follow security patterns**: Always sanitize inputs, validate URLs
5. **Handle errors properly**: Use custom exception types
6. **Support caching**: Integrate with Redis client when available

### Example Implementation Structure
```python
from src.core.datafetch import DataFetch, FetchResult
from src.collectors.http_client import HTTPClient

class NewsArticleFetch(DataFetch):
    def __init__(self, spec, cache_client=None):
        super().__init__(spec, cache_client)
        self.http_client = HTTPClient(spec, cache_client)
    
    def fetch(self) -> FetchResult:
        return self.http_client.fetch()
    
    # Implement other required methods...
```

## Testing Patterns

### Test Structure
- `tests/test_*.py` - Main test files
- `tests/unit/` - Unit tests (no external dependencies)  
- `tests/integration/` - Integration tests (require services)
- `conftest.py` - Shared fixtures and configuration

### Key Fixtures Available
- `sample_fetch_spec` - Standard FetchSpec for testing
- `sample_fetch_result` - Sample FetchResult object
- `mock_redis_client`, `mock_s3_manager`, `mock_gemini_client` - Mock clients
- `redis_client`, `s3_manager`, `gemini_client` - Real clients (integration tests)

### Test Markers
- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests requiring external services
- `@pytest.mark.slow` - Tests that take longer to execute

## Security Considerations

### Input Validation
- All user inputs must be sanitized using `sanitize_input()`
- URLs validated with `_is_safe_url()` - blocks localhost, malicious schemes
- Prompt injection prevention in AI interactions

### Safe Patterns
```python
# Good: Always sanitize user input
clean_input = self.sanitize_input(raw_user_input)

# Good: Validate URLs before fetching  
if not self._is_safe_url(url):
    raise SecurityError("Unsafe URL")

# Good: Use timeout and retry limits
spec = FetchSpec(timeout=30, retry_count=3)
```

## Performance Optimization

### Caching Strategy
- Redis for <1s latency requirement
- S3 for long-term storage with compression
- Built-in cache key generation in base class

### Concurrency
- Use async methods when possible: `afetch()`, `afetch_stream()`
- HTTPClient and BrowserClient both support async operations
- Connection pooling built into clients

## Common Issues & Solutions

### Import Errors
- Always use absolute imports: `from src.core.datafetch import ...`
- Add project root to Python path if needed

### Missing Dependencies
- Use Docker environment: `make docker-dev`
- Or install with: `pip install -r requirements.txt`

### News Website Specific
- HTTPClient is optimized for news sites by default
- Use BrowserClient only when JavaScript rendering needed
- Built-in article extraction patterns for common news sites

## File Organization

```
src/
├── core/datafetch.py          # Abstract base class (IMMUTABLE)
├── spec/parser.py             # Specification parsing system
├── collectors/
│   ├── http_client.py         # Requests-based client
│   └── browser_client.py      # Playwright browser client
├── ai/gemini_client.py        # AI integration
├── cache/redis_client.py      # Redis caching
└── storage/s3_manager.py      # S3 storage

tests/
├── test_core_datafetch.py     # Core class tests
├── test_http_client.py        # HTTP client tests
├── test_spec_parser.py        # Parser tests
└── conftest.py               # Test configuration
```

## Environment Variables

Required for full functionality:
```bash
# AI Integration - Currently using Claude 3 Haiku
ANTHROPIC_API_KEY=your_anthropic_api_key
# Alternative: CLAUDE_API_KEY=your_anthropic_api_key

# AWS Storage
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1

# Redis Cache
REDIS_HOST=localhost
REDIS_PORT=6379

# Optional
DATAFETCH_ENV=development
```

## Deployment

### Docker (Recommended)
```bash
# Development
make docker-dev

# Production  
make docker-prod

# Testing
make docker-test
```

### Manual Setup
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key  # Claude 3 Haiku API key
python example_usage.py
```

## Important Constraints

1. **DataFetch class is IMMUTABLE** - never modify `src/core/datafetch.py`
2. **Prioritize Requests over Playwright** - use browser client only when necessary
3. **News website focus** - optimize for NYT, Reuters, BBC, CNN patterns  
4. **Security first** - always validate inputs and URLs
5. **Sub-1s latency target** - use caching effectively
6. **AI safety** - prevent prompt injection in Claude Haiku interactions (fast and cost-effective for URL/structure discovery)
- always use git conmmit and git push after modifying the code. this is crutial for backups. in addition, once you've developed a new feature, you must run 'npm run build' command, check if there are any errors, only if there are errors - fix it all and and then commit and push the code to the branch you are working on only.