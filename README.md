# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Data Fetch Abstraction system that creates specification-driven data fetching modules. The project aims to build a `DataFetch` class abstraction that can automatically fetch and structure data from web sources.
The central idea is simple but strict:
👉 Our entire strategy is to design a single, robust abstract class (DataFetch) that AI can implement quickly and efficiently.

Everything else in this repository supports this goal.
this abstract class should be fit mostly to news websites

## Architecture & Design

### Core Components

1. **DataFetch Class** - Abstract base class for data fetching operations
   - Initially works without AI
   - Later enhanced with AI for automation

2. **Specification System** - Three-part specification:
   - Raw Text: User's description of requirements
   - URL(s): Data source(s) (initially single-URL support)
   - Structure: Data format definition (classes, types, restrictions)

3. **Workflow Pipeline**:
   - URL Discovery: Generate URLs from raw text if missing
   - Data Collection: Fetch raw data (HTTP responses, HAR files, screenshots)
   - Structure Generation: Use AI to define structure if missing
   - Implementation Generation: Create DataFetch class implementation via AI with filesystem access

### Key Design Considerations

- **Security**: Sanitize all inputs to prevent prompt injection attacks
- **Efficiency**: Use caching to avoid redundant fetches
- **Sandboxing**: Execute DataFetch classes in sandboxed environments
- **Token Optimization**: Reference external files instead of inline content in prompts
- **Concurrency**: Support multiple concurrent users

## Technical Stack

- **Language**: Python 3.10+
- **AI Model**: Gemini 1.5
- **Containerization**: Docker
- **Storage**: AWS S3 (with gzip compression)
- **Caching**: Redis
- **HTTP Client**: Requests (primary), Playwright (when needed)
- **Target Sites**: News websites (NYT, Reuters, The Telegraph, etc.)

## Development Commands

```bash
# Environment setup
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Linting
black src/ tests/
flake8 src/ tests/

# Docker commands
docker build -t datafetch .
docker run -p 8000:8000 datafetch

# Redis (local development)
docker run -d -p 6379:6379 redis
```

## Implementation Notes

### Key Constraints
- The abstract DataFetch class is **IMMUTABLE** - AI agents can only implement it, not modify it
- Focus on news websites initially (NYT, Reuters, The Telegraph)
- Prioritize Requests library, use Playwright only when necessary
- Store data in S3 with compression and TTL
- Use Redis for caching to achieve <1s latency

### Implementation Strategy
- Start with Requests-based fetching
- Add Playwright only for dynamic content
- Use Gemini 1.5 for AI-powered implementation generation
- Reference external files in prompts (don't inline data)
- Ensure proper sandboxing with Docker