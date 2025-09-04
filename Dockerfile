# Multi-stage build for optimized production image
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_ENV=production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for better caching
COPY requirements.txt /tmp/

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r /tmp/requirements.txt

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    DATAFETCH_ENV=production

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    # Essential runtime libraries
    libc6-dev \
    libssl3 \
    ca-certificates \
    # Playwright browser dependencies
    libnss3 \
    libatk-bridge2.0-0 \
    libdrm2 \
    libxkbcommon0 \
    libgtk-3-0 \
    libgbm1 \
    libasound2 \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Install Playwright browsers (in builder stage for caching)
FROM builder as playwright-installer
RUN playwright install chromium firefox webkit --with-deps

# Final production stage
FROM production

# Copy Playwright browsers from installer
COPY --from=playwright-installer /opt/venv/lib/python3.11/site-packages/playwright /opt/venv/lib/python3.11/site-packages/playwright
COPY --from=playwright-installer /root/.cache/ms-playwright /root/.cache/ms-playwright

# Create app user for security
RUN groupadd -r datafetch && useradd -r -g datafetch datafetch

# Create application directories
RUN mkdir -p /app /app/logs /app/temp && \
    chown -R datafetch:datafetch /app

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=datafetch:datafetch src/ ./src/
COPY --chown=datafetch:datafetch tests/ ./tests/
COPY --chown=datafetch:datafetch README.md ./
COPY --chown=datafetch:datafetch pytest.ini ./
COPY --chown=datafetch:datafetch conftest.py ./

# Copy entrypoint script
COPY --chown=datafetch:datafetch docker/entrypoint.sh ./
RUN chmod +x entrypoint.sh

# Create .env file placeholder
RUN touch .env && chown datafetch:datafetch .env

# Switch to non-root user
USER datafetch

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.path.append('/app'); from src.cache.redis_client import RedisClient; print('OK')" || exit 1

# Expose port (if running as web service)
EXPOSE 8000

# Set entrypoint
ENTRYPOINT ["./entrypoint.sh"]

# Default command
CMD ["python", "-m", "pytest", "tests/", "-v"]


# Development stage (for local development)
FROM builder as development

ENV DATAFETCH_ENV=development

# Install development dependencies
RUN pip install \
    ipython \
    jupyter \
    black \
    flake8 \
    mypy \
    pre-commit

# Install Playwright browsers for development
RUN playwright install chromium firefox webkit --with-deps

# Create development directories
RUN mkdir -p /app /app/logs /app/temp /app/notebooks

WORKDIR /app

# Copy application code (in development we'll mount volumes)
COPY src/ ./src/
COPY tests/ ./tests/
COPY README.md pytest.ini conftest.py ./

# Create development entrypoint
RUN echo '#!/bin/bash\nset -e\nif [ "$1" = "test" ]; then\n    exec python -m pytest tests/ -v\nelif [ "$1" = "shell" ]; then\n    exec bash\nelif [ "$1" = "jupyter" ]; then\n    exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root\nelse\n    exec "$@"\nfi' > entrypoint-dev.sh && \
    chmod +x entrypoint-dev.sh

ENTRYPOINT ["./entrypoint-dev.sh"]
CMD ["test"]