#!/bin/bash
set -e

# DataFetch Docker Entrypoint Script
# This script handles different runtime modes and initialization

echo "🚀 Starting DataFetch container..."
echo "Environment: ${DATAFETCH_ENV:-production}"
echo "Python version: $(python --version)"

# Function to wait for services
wait_for_service() {
    local service_name=$1
    local host=$2
    local port=$3
    local timeout=${4:-30}
    
    echo "⏳ Waiting for $service_name at $host:$port..."
    
    for i in $(seq 1 $timeout); do
        if timeout 1 bash -c "echo >/dev/tcp/$host/$port" 2>/dev/null; then
            echo "✅ $service_name is ready!"
            return 0
        fi
        echo "⏳ Waiting for $service_name... ($i/$timeout)"
        sleep 1
    done
    
    echo "❌ $service_name not available after $timeout seconds"
    return 1
}

# Wait for Redis if configured
if [ -n "$REDIS_HOST" ] && [ -n "$REDIS_PORT" ]; then
    wait_for_service "Redis" "$REDIS_HOST" "$REDIS_PORT"
fi

# Wait for S3-compatible service if using LocalStack or MinIO
if [ -n "$S3_ENDPOINT_URL" ]; then
    S3_HOST=$(echo "$S3_ENDPOINT_URL" | sed 's|http[s]*://||' | cut -d':' -f1)
    S3_PORT=$(echo "$S3_ENDPOINT_URL" | sed 's|http[s]*://||' | cut -d':' -f2)
    if [ -n "$S3_HOST" ] && [ -n "$S3_PORT" ]; then
        wait_for_service "S3 Service" "$S3_HOST" "$S3_PORT"
    fi
fi

# Validate environment variables
validate_env() {
    local required_vars=()
    local warnings=()
    
    # Check for production requirements
    if [ "$DATAFETCH_ENV" = "production" ]; then
        if [ -z "$GEMINI_API_KEY" ]; then
            warnings+=("GEMINI_API_KEY not set - AI features will be disabled")
        fi
        
        if [ -z "$AWS_ACCESS_KEY_ID" ] && [ -z "$AWS_PROFILE" ]; then
            warnings+=("AWS credentials not configured - S3 storage will fail")
        fi
        
        if [ -z "$REDIS_HOST" ]; then
            warnings+=("REDIS_HOST not set - caching will be disabled")
        fi
    fi
    
    # Print warnings
    for warning in "${warnings[@]}"; do
        echo "⚠️  Warning: $warning"
    done
    
    # Check required variables
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            echo "❌ Error: Required environment variable $var is not set"
            exit 1
        fi
    done
}

# Initialize application
initialize_app() {
    echo "🔧 Initializing DataFetch application..."
    
    # Create necessary directories
    mkdir -p /app/logs /app/temp
    
    # Set permissions
    chmod 755 /app/logs /app/temp
    
    # Initialize Python path
    export PYTHONPATH="/app:$PYTHONPATH"
    
    echo "✅ Application initialized"
}

# Health check function
health_check() {
    echo "🏥 Running health checks..."
    
    # Check Python imports
    python -c "
import sys
sys.path.append('/app')

try:
    from src.core.datafetch import DataFetch
    from src.collectors.http_client import HTTPClient
    print('✅ Core modules import successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)

# Check optional dependencies
try:
    import redis
    print('✅ Redis client available')
except ImportError:
    print('⚠️ Redis not available')

try:
    import boto3
    print('✅ AWS SDK available')
except ImportError:
    print('⚠️ AWS SDK not available')

try:
    from playwright.sync_api import sync_playwright
    print('✅ Playwright available')
except ImportError:
    print('⚠️ Playwright not available')

try:
    import google.generativeai
    print('✅ Gemini AI client available')
except ImportError:
    print('⚠️ Gemini AI client not available')
"
    
    if [ $? -eq 0 ]; then
        echo "✅ Health check passed"
        return 0
    else
        echo "❌ Health check failed"
        return 1
    fi
}

# Main execution logic
main() {
    validate_env
    initialize_app
    
    case "$1" in
        "test")
            echo "🧪 Running tests..."
            health_check
            exec python -m pytest tests/ -v --tb=short
            ;;
        "test-unit")
            echo "🧪 Running unit tests..."
            exec python -m pytest tests/ -m unit -v
            ;;
        "test-integration")
            echo "🧪 Running integration tests..."
            health_check
            exec python -m pytest tests/ -m integration -v
            ;;
        "health")
            echo "🏥 Health check mode..."
            health_check
            ;;
        "shell"|"bash")
            echo "🐚 Starting interactive shell..."
            exec bash
            ;;
        "python")
            echo "🐍 Starting Python interactive shell..."
            exec python
            ;;
        "serve")
            echo "🌐 Starting web server..."
            # This would start a web server if implemented
            exec python -c "print('Web server mode not implemented yet')"
            ;;
        "worker")
            echo "⚙️ Starting background worker..."
            # This would start a background worker if implemented
            exec python -c "print('Worker mode not implemented yet')"
            ;;
        "demo")
            echo "📋 Running demo..."
            exec python -c "
import sys
sys.path.append('/app')
from src.core.datafetch import FetchSpec, DataFormat, FetchMethod
from src.collectors.http_client import HTTPClient

# Demo fetch
spec = FetchSpec(
    raw_text='Demo: fetch sample data',
    urls=['https://httpbin.org/json'],
    expected_format=DataFormat.JSON,
    method=FetchMethod.REQUESTS
)

client = HTTPClient(spec)
print('🚀 Running demo fetch...')
try:
    result = client.fetch()
    print(f'✅ Success: {result.url}')
    print(f'📊 Data keys: {list(result.data.keys()) if isinstance(result.data, dict) else type(result.data)}')
    print(f'⏱️ Execution time: {result.execution_time:.2f}s')
except Exception as e:
    print(f'❌ Demo failed: {e}')
"
            ;;
        *)
            if [ $# -eq 0 ]; then
                echo "🧪 No command specified, running default tests..."
                health_check
                exec python -m pytest tests/ -v --tb=short
            else
                echo "🚀 Executing custom command: $*"
                exec "$@"
            fi
            ;;
    esac
}

# Error handling
trap 'echo "❌ Script failed with exit code $?"; exit 1' ERR

# Execute main function
main "$@"