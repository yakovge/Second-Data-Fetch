import asyncio
import json
import pickle
import gzip
import hashlib
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import time

import redis
import redis.asyncio as aioredis
from redis.exceptions import RedisError, ConnectionError, TimeoutError

from ..core.datafetch import FetchResult, FetchSpec


@dataclass
class CacheConfig:
    host: str
    port: int
    db: int
    password: Optional[str]
    max_connections: int
    socket_timeout: int
    socket_connect_timeout: int
    decode_responses: bool
    compression_enabled: bool
    compression_threshold: int
    key_prefix: str
    default_ttl: int


@dataclass
class CacheStats:
    hits: int
    misses: int
    stores: int
    deletes: int
    errors: int
    total_size_bytes: int
    average_response_time_ms: float


class RedisClient:
    """
    Redis caching client with compression, connection pooling, and high availability.
    
    Features:
    - Automatic compression for large values
    - Connection pooling and retry logic
    - Async and sync interfaces
    - Performance metrics and monitoring
    - TTL management and cache invalidation
    - Serialization with pickle for complex objects
    """
    
    DEFAULT_CONFIG = CacheConfig(
        host='localhost',
        port=6379,
        db=0,
        password=None,
        max_connections=20,
        socket_timeout=5,
        socket_connect_timeout=5,
        decode_responses=False,  # We handle binary data
        compression_enabled=True,
        compression_threshold=1024,  # Compress if > 1KB
        key_prefix='datafetch:',
        default_ttl=3600  # 1 hour
    )
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """
        Initialize Redis client.
        
        Args:
            config: Cache configuration, uses defaults if None
        """
        self.config = config or self.DEFAULT_CONFIG
        self.logger = logging.getLogger("datafetch.cache.redis")
        self.logger.setLevel(logging.INFO)
        
        self._sync_pool = None
        self._async_pool = None
        self._stats = CacheStats(0, 0, 0, 0, 0, 0, 0.0)
        self._response_times = []
        
        self._setup_connection_pools()
    
    def _setup_connection_pools(self):
        """Setup Redis connection pools."""
        try:
            # Sync connection pool
            self._sync_pool = redis.ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                max_connections=self.config.max_connections,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                decode_responses=self.config.decode_responses,
                retry_on_timeout=True,
                retry_on_error=[ConnectionError, TimeoutError],
                retry=redis.Retry(retries=3, backoff=redis.ExponentialBackoff())
            )
            
            # Async connection pool
            self._async_pool = aioredis.ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                max_connections=self.config.max_connections,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                decode_responses=self.config.decode_responses,
                retry_on_timeout=True,
                retry_on_error=[ConnectionError, TimeoutError],
                retry=aioredis.Retry(backoff=aioredis.ExponentialBackoff(), retries=3)
            )
            
            self.logger.info("Redis connection pools initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to setup Redis connection pools: {str(e)}")
            raise
    
    def _get_sync_client(self) -> redis.Redis:
        """Get synchronous Redis client."""
        return redis.Redis(connection_pool=self._sync_pool)
    
    def _get_async_client(self) -> aioredis.Redis:
        """Get asynchronous Redis client."""
        return aioredis.Redis(connection_pool=self._async_pool)
    
    def _create_key(self, key: str) -> str:
        """Create prefixed cache key."""
        return f"{self.config.key_prefix}{key}"
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage."""
        try:
            # Use pickle for complex objects
            serialized = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Compress if enabled and above threshold
            if (self.config.compression_enabled and 
                len(serialized) > self.config.compression_threshold):
                compressed = gzip.compress(serialized)
                # Add compression marker
                return b'GZIP:' + compressed
            
            return serialized
            
        except Exception as e:
            self.logger.error(f"Serialization failed: {str(e)}")
            raise
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        try:
            # Check for compression marker
            if data.startswith(b'GZIP:'):
                compressed_data = data[5:]  # Remove marker
                decompressed = gzip.decompress(compressed_data)
                return pickle.loads(decompressed)
            
            return pickle.loads(data)
            
        except Exception as e:
            self.logger.error(f"Deserialization failed: {str(e)}")
            raise
    
    def _record_response_time(self, response_time_ms: float):
        """Record response time for stats."""
        self._response_times.append(response_time_ms)
        # Keep only last 1000 measurements
        if len(self._response_times) > 1000:
            self._response_times.pop(0)
        
        # Update average
        if self._response_times:
            self._stats.average_response_time_ms = sum(self._response_times) / len(self._response_times)
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache (sync).
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        start_time = time.time()
        
        try:
            client = self._get_sync_client()
            cache_key = self._create_key(key)
            
            data = client.get(cache_key)
            
            response_time_ms = (time.time() - start_time) * 1000
            self._record_response_time(response_time_ms)
            
            if data is None:
                self._stats.misses += 1
                return None
            
            value = self._deserialize_value(data)
            self._stats.hits += 1
            
            self.logger.debug(f"Cache hit for key: {key}")
            return value
            
        except RedisError as e:
            self._stats.errors += 1
            self.logger.warning(f"Redis get failed for key {key}: {str(e)}")
            return None
        except Exception as e:
            self._stats.errors += 1
            self.logger.error(f"Cache get error for key {key}: {str(e)}")
            return None
    
    async def aget(self, key: str) -> Optional[Any]:
        """
        Get value from cache (async).
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        start_time = time.time()
        
        try:
            client = self._get_async_client()
            cache_key = self._create_key(key)
            
            data = await client.get(cache_key)
            
            response_time_ms = (time.time() - start_time) * 1000
            self._record_response_time(response_time_ms)
            
            if data is None:
                self._stats.misses += 1
                return None
            
            value = self._deserialize_value(data)
            self._stats.hits += 1
            
            self.logger.debug(f"Cache hit for key: {key}")
            return value
            
        except RedisError as e:
            self._stats.errors += 1
            self.logger.warning(f"Redis async get failed for key {key}: {str(e)}")
            return None
        except Exception as e:
            self._stats.errors += 1
            self.logger.error(f"Cache async get error for key {key}: {str(e)}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache (sync).
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)
            
        Returns:
            True if successful, False otherwise
        """
        start_time = time.time()
        
        try:
            client = self._get_sync_client()
            cache_key = self._create_key(key)
            ttl_seconds = ttl or self.config.default_ttl
            
            serialized_data = self._serialize_value(value)
            
            result = client.setex(cache_key, ttl_seconds, serialized_data)
            
            response_time_ms = (time.time() - start_time) * 1000
            self._record_response_time(response_time_ms)
            
            if result:
                self._stats.stores += 1
                self._stats.total_size_bytes += len(serialized_data)
                self.logger.debug(f"Cache set for key: {key}, size: {len(serialized_data)} bytes")
                return True
            
            return False
            
        except RedisError as e:
            self._stats.errors += 1
            self.logger.warning(f"Redis set failed for key {key}: {str(e)}")
            return False
        except Exception as e:
            self._stats.errors += 1
            self.logger.error(f"Cache set error for key {key}: {str(e)}")
            return False
    
    async def aset(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache (async).
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)
            
        Returns:
            True if successful, False otherwise
        """
        start_time = time.time()
        
        try:
            client = self._get_async_client()
            cache_key = self._create_key(key)
            ttl_seconds = ttl or self.config.default_ttl
            
            serialized_data = self._serialize_value(value)
            
            result = await client.setex(cache_key, ttl_seconds, serialized_data)
            
            response_time_ms = (time.time() - start_time) * 1000
            self._record_response_time(response_time_ms)
            
            if result:
                self._stats.stores += 1
                self._stats.total_size_bytes += len(serialized_data)
                self.logger.debug(f"Cache set for key: {key}, size: {len(serialized_data)} bytes")
                return True
            
            return False
            
        except RedisError as e:
            self._stats.errors += 1
            self.logger.warning(f"Redis async set failed for key {key}: {str(e)}")
            return False
        except Exception as e:
            self._stats.errors += 1
            self.logger.error(f"Cache async set error for key {key}: {str(e)}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        Delete value from cache (sync).
        
        Args:
            key: Cache key
            
        Returns:
            True if key existed and was deleted, False otherwise
        """
        try:
            client = self._get_sync_client()
            cache_key = self._create_key(key)
            
            result = client.delete(cache_key)
            
            if result > 0:
                self._stats.deletes += 1
                self.logger.debug(f"Cache delete for key: {key}")
                return True
            
            return False
            
        except RedisError as e:
            self._stats.errors += 1
            self.logger.warning(f"Redis delete failed for key {key}: {str(e)}")
            return False
        except Exception as e:
            self._stats.errors += 1
            self.logger.error(f"Cache delete error for key {key}: {str(e)}")
            return False
    
    async def adelete(self, key: str) -> bool:
        """
        Delete value from cache (async).
        
        Args:
            key: Cache key
            
        Returns:
            True if key existed and was deleted, False otherwise
        """
        try:
            client = self._get_async_client()
            cache_key = self._create_key(key)
            
            result = await client.delete(cache_key)
            
            if result > 0:
                self._stats.deletes += 1
                self.logger.debug(f"Cache delete for key: {key}")
                return True
            
            return False
            
        except RedisError as e:
            self._stats.errors += 1
            self.logger.warning(f"Redis async delete failed for key {key}: {str(e)}")
            return False
        except Exception as e:
            self._stats.errors += 1
            self.logger.error(f"Cache async delete error for key {key}: {str(e)}")
            return False
    
    def exists(self, key: str) -> bool:
        """
        Check if key exists in cache (sync).
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists, False otherwise
        """
        try:
            client = self._get_sync_client()
            cache_key = self._create_key(key)
            
            return client.exists(cache_key) > 0
            
        except RedisError as e:
            self.logger.warning(f"Redis exists failed for key {key}: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Cache exists error for key {key}: {str(e)}")
            return False
    
    async def aexists(self, key: str) -> bool:
        """
        Check if key exists in cache (async).
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists, False otherwise
        """
        try:
            client = self._get_async_client()
            cache_key = self._create_key(key)
            
            return await client.exists(cache_key) > 0
            
        except RedisError as e:
            self.logger.warning(f"Redis async exists failed for key {key}: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Cache async exists error for key {key}: {str(e)}")
            return False
    
    def get_ttl(self, key: str) -> int:
        """
        Get TTL for key (sync).
        
        Args:
            key: Cache key
            
        Returns:
            TTL in seconds, -1 if no TTL, -2 if key doesn't exist
        """
        try:
            client = self._get_sync_client()
            cache_key = self._create_key(key)
            
            return client.ttl(cache_key)
            
        except RedisError as e:
            self.logger.warning(f"Redis ttl failed for key {key}: {str(e)}")
            return -2
        except Exception as e:
            self.logger.error(f"Cache ttl error for key {key}: {str(e)}")
            return -2
    
    async def aget_ttl(self, key: str) -> int:
        """
        Get TTL for key (async).
        
        Args:
            key: Cache key
            
        Returns:
            TTL in seconds, -1 if no TTL, -2 if key doesn't exist
        """
        try:
            client = self._get_async_client()
            cache_key = self._create_key(key)
            
            return await client.ttl(cache_key)
            
        except RedisError as e:
            self.logger.warning(f"Redis async ttl failed for key {key}: {str(e)}")
            return -2
        except Exception as e:
            self.logger.error(f"Cache async ttl error for key {key}: {str(e)}")
            return -2
    
    def expire(self, key: str, ttl: int) -> bool:
        """
        Set expiration for key (sync).
        
        Args:
            key: Cache key
            ttl: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            client = self._get_sync_client()
            cache_key = self._create_key(key)
            
            return client.expire(cache_key, ttl)
            
        except RedisError as e:
            self.logger.warning(f"Redis expire failed for key {key}: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Cache expire error for key {key}: {str(e)}")
            return False
    
    async def aexpire(self, key: str, ttl: int) -> bool:
        """
        Set expiration for key (async).
        
        Args:
            key: Cache key
            ttl: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            client = self._get_async_client()
            cache_key = self._create_key(key)
            
            return await client.expire(cache_key, ttl)
            
        except RedisError as e:
            self.logger.warning(f"Redis async expire failed for key {key}: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Cache async expire error for key {key}: {str(e)}")
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """
        Clear keys matching pattern (sync).
        
        Args:
            pattern: Pattern to match (Redis glob-style)
            
        Returns:
            Number of keys deleted
        """
        try:
            client = self._get_sync_client()
            search_pattern = self._create_key(pattern)
            
            keys = client.keys(search_pattern)
            if keys:
                deleted = client.delete(*keys)
                self._stats.deletes += deleted
                self.logger.info(f"Cleared {deleted} keys matching pattern: {pattern}")
                return deleted
            
            return 0
            
        except RedisError as e:
            self._stats.errors += 1
            self.logger.warning(f"Redis clear pattern failed for {pattern}: {str(e)}")
            return 0
        except Exception as e:
            self._stats.errors += 1
            self.logger.error(f"Cache clear pattern error for {pattern}: {str(e)}")
            return 0
    
    async def aclear_pattern(self, pattern: str) -> int:
        """
        Clear keys matching pattern (async).
        
        Args:
            pattern: Pattern to match (Redis glob-style)
            
        Returns:
            Number of keys deleted
        """
        try:
            client = self._get_async_client()
            search_pattern = self._create_key(pattern)
            
            keys = await client.keys(search_pattern)
            if keys:
                deleted = await client.delete(*keys)
                self._stats.deletes += deleted
                self.logger.info(f"Cleared {deleted} keys matching pattern: {pattern}")
                return deleted
            
            return 0
            
        except RedisError as e:
            self._stats.errors += 1
            self.logger.warning(f"Redis async clear pattern failed for {pattern}: {str(e)}")
            return 0
        except Exception as e:
            self._stats.errors += 1
            self.logger.error(f"Cache async clear pattern error for {pattern}: {str(e)}")
            return 0
    
    def health_check(self) -> bool:
        """
        Check Redis connection health.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            client = self._get_sync_client()
            result = client.ping()
            return result is True
            
        except Exception as e:
            self.logger.error(f"Redis health check failed: {str(e)}")
            return False
    
    async def ahealth_check(self) -> bool:
        """
        Check Redis connection health (async).
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            client = self._get_async_client()
            result = await client.ping()
            return result is True
            
        except Exception as e:
            self.logger.error(f"Redis async health check failed: {str(e)}")
            return False
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return CacheStats(
            hits=self._stats.hits,
            misses=self._stats.misses,
            stores=self._stats.stores,
            deletes=self._stats.deletes,
            errors=self._stats.errors,
            total_size_bytes=self._stats.total_size_bytes,
            average_response_time_ms=self._stats.average_response_time_ms
        )
    
    def reset_stats(self):
        """Reset cache statistics."""
        self._stats = CacheStats(0, 0, 0, 0, 0, 0, 0.0)
        self._response_times = []
        self.logger.info("Cache stats reset")
    
    def get_info(self) -> Dict[str, Any]:
        """Get Redis server info."""
        try:
            client = self._get_sync_client()
            info = client.info()
            
            return {
                'redis_version': info.get('redis_version', 'unknown'),
                'used_memory_human': info.get('used_memory_human', 'unknown'),
                'connected_clients': info.get('connected_clients', 0),
                'total_commands_processed': info.get('total_commands_processed', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'uptime_in_seconds': info.get('uptime_in_seconds', 0)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get Redis info: {str(e)}")
            return {}
    
    async def aclose(self):
        """Close async connection pool."""
        if self._async_pool:
            await self._async_pool.disconnect()
            self.logger.info("Async Redis connection pool closed")
    
    def close(self):
        """Close sync connection pool."""
        if self._sync_pool:
            self._sync_pool.disconnect()
            self.logger.info("Sync Redis connection pool closed")
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except Exception:
            pass


# Convenience functions for FetchResult caching
def cache_fetch_result(redis_client: RedisClient, 
                      cache_key: str, 
                      result: FetchResult, 
                      ttl: Optional[int] = None) -> bool:
    """
    Cache a FetchResult object.
    
    Args:
        redis_client: Redis client instance
        cache_key: Cache key to use
        result: FetchResult to cache
        ttl: Time to live in seconds
        
    Returns:
        True if successful, False otherwise
    """
    return redis_client.set(cache_key, result, ttl)


async def acache_fetch_result(redis_client: RedisClient, 
                             cache_key: str, 
                             result: FetchResult, 
                             ttl: Optional[int] = None) -> bool:
    """
    Cache a FetchResult object (async).
    
    Args:
        redis_client: Redis client instance
        cache_key: Cache key to use
        result: FetchResult to cache
        ttl: Time to live in seconds
        
    Returns:
        True if successful, False otherwise
    """
    return await redis_client.aset(cache_key, result, ttl)


def get_cached_fetch_result(redis_client: RedisClient, cache_key: str) -> Optional[FetchResult]:
    """
    Get cached FetchResult.
    
    Args:
        redis_client: Redis client instance
        cache_key: Cache key to lookup
        
    Returns:
        Cached FetchResult or None if not found
    """
    return redis_client.get(cache_key)


async def aget_cached_fetch_result(redis_client: RedisClient, cache_key: str) -> Optional[FetchResult]:
    """
    Get cached FetchResult (async).
    
    Args:
        redis_client: Redis client instance
        cache_key: Cache key to lookup
        
    Returns:
        Cached FetchResult or None if not found
    """
    return await redis_client.aget(cache_key)