import hashlib
import json
import logging
import os
from typing import Any

import redis.asyncio as aioredis

logger = logging.getLogger("cache")


class QueryCache:
    """
    Async Redis-backed query cache.
    Keys are md5(question.strip().lower()), values are JSON-serialised responses.
    TTL is configurable via config.cache.ttl (default 3600 s).
    Redis connection is configured via REDIS_HOST / REDIS_PORT env vars.
    """

    def __init__(self, ttl: int = 3600) -> None:
        self._ttl = ttl
        host = os.getenv("REDIS_HOST", "redis")
        port = int(os.getenv("REDIS_PORT", "6379"))
        self._redis: aioredis.Redis = aioredis.Redis(
            host=host, port=port, decode_responses=True
        )
        logger.info(f"Redis cache initialised: {host}:{port}, ttl={ttl}s")

    def _key(self, question: str) -> str:
        return "rag:query:" + hashlib.md5(
            question.strip().lower().encode()
        ).hexdigest()

    async def get(self, question: str) -> Any | None:
        raw = await self._redis.get(self._key(question))
        if raw is None:
            return None
        logger.info(f"Cache hit for question hash {self._key(question)[-8:]}")
        return json.loads(raw)

    async def set(self, question: str, response: Any) -> None:
        key = self._key(question)
        # response may be a Pydantic model or plain dict
        if hasattr(response, "model_dump"):
            payload = response.model_dump()
        else:
            payload = response
        await self._redis.setex(key, self._ttl, json.dumps(payload))

    async def close(self) -> None:
        await self._redis.aclose()
