import redis
from functools import wraps
import json
from backend.config import get_redis_url

redis_client = redis.Redis.from_url(get_redis_url())

def cache_result(expire_time=3600):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create a cache key based on the function name and arguments
            key = f"{func.__name__}:{str(args)}:{str(kwargs)}"

            # Try to get the cached result
            cached_result = redis_client.get(key)
            if cached_result:
                return json.loads(cached_result)

            # If not cached, call the function and cache the result
            result = await func(*args, **kwargs)
            redis_client.setex(key, expire_time, json.dumps(result))

            return result
        return wrapper
    return decorator

def clear_cache():
    redis_client.flushall()
