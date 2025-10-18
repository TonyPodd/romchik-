"""Async utilities"""

import asyncio
from typing import TypeVar, Callable, Any
from functools import wraps

T = TypeVar('T')


async def run_in_threadpool(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """
    Запустить синхронную функцию в thread pool
    Полезно для CPU-bound операций (детекция, обработка)
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


def async_retry(max_attempts: int = 3, delay: float = 1.0):
    """
    Decorator для повторных попыток асинхронных функций

    Usage:
        @async_retry(max_attempts=3, delay=1.0)
        async def my_function():
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(delay)
            raise last_exception
        return wrapper
    return decorator
