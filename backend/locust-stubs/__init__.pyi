from typing import Callable, Any

class HttpUser:
    wait_time: Callable[[], float]
    client: Any

def task(func: Callable) -> Callable: ...
def between(min_wait: float, max_wait: float) -> Callable[[], float]: ...
