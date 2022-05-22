
import datetime
import time
from typing import Callable, Any


def run_time(f: Callable) -> Callable:
    """Decorator that meassures the execution time of a function.

    Args:
        f (Callable): Function to decorate.
    """
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        t = time.time()
        result = f(*args, **kwargs)
        t = time.time() - t
        print(f"Running time -> {datetime.timedelta(seconds=t//1)}")
        return result
    return wrapper
