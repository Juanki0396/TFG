import datetime
import time


def run_time(f):
    def wrapper(*args, **kwargs):
        t = time.time()
        result = f(*args, **kwargs)
        t = time.time() - t
        print(f"Running time -> {datetime.timedelta(seconds=t//1)}")
        return result
    return wrapper


if __name__ == "__main__":

    @run_time
    def hello():
        time.sleep(5)
        print("hello")

    hello()
