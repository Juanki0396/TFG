import datetime
import time

import numpy as np
import matplotlib.pyplot as plt


def run_time(f):
    def wrapper(*args, **kwargs):
        t = time.time()
        result = f(*args, **kwargs)
        t = time.time() - t
        print(f"Running time -> {datetime.timedelta(seconds=t//1)}")
        return result
    return wrapper


def plot_image(img: np.ndarray, label: int):
    fig = plt.imshow(img)
    fig.set_cmap("gray")
    plt.title(f"Label {label}")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":

    @run_time
    def hello():
        time.sleep(5)
        print("hello")

    hello()
