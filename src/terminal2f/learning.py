import time
from typing import Self

class Timer:
    def __init__(self, name: str):
        self.name = name

    def __enter__(self) -> Self:
        self._start_time: float = time.perf_counter()
        print("Timer started!")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._end_time: float = time.perf_counter()
        elapsed_time: float = self._end_time - self._start_time
        print(f"Elapsed time: {elapsed_time:.4f}s")
        print(exc_type, exc_val, exc_tb)


timer = Timer("adams")
timer.__enter__()
time.sleep(4)
print(timer._start_time)
