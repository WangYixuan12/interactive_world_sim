import time
from typing import Callable


def precise_sleep(
    dt: float, slack_time: float = 0.001, time_func: Callable = time.monotonic
) -> None:
    """Use hybrid of time.sleep and spinning to minimize jitter.

    Sleep dt - slack_time seconds first, then spin for the rest.
    """
    t_start = time_func()
    if dt > slack_time:
        time.sleep(dt - slack_time)
    t_end = t_start + dt
    while time_func() < t_end:
        pass
    return


def precise_wait(
    t_end: float, slack_time: float = 0.001, time_func: Callable = time.monotonic
) -> None:
    t_start = time_func()
    t_wait = t_end - t_start
    if t_wait > 0:
        t_sleep = t_wait - slack_time
        if t_sleep > 0:
            time.sleep(t_sleep)
        while time_func() < t_end:
            pass
    return
