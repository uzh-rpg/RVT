import atexit
import time
from functools import wraps

import numpy as np
import torch

cuda_timers = {}
timers = {}


class CudaTimer:
    def __init__(self, device: torch.device, timer_name: str):
        assert isinstance(device, torch.device)
        assert isinstance(timer_name, str)
        self.timer_name = timer_name
        if self.timer_name not in cuda_timers:
            cuda_timers[self.timer_name] = []

        self.device = device
        self.start = None
        self.end = None

    def __enter__(self):
        torch.cuda.synchronize(device=self.device)
        self.start = time.time()
        return self

    def __exit__(self, *args):
        assert self.start is not None
        torch.cuda.synchronize(device=self.device)
        end = time.time()
        cuda_timers[self.timer_name].append(end - self.start)


def cuda_timer_decorator(device: torch.device, timer_name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with CudaTimer(device=device, timer_name=timer_name):
                out = func(*args, **kwargs)
            return out

        return wrapper

    return decorator


class TimerDummy:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


class Timer:
    def __init__(self, timer_name=''):
        self.timer_name = timer_name
        if self.timer_name not in timers:
            timers[self.timer_name] = []

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        end = time.time()
        time_diff_s = end - self.start  # measured in seconds
        timers[self.timer_name].append(time_diff_s)


def print_timing_info():
    print('== Timing statistics ==')
    skip_warmup = 10
    for timer_name, timing_values in [*cuda_timers.items(), *timers.items()]:
        if len(timing_values) <= skip_warmup:
            continue
        values = timing_values[skip_warmup:]
        timing_value_s_mean = np.mean(np.array(values))
        timing_value_s_median = np.median(np.array(values))
        timing_value_ms_mean = timing_value_s_mean * 1000
        timing_value_ms_median = timing_value_s_median * 1000
        if timing_value_ms_mean > 1000:
            print('{}: mean={:.2f} s, median={:.2f} s'.format(timer_name, timing_value_s_mean, timing_value_s_median))
        else:
            print(
                '{}: mean={:.2f} ms, median={:.2f} ms'.format(timer_name, timing_value_ms_mean, timing_value_ms_median))


# this will print all the timer values upon termination of any program that imported this file
atexit.register(print_timing_info)
