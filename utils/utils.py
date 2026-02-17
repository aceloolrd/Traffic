import logging
import time
import numpy as np
from dataclasses import fields, is_dataclass

logger_profile = logging.getLogger("profile")

def profile_time(func):
    def exec_and_print_status(*args, **kwargs):
        t_start = time.time()
        out = func(*args, **kwargs)
        t_end = time.time()
        dt_msecs = (t_end - t_start) * 1000

        self = args[0]
        logger_profile.debug(
            f"{self.__class__.__name__}.{func.__name__}, time spent {dt_msecs:.2f} msecs"
        )
        return out

    return exec_and_print_status


def print_dataclass_vertical(obj, skip=("frame", "roads_info", "frame_result")):
    assert is_dataclass(obj)
    for f in fields(obj):
        name = f.name
        val = getattr(obj, name)

        if name in skip:
            if name == "frame" and val is not None:
                print(f"{name}: shape={val.shape}, dtype={val.dtype}")
            elif name == "roads_info" and val is not None:
                keys = list(val.keys())
                print(f"{name}: keys={keys[:10]}{' ...' if len(keys) > 10 else ''}")
            else:
                print(f"{name}: <skipped>")
        else:
            print(f"{name}: {val!r}")
            
class FPS_Counter:
    def __init__(self, calc_time_perion_N_frames: int) -> None:
        """Счетчик FPS по ограниченным участкам видео (скользящему окну).

        Args:
            calc_time_perion_N_frames (int): количество фреймов окна подсчета статистики.
        """
        self.time_buffer = []
        self.calc_time_perion_N_frames = calc_time_perion_N_frames

    def calc_FPS(self) -> float:
        """Производит рассчет FPS по нескольким кадрам видео.

        Returns:
            float: значение FPS.
        """
        time_buffer_is_full = len(self.time_buffer) == self.calc_time_perion_N_frames
        t = time.time()
        self.time_buffer.append(t)

        if time_buffer_is_full:
            self.time_buffer.pop(0)
            fps = len(self.time_buffer) / (self.time_buffer[-1] - self.time_buffer[0])
            return np.round(fps, 2)
        else:
            return 0.0