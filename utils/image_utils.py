from typing import Optional, Tuple

import numpy as np


def crop_image(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    padding: int = 0,
    make_contiguous: bool = True,
) -> Tuple[Optional[np.ndarray], int, int]:
    h, w = frame.shape[:2]
    x1_p = max(0, x1 - padding)
    y1_p = max(0, y1 - padding)
    x2_p = min(w, x2 + padding)
    y2_p = min(h, y2 + padding)

    if x2_p <= x1_p or y2_p <= y1_p:
        return None, x1_p, y1_p

    crop = frame[y1_p:y2_p, x1_p:x2_p]
    if make_contiguous and not crop.flags["C_CONTIGUOUS"]:
        crop = np.ascontiguousarray(crop)
    return crop, x1_p, y1_p


def shift_bbox(bbox: Tuple[int, int, int, int], dx: int, dy: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
