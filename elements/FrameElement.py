import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import time

@dataclass(slots=True)
class FrameElement:
    source: str
    frame_num: int 
    frame: np.ndarray = field(repr=False)          # не печатаем кадр
        
    timestamp: float = 0.0
    
    frame_result: Optional[np.ndarray] = field(default=None, repr=False)

    # YOLO output
    detected_conf: Optional[list] = None
    detected_cls: Optional[list] = None
    detected_xyxy: Optional[list[list]] = None
    
    # classifier output
    vehicle_cls_id_list: Optional[list] | None = None
    vehicle_cls_name_list: Optional[list] | None = None
    vehicle_cls_conf_list: Optional[list] | None = None

    # tracking output
    tracked_conf: Optional[list] = None
    tracked_cls: Optional[list] = None
    tracked_xyxy: Optional[list[list]] = None
    id_list: Optional[list] = None

    # postprocessing
    buffer_tracks: Optional[dict] = None
    info: dict = field(default_factory=dict)       # ВАЖНО: default_factory
    send_info_of_frame_to_db: bool = False

    # время обработки кадра (ставим автоматически)
    timestamp_date: float = field(default_factory=time.time, repr=False)

