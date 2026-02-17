import torch
from ultralytics import YOLO
from elements.FrameElement import FrameElement
from elements.VideoEndBreakElement import VideoEndBreakElement
from utils.utils import profile_time
import logging 

logger = logging.getLogger(__name__)

class DetectionNode:
    def __init__(self, config) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Детекция будет производиться на {device}")
        self.model = YOLO(config["weight_pth"], task="detect")
        self.conf = float(config["confidence"])
        self.classes_to_detect = config.get("classes_to_detect", None)  # list[int] или None
        
    @profile_time
    def process(self, batch):
        """
        batch: list[FrameElement | VideoEndBreakElement] 
        return: тот же тип, но с заполненными полями det_xyxy/det_conf/det_cls
        """

        # Фильтруем/проверяем типы
        for fe in batch:
            if isinstance(fe, VideoEndBreakElement):
                return fe
            elif not isinstance(fe, FrameElement):
                raise TypeError(f"DetectionNode | ожидался FrameElement, получено {type(fe)}")

        frames = [fe.frame for fe in batch]  # список np.ndarray (BGR)

        # Один вызов модели на весь батч (любой длины)
        results = self.model.predict(
            frames,
            conf=self.conf,
            classes=self.classes_to_detect,
            verbose=False,
        )
        logger.info(f"Размер батча: {len(frames)}\nКоличество детекций: {[len(r.boxes) for r in results]}\nВсего объектов: {sum(len(r.boxes) for r in results)}")

        # Записываем результаты обратно в элементы
        for fe, r in zip(batch, results):
            det_cls, det_conf, det_xyxy = self._extract_boxes(r)
            fe.detected_cls = det_cls          # list[int]
            fe.detected_conf = det_conf        # list[float]
            fe.detected_xyxy = det_xyxy        # list[list[float]] или int (как сделаешь)

        return batch

    @staticmethod
    def _extract_boxes(r):
        """
        Возвращает (cls_list, conf_list, xyxy_list) даже если детекций нет.
        """
        b = getattr(r, "boxes", None)
        if b is None or len(b) == 0:
            return [], [], []

        # Ultralytics: b.cls/b.conf/b.xyxy — torch.Tensor
        cls_list = b.cls.detach().cpu().to(torch.int64).tolist()
        conf_list = b.conf.detach().cpu().tolist()
        xyxy_list = b.xyxy.detach().cpu().tolist()
        return cls_list, conf_list, xyxy_list
