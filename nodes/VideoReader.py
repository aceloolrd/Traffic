import os
import logging
from typing import Generator, Union
import cv2

from elements.FrameElement import FrameElement
from elements.VideoEndBreakElement import VideoEndBreakElement

logger = logging.getLogger(__name__)


class VideoReader:
    def __init__(self, config: dict) -> None:
        self.video_pth = config["src"]
        assert (os.path.isfile(self.video_pth)), f"VideoReader| Файл {self.video_pth} не найден"

        self.stream = cv2.VideoCapture(self.video_pth)

        self.skip_frames = config["skip_frames"]
        self.break_element_sent = False  # Был ли отправлен элемент прерывания видеопотока

    def process(self) -> Generator[Union[FrameElement, VideoEndBreakElement], None, None]:
        frame_number = -1
        while True:
            ret, frame = self.stream.read()
            if not ret:
                logger.warning("Can't receive frame (stream end?). Exiting ...")
                if not self.break_element_sent:
                    self.break_element_sent = True
                    # отправим VideoEndBreakElement чтобы обозначить окончание потока
                    yield VideoEndBreakElement(str(self.video_pth))
                break
            
            frame_number += 1
            
            if self.skip_frames > 1 and frame_number % self.skip_frames != 0:
                continue

            

            yield FrameElement(self.video_pth, frame_number, frame)



from typing import Generator, Iterable, List, Union, Optional


logger = logging.getLogger(__name__)

Element = Union[FrameElement, VideoEndBreakElement]
BatchOut = Union[List[FrameElement], VideoEndBreakElement]


class FrameBatcher:
    """
    Батчер кадров:
    - на вход: поток Element = FrameElement | VideoEndBreakElement
    - на выход: List[FrameElement] батчами и/или VideoEndBreakElement
    """

    def __init__(
        self,
        batch_size: int,
        *,
        yield_tail: bool = True,
        stop_on_end: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("FrameBatcher| batch_size должен быть > 0")

        self.batch_size = batch_size
        self.yield_tail = yield_tail

    def process(self, source: Iterable[Element]) -> Generator[BatchOut, None, None]:
        batch: List[FrameElement] = []

        for el in source:
            if isinstance(el, VideoEndBreakElement):
                # перед концом сначала отдадим хвост батча
                if batch and self.yield_tail:
                    yield batch
                    batch = []

                # затем сам end
                yield el

            batch.append(el)
            if len(batch) >= self.batch_size:
                yield batch
                batch = []


    def __call__(self, source: Iterable[Element]) -> Generator[BatchOut, None, None]:
        return self.process(source)

