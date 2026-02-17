import random
import cv2
import numpy as np
import json

from utils.utils import profile_time, FPS_Counter
from elements.VideoEndBreakElement import VideoEndBreakElement
from elements.FrameElement import FrameElement


class ShowNode:
    def __init__(self, config) -> None:
        self.scale = config["scale"]
        self.fps_counter_N_frames_stat = config["fps_counter_N_frames_stat"]
        self.default_fps_counter = FPS_Counter(self.fps_counter_N_frames_stat)
        self.draw_fps_info = config["draw_fps_info"]
        self.show_roi = config["show_roi"]
        self.show_only_yolo_detections = config["show_only_yolo_detections"]
        self.show_number_of_road = True
        self.imshow = config["imshow"]
        self.roads_info_path = config["roads_info"]
        with open(self.roads_info_path, "r", encoding="utf-8") as f:
                self.roads_info = json.load(f)
                
        data_colors = config["colors_of_roads"]
        self.colors_roads = {key: tuple(value) for key, value in data_colors.items()}

        # Шрифты
        self.fontFace = 1
        self.fontScale = 2.0
        self.thickness = 2
        # Линии
        self.thickness_lines = 3


    def _draw_label(self, img, x1, y1, text, color=(0, 0, 0)):
        (tw, th), baseline = cv2.getTextSize(text, self.fontFace, self.fontScale, self.thickness)
        y_top = max(0, y1 - th - baseline - 6)
        x_left = max(0, x1)
        cv2.rectangle(img, (x_left, y_top), (x_left + tw + 8, y_top + th + baseline + 6), color, -1)
        cv2.putText(
            img,
            text,
            (x_left + 4, y_top + th + 2),
            fontFace=self.fontFace,
            fontScale=self.fontScale,
            thickness=self.thickness,
            color=(255, 255, 255),
        )

    @profile_time
    def process(self, frame_element: FrameElement, fps_counter=None) -> FrameElement:
        if isinstance(frame_element, VideoEndBreakElement):
            return frame_element
        assert isinstance(frame_element, FrameElement), (
            f"ShowNode | Неправильный формат входного элемента {type(frame_element)}"
        )

        frame_result = frame_element.frame.copy()

        # Только YOLO-детекции (без трекинга)
        if self.show_only_yolo_detections:
            if frame_element.detected_xyxy and frame_element.detected_cls:
                for box, class_name in zip(frame_element.detected_xyxy, frame_element.detected_cls):
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame_result, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 2)
                    self._draw_label(frame_result, int(x1), int(y1), f"{class_name}", color=(0, 0, 0))


        # ROI дороги
        if self.show_roi:
            for road_id, points in self.roads_info.items():
                color = self.colors_roads[int(road_id)]
                points = np.array(points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame_result, [points], isClosed=True, color=color, thickness=self.thickness_lines)

                if self.show_number_of_road:
                    moments = cv2.moments(points)
                    if moments["m00"] != 0:
                        cx = int(moments["m10"] / moments["m00"])
                        cy = int(moments["m01"] / moments["m00"])
                        (lw, lh), _ = cv2.getTextSize(
                            str(road_id), self.fontFace, self.fontScale * 1.3, self.thickness
                        )
                        r = max(lw, lh) // 2
                        cv2.circle(frame_result, (cx, cy), r + 6, (200, 200, 200), -1)
                        cv2.putText(
                            frame_result,
                            str(road_id),
                            (cx + 2 - lw // 2, cy + 2 + lh // 2),
                            fontFace=self.fontFace,
                            fontScale=self.fontScale * 1.3,
                            thickness=self.thickness,
                            color=(0, 0, 0),
                        )

        # FPS
        if self.draw_fps_info:
            fps_counter = fps_counter if fps_counter is not None else self.default_fps_counter
            fps_real = fps_counter.calc_FPS()
            text = f"FPS: {fps_real:.1f}"
            (lw, lh), _ = cv2.getTextSize(text, self.fontFace, self.fontScale, self.thickness)
            cv2.rectangle(frame_result, (0, 0), (10 + lw, 35 + lh), (0, 0, 0), -1)
            cv2.putText(
                frame_result, text, (10, 40),
                fontFace=self.fontFace, fontScale=self.fontScale,
                thickness=self.thickness, color=(255, 255, 255)
            )

        frame_element.frame_result = frame_result
        frame_show = cv2.resize(frame_result.copy(), (-1, -1), fx=self.scale, fy=self.scale)

        if self.imshow:
            cv2.imshow(frame_element.source, frame_show)
            cv2.waitKey(1)

        return frame_element
