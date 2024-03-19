import cv2 as cv
from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions, FaceDetectorResult, RunningMode
from mediapipe.tasks.python import BaseOptions
import mediapipe as mp
import numpy as np


class Face_Detector():
    def __init__(self, model_path: str, running_mode: RunningMode = RunningMode.IMAGE, min_detection_confidence: float = 0.5, min_suppression_threshold: float = 0.3):
        options = FaceDetectorOptions(base_options=BaseOptions(model_asset_path=model_path),
                                      running_mode=running_mode,
                                      min_detection_confidence=min_detection_confidence,
                                      min_suppression_threshold=min_suppression_threshold,
                                      result_callback=(self.result_callback if running_mode == RunningMode.LIVE_STREAM else None))
        self.detector = FaceDetector.create_from_options(options)
        self.result = None
        self.output_img = None
        self.time_stamp_ms = None

    def result_callback(self, result: FaceDetectorResult, output_image: mp.Image, time_stamp_ms: int):
        self.result = result
        self.output_img = output_image
        self.time_stamp_ms = time_stamp_ms

    def draw_bounding_box(self, img: np.ndarray, result: FaceDetectorResult):
        annotated_img = img.copy()
        w_img, h_img, _ = img.shape
        if result:
            for detection in result.detections:
                bounding_box = detection.bounding_box
                x, y, width, height = bounding_box.origin_x, bounding_box.origin_y, bounding_box.width, bounding_box.height
                cv.rectangle(annotated_img, (x, y),
                             (x+width, y+height), (0, 255, 0), 1)
                category = detection.categories[0]
                score = category.score
                cv.putText(annotated_img, str(int(score*100))+'%', (x-10, y-20),
                           cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 1)
        return annotated_img

    def detect(self, src: str, display: bool = True):
        img = mp.Image.create_from_file(src)
        self.result = self.detector.detect(img)
        if display:
            cv.imshow("Detect", img.numpy_view())
            if cv.waitKey(0) == ord('q'):
                cv.destroyAllWindows()

    def detect_video(self, src: str, display: bool = True, bounding_box: bool = True):
        time_stamp = 0
        cap = cv.VideoCapture(src)
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv.resize(frame, (960, 540))
            time_stamp += 1
            if not ret:
                break
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            self.result = self.detector.detect_for_video(img, time_stamp)
            img = img.numpy_view()
            if display:
                if bounding_box:
                    img = self.draw_bounding_box(img, self.result)
                cv.imshow("Detect", img)
                if cv.waitKey(1) == ord('q'):
                    break
        cap.release()
        cv.destroyAllWindows()

    def detect_async(self, display: bool = True, bounding_box: bool = True):
        time_stamp = 0
        cap = cv.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv.resize(frame, (960, 540))
            time_stamp += 1
            if not ret:
                break
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            self.detector.detect_async(img, time_stamp)
            img = img.numpy_view()
            if display:
                if bounding_box:
                    img = self.draw_bounding_box(img, self.result)
                cv.imshow("Detect", img)
                if cv.waitKey(1) == ord('q'):
                    break
        cap.release()
