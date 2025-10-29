#!/usr/bin/env python3
"""
Video Object Detection and Annotation Module.

:Example:
    >>> process_video(
    ...     input_path="input.mp4",
    ...     output_path="output.mp4",
    ...     model_weights="yolov8n.pt",
    ...     conf_thres=0.5,
    ...     device="cuda:0",
    ...     show=True
    ... )
"""

from __future__ import annotations
import sys
import cv2
import time

import numpy as np

from pathlib import Path
from typing import Tuple
from ultralytics import YOLO


def init_video_io(
    input_path: str,
    output_path: str
) -> Tuple[cv2.VideoCapture, cv2.VideoWriter] | None:
    """
    Initialize video input and output streams.

    :Parameters:
        input_path (str): Path to the input video file.
        output_path (str): Path to the output video file to be created.

    :Returns:
        Tuple[cv2.VideoCapture, cv2.VideoWriter]: A tuple containing initialized
            OpenCV video reader and writer objects.
        None: If an error occurs during initialization.

    :Exceptions:
        RuntimeError: Raised if the input video cannot be opened or the output
            video cannot be created.
    """

    try:
        cap = cv2.VideoCapture(input_path, )

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open input video: {input_path}", )

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH, ), )
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT, ), )
        fps = cap.get(cv2.CAP_PROP_FPS, ) or 25.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height, ),
        )

        if not writer.isOpened():
            cap.release()
            raise RuntimeError(
                f"Cannot open output video for writing: {output_path}",
            )

        return cap, writer
    except Exception as err:
        print(
            f"\nError file: {__file__}" +
            f"\nError message: {err}",
        )


def draw_boxes(
    frame: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray
) -> np.ndarray | None:
    """
    Draw bounding boxes with labels on a video frame.

    :Parameters:
        frame (np.ndarray): Input video frame as a NumPy array (BGR format).
        boxes (np.ndarray): Array of bounding boxes with coordinates
            (x1, y1, x2, y2) for each detected object.
        scores (np.ndarray): Confidence scores corresponding to each box.
        classes (np.ndarray): Class indices for each detection.

    :Returns:
        np.ndarray: The frame with drawn bounding boxes and labels.
        None: If an error occurs during drawing.

    :Exceptions:
        Exception: Raised if an unexpected error occurs while processing
            or drawing boxes.
    """

    try:
        for (x1, y1, x2, y2), score, cls in zip(
            boxes.astype(int),
            scores,
            classes.astype(int)
        ):
            if cls != 0:
                continue
          
            label = f"person {score:.2f}"
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color=(0, 200, 0),
                thickness=2
            )
            (tw, th), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                1
            )
            cv2.rectangle(
                frame,
                (x1, y1 - th - 6),
                (x1 + tw + 6, y1),
                (0, 200, 0),
                1
            )
            cv2.putText(
                frame,
                label,
                (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )
        return frame
    except Exception as err:
        print(
            f"\nError file: {__file__}" +
            f"\nError message: {err}",
        )


def process_video(
    input_path: str,
    output_path: str,
    model_weights: str,
    conf_thres: float,
    device: str,
    show: bool = False
) -> None:
    """
    Process an input video using a YOLO model for object detection.

    :Parameters:
        input_path (str): Path to the input video file.
        output_path (str): Path where the processed video will be saved.
        model_weights (str): Path to the YOLO model weights file.
        conf_thres (float): Confidence threshold for filtering detections.
        device (str): Device identifier for model inference (e.g., "cpu" or "cuda:0").
        show (bool): Whether to display processed frames during execution.
                     Default: False.

    :Returns:
        None: The function saves the output video to the specified path.

    :Exceptions:
        RuntimeError: Raised if video input or output initialization fails.
        Exception: Raised if an unexpected error occurs during video processing.
    """

    cap, writer = init_video_io(input_path, output_path, )
    model = YOLO(model_weights, )

    model.fuse()

    frame_idx = 0
    t0 = time.time()

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame_idx += 1
            results = model.predict(
                frame,
                device=device,
                imgsz=640,
                conf=conf_thres,
                max_det=300,
                verbose=False
            )

            if len(results) == 0:
                writer.write(frame)

                if show:
                    cv2.imshow("out", frame)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                continue

            r = results[0]
            boxes = r.boxes.xyxy.cpu().numpy()if hasattr(r.boxes, "xyxy") else np.empty((0, 4))
            scores = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, "conf") else np.empty((0,))
            classes = r.boxes.cls.cpu().numpy() if hasattr(r.boxes, "cls") else np.empty((0,))
            annotated = draw_boxes(
                frame,
                boxes,
                scores,
                classes,
            )
            writer.write(annotated, )

            if show:
                cv2.imshow("out", annotated)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if frame_idx % 50 == 0:
                elapsed = time.time() - t0

                print(
                    f"[INFO] frames={frame_idx} elapsed={elapsed:.1f}s",
                    file=sys.stderr
                )
    except Exception as err:
        print(
            f"\nError file: {__file__}" +
            f"\nError message: {err}",
        )
    finally:
        cap.release()
        writer.release()
        
        if show:
            cv2.destroyAllWindows()

    total_time = time.time() - t0
    print(
        f"[INFO] Done. Frames processed: {frame_idx}. Time: {total_time:.2f}s. Output: {output_path}"
    )

