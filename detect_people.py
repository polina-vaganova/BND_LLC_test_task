#!/usr/bin/env python3
"""
detect_people.py

Точка входа для детекции людей в видео. Загружает предобученную модель YOLOv8 (ultralytics),
делает инференс по кадрам, рисует рамки с подписью "person" и confidence, сохраняет видео.

Пример:
    python detect_people.py --input crowd.mp4 --output crowd_output.mp4 --model yolov8n.pt --conf 0.35

Требования:
    - Python 3.8+
    - ultralytics
    - opencv-python
    - numpy
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(description="Detect people in a video and save annotated output.")
    parser.add_argument("--input", "-i", required=True, help="Path to input video (e.g., crowd.mp4).")
    parser.add_argument("--output", "-o", default="output.mp4", help="Path to output annotated video.")
    parser.add_argument("--model", "-m", default="yolov8n.pt", help="YOLOv8 weights (will be downloaded if absent).")
    parser.add_argument("--conf", "-c", type=float, default=0.25, help="Confidence threshold (0..1).")
    parser.add_argument("--device", "-d", default="cpu", help="Device for inference, e.g., 'cpu' or '0' (GPU).")
    parser.add_argument("--show", action="store_true", help="Show progress window while processing (slower).")
    return parser.parse_args()


def init_video_io(input_path: str, output_path: str) -> Tuple[cv2.VideoCapture, cv2.VideoWriter]:
    """
    Инициализирует VideoCapture и VideoWriter.

    Возвращает (cap, writer).
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open output video for writing: {output_path}")

    return cap, writer


def draw_boxes(frame: np.ndarray, boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """
    Нарисовать рамки на кадре.

    Args:
        frame: BGR изображение.
        boxes: массив [N,4] с координатами в формате xyxy.
        scores: массив [N] с confidence.
        classes: массив [N] с id классов (int).

    Returns:
        Annotated frame (in-place модификация).
    """
    for (x1, y1, x2, y2), score, cls in zip(boxes.astype(int), scores, classes.astype(int)):
        # только person (COCO class id == 0)
        if cls != 0:
            continue
        label = f"person {score:.2f}"
        # draw rectangle (thin, semi-transparent-like by overlay)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 200, 0), thickness=2)
        # put background for text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), (0, 200, 0), -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def process_video(input_path: str, output_path: str, model_weights: str, conf_thres: float, device: str, show: bool = False) -> None:
    """
    Основная функция: читает видео, прогоняет модель, рисует и сохраняет видео.
    """
    cap, writer = init_video_io(input_path, output_path)

    # load model
    model = YOLO(model_weights)
    model.fuse()  # optional: ускорение при cpu

    frame_idx = 0
    t0 = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
                        # ultralytics принимает BGR np.ndarray
            # run inference (single image)
            results = model.predict(frame, device=device, imgsz=640, conf=conf_thres, max_det=300, verbose=False)
            # results is a list-like; take first result
            if len(results) == 0:
                writer.write(frame)
                if show:
                    cv2.imshow("out", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue

            r = results[0]
            # boxes: xyxy numpy, scores, classes
            boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes, "xyxy") else np.empty((0, 4))
            scores = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, "conf") else np.empty((0,))
            classes = r.boxes.cls.cpu().numpy() if hasattr(r.boxes, "cls") else np.empty((0,))

            annotated = draw_boxes(frame, boxes, scores, classes)
            writer.write(annotated)

            if show:
                cv2.imshow("out", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # simple progress to stderr
            if frame_idx % 50 == 0:
                elapsed = time.time() - t0
                print(f"[INFO] frames={frame_idx} elapsed={elapsed:.1f}s", file=sys.stderr)
    finally:
        cap.release()
        writer.release()
        if show:
            cv2.destroyAllWindows()

    total_time = time.time() - t0
    print(f"[INFO] Done. Frames processed: {frame_idx}. Time: {total_time:.2f}s. Output: {output_path}")


def main() -> None:
    """Точка входа скрипта."""
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    try:
        process_video(str(input_path), args.output, args.model, args.conf, args.device, args.show)
    except Exception as e:
        print(f"Error during processing: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()