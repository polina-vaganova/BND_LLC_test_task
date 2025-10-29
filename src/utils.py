#!/usr/bin/env python3

"""
Video People Detection Command-Line Tool.

:Command-Line Arguments:
    --input / -i     : Path to input video file (required).
    --output / -o    : Path to output annotated video (default: "output.mp4").
    --model / -m     : YOLOv8 model weights path (default: "yolov8n.pt").
    --conf / -c      : Confidence threshold for detections (0..1, default: 0.25).
    --device / -d    : Device for inference, e.g., "cpu" or GPU index (default: "cpu").
    --show           : Flag to display frames during processing (optional).

:Usage Example:
    >>> python detect_people.py --input crowd.mp4 --output annotated.mp4 \
        --model yolov8n.pt --conf 0.5 --device 0 --show
"""


import sys
import argparse

from pathlib import Path


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for video object detection.

    :Parameters:
        None

    :Returns:
        argparse.Namespace: Parsed command-line arguments containing:
            - input (str): Path to the input video file.
            - output (str): Path to the output annotated video.
            - model (str): Path to YOLOv8 model weights.
            - conf (float): Confidence threshold for detections.
            - device (str): Device for inference ("cpu" or GPU index).
            - show (bool): Whether to display processed frames during execution.

    :Exceptions:
        SystemExit: Raised automatically by argparse on invalid arguments or
            when displaying help.
    """

    parser = argparse.ArgumentParser(
        description="Detect people in a video and save annotated output."
    )

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to input video (e.g., crowd.mp4)."
    )
    parser.add_argument(
        "--output",
        "-o",
        default="output.mp4",
        help="Path to output annotated video."
    )
    parser.add_argument(
        "--model",
        "-m",
        default="yolov8n.pt",
        help="YOLOv8 weights (will be downloaded if absent)."
    )
    parser.add_argument(
        "--conf",
        "-c",
        type=float,
        default=0.25,
        help="Confidence threshold (0..1)."
    )
    parser.add_argument(
        "--device",
        "-d",
        default="cpu",
        help="Device for inference, e.g., 'cpu' or '0' (GPU)."
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show progress window while processing (slower)."
    )

    return parser.parse_args()
