#!/usr/bin/env python3

"""
The launch point module of the app.
"""


import sys

from pathlib import Path

from cv import process_video
from utils import parse_args


def main() -> None:
    """
    Entry point for the video people detection script.

    Parses command-line arguments, validates the input video file, and
    invokes the video processing function with the specified parameters.

    :Parameters:
        None

    :Returns:
        None: The function executes the video processing pipeline and
              handles output video creation.

    :Exceptions:
        SystemExit: Raised if the input video file does not exist.
        Exception: Raised if an unexpected error occurs during video processing.
    """

    args = parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    try:
        process_video(
            str(input_path),
            args.output,
            args.model,
            args.conf,
            args.device,
            args.show
        )
    except Exception as err:
        print(
            f"\nError file: {__file__}" +
            f"\nError message: {err}",
        )


if __name__ == "__main__":
    main()
