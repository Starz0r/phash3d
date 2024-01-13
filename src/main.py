import argparse
import math
import subprocess
import sys
from queue import Empty, Queue
from typing import Any, Final, List

import cv2
import imagehash
import structlog
from cv2.typing import MatLike
from PIL import Image

argparser: Final[argparse.ArgumentParser] = argparse.ArgumentParser()
argparser.add_argument("--file", action="store", required=True, type=str)

ARGS: Final[argparse.Namespace] = argparser.parse_args()
LOGGER: Final[structlog.stdlib.BoundLogger] = structlog.getLogger()


def get_video_frame_rate(path: str) -> float:
    cmd = subprocess.Popen(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-select_streams",
            "0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "csv=p=0:s=x",
            "-i",
            f"file:{path}",
        ],
        stdout=subprocess.PIPE,
    )
    cmd.wait()
    if cmd.stdout is None:
        raise Exception("Process did not have a StdOut buffer.")
    nums = cmd.stdout.read().decode("utf-8").split("/")
    return int(nums[0]) / int(nums[1].rstrip("x\r\n"))


def get_video_length(path: str) -> float:
    cmd = subprocess.Popen(
        [
            "ffprobe",
            "-i",
            f"file:{path}",
            "-show_entries",
            "format=duration",
            "-v",
            "quiet",
            "-of",
            "csv=%s" % ("p=0"),
        ],
        stdout=subprocess.PIPE,
    )
    cmd.wait()
    if cmd.stdout is None:
        raise Exception("Process did not have a StdOut buffer.")
    return float(cmd.stdout.read().decode("utf-8"))


def matlike_to_pimg(img: MatLike) -> Image:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


def concat_img(img1: Image, img2: Image, color=(0, 0, 0)) -> Image:
    dst = Image.new("RGB",
                    (img1.width + img2.width, max(img1.height, img2.height)),
                    color)
    dst.paste(img1, (0, 0))
    dst.paste(img2, (img1.width, 0))
    return dst


def concat_multi_img(imgs: List[Any]) -> Image:
    base = imgs.pop(0)
    for img in imgs:
        base = concat_img(base, img)
    return base


def main(argv: argparse.Namespace) -> int:
    cap = cv2.VideoCapture(argv.file)
    if not cap.isOpened():
        LOGGER.fatal("opencv could not open file")
        return 1

    frames: Queue[Image] = Queue()
    while True:
        ndone, frame = cap.read()
        if ndone:
            img = matlike_to_pimg(frame)
            frames.put(img)
        else:
            break

    steps = 8
    video_length = get_video_length(argv.file)
    frame_rate = get_video_frame_rate(argv.file)
    frames_per_step = math.floor(frame_rate / steps)
    hashes: List[str] = []
    current_length: float = 0.0
    while current_length < video_length:
        working_frames: List[Any] = []
        for _ in range(frames_per_step):
            try:
                working_frames.append(frames.get(block=False))
            except Empty as e:
                #LOGGER.info("worked on all available frames", err=e)
                break
        megaframe = concat_multi_img(working_frames)
        phash = imagehash.phash(megaframe, hash_size=8, highfreq_factor=16)
        hashes.append(phash.__str__())
        current_length += 1 / steps

    for hash in hashes:
        print(hash.upper(), end="", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main(ARGS))
