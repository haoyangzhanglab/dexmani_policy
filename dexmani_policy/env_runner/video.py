import subprocess
from pathlib import Path

import numpy as np


def save_episode_video(frames, fps, path) -> str:
    from imageio_ffmpeg import get_ffmpeg_exe
    if frames is None or len(frames) == 0:
        raise ValueError(f"No video frames for {path}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _, height, width, _ = frames.shape
    cmd = [
        get_ffmpeg_exe(),
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "-",
        "-pix_fmt",
        "yuv420p",
        "-vcodec",
        "libx264",
        "-crf",
        "23",
        str(path),
    ]
    frames = np.ascontiguousarray(frames, dtype=np.uint8)
    subprocess.run(cmd, input=frames.data.cast("B"), check=True)
    return str(path)
