import subprocess
from pathlib import Path

import numpy as np
from imageio_ffmpeg import get_ffmpeg_exe


def save_episode_video(env, directory, name) -> str:
    frames = env.get_video()
    if frames is None or len(frames) == 0:
        raise ValueError(f"No video frames for {name}")

    path = Path(directory) / f"{name}.mp4"
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
        str(env.video_fps),
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
