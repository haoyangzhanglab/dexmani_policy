"""Best-effort episode video export, independent of evaluation metrics."""

import subprocess
import tempfile
from contextlib import suppress
from pathlib import Path

import numpy as np
from imageio_ffmpeg import get_ffmpeg_exe
from termcolor import cprint


def encode_video(path: Path, frames: np.ndarray, fps: int) -> None:
    """Stream RGB frames to ffmpeg and propagate encoding failures."""
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
    # A file prevents stderr backpressure from blocking the stdin writer.
    with tempfile.TemporaryFile() as stderr:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=stderr)
        try:
            for frame in frames:
                proc.stdin.write(memoryview(np.ascontiguousarray(frame, dtype=np.uint8)))
            proc.stdin.close()
            returncode = proc.wait(timeout=60)
            if returncode != 0:
                stderr.seek(0, 2)
                stderr.seek(max(0, stderr.tell() - 4096))
                detail = stderr.read().decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"ffmpeg exited with status {returncode}: {detail}")
        finally:
            # Reap the encoder even after a broken pipe or timeout.
            if proc.poll() is None:
                with suppress(Exception):
                    proc.stdin.close()
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


def save_episode_video(env, directory, name, fps):
    if directory is None:
        return None
    try:
        frames = env.get_video()
        if frames is None:
            return None
        path = Path(directory) / f"{name}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        encode_video(path, frames, fps)
        return {name: str(path)}
    except Exception as exc:
        cprint(f"Video export failed for {name}: {exc}", "yellow")
        return None
