"""Local rsync checks: python -m dexmani_policy.test_tools.remote_checks."""

import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path


def check_sync_down():
    script = Path(__file__).resolve().parents[2] / "scripts/remote/sync_down.sh"
    rsync = shutil.which("rsync")
    if rsync is None:
        raise RuntimeError("These checks require local rsync")
    with tempfile.TemporaryDirectory(prefix="dexmani-sync-check-") as temporary:
        root = Path(temporary)
        remote, local = root / "remote run", root / "local run"
        (remote / "checkpoints").mkdir(parents=True)
        (remote / "wandb").mkdir()
        (remote / "eval/evaluation").mkdir(parents=True)
        (remote / "config.yaml").write_text("remote: true\n")
        (remote / "notes.txt").write_text("first\n")
        (remote / "metrics.jsonl").write_text('{"step": 1}\n')
        (remote / "checkpoints/step_00000001.pt").write_bytes(b"first checkpoint")
        (remote / "checkpoints/latest.pt").symlink_to("step_00000001.pt")
        external = root / "external.pt"
        external.write_bytes(b"outside the run")
        (remote / "checkpoints/external.pt").symlink_to(external)
        (remote / "eval/evaluation/result_details.json").write_text('{"step": 1}\n')
        (remote / "wandb/offline.log").write_text("wandb log\n")
        for parent in (remote, remote / "checkpoints", remote / "eval/evaluation"):
            (parent / ".dexmani-publish-incomplete.tmp").write_bytes(b"incomplete")

        # Replace only SSH transport; file transfer still runs the real local rsync.
        bin_dir = root / "bin"
        bin_dir.mkdir()
        wrapper = bin_dir / "rsync"
        wrapper.write_text('''#!/usr/bin/env python3
import os
import sys

args = sys.argv[1:]
assert args[-2].startswith("local:")
args[-2] = args[-2].removeprefix("local:")
os.execv(os.environ["SYNC_CHECK_RSYNC"], ["rsync", *args])
''')
        wrapper.chmod(0o755)
        env = dict(os.environ, PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
                   SYNC_CHECK_RSYNC=rsync)

        def sync(*options, destination=local):
            result = subprocess.run(
                ["bash", str(script), "--host", "local", "--remote-run", str(remote),
                 "--local-run", str(destination), *options],
                env=env, capture_output=True, text=True,
            )
            assert result.returncode == 0, result.stdout + result.stderr

        def snapshot():
            return {str(path.relative_to(local)): (
                os.readlink(path) if path.is_symlink() else None if path.is_dir() else path.read_bytes()
            ) for path in local.rglob("*")}

        absent = root / "absent parent/run"
        sync("--dry-run", destination=absent)
        assert not absent.parent.exists()
        sync()
        for path in ("config.yaml", "notes.txt", "metrics.jsonl", "checkpoints/step_00000001.pt",
                     "eval/evaluation/result_details.json"):
            assert (local / path).read_bytes() == (remote / path).read_bytes()
        assert (local / "checkpoints/latest.pt").is_symlink()
        assert os.readlink(local / "checkpoints/latest.pt") == "step_00000001.pt"
        assert (local / "checkpoints/external.pt").is_symlink()
        assert os.readlink(local / "checkpoints/external.pt") == str(external)
        assert not list(local.rglob(".dexmani-publish-*"))
        assert not (local / "wandb").exists()

        (local / "config.yaml").write_text("different local config\n")
        (local / "local-only.txt").write_text("keep\n")
        (remote / "notes.txt").write_text("again\n")
        old_time = (local / "notes.txt").stat().st_mtime
        os.utime(remote / "notes.txt", (old_time + 5, old_time + 5))
        (remote / "metrics.jsonl").write_text('{"step": 200}\n')
        (remote / "checkpoints/step_00000002.pt").write_bytes(b"second checkpoint")
        (remote / "checkpoints/latest.pt").unlink()
        (remote / "checkpoints/latest.pt").symlink_to("step_00000002.pt")
        before = snapshot()
        sync("--dry-run", "--with-wandb")
        assert snapshot() == before

        sync()
        assert (local / "config.yaml").read_bytes() == (remote / "config.yaml").read_bytes()
        assert (local / "notes.txt").read_text() == "again\n"
        assert (local / "metrics.jsonl").read_text() == '{"step": 200}\n'
        assert os.readlink(local / "checkpoints/latest.pt") == "step_00000002.pt"
        assert (local / "checkpoints/latest.pt").read_bytes() == b"second checkpoint"
        assert (local / "local-only.txt").read_text() == "keep\n"
        assert not (local / "wandb").exists()
        sync("--with-wandb")
        assert (local / "wandb/offline.log").read_text() == "wandb log\n"
        assert not list(local.rglob(".dexmani-publish-*"))


def check_launcher():
    script = Path(__file__).resolve().parents[2] / "scripts/remote/train_remote.sh"
    with tempfile.TemporaryDirectory(prefix="dexmani-launch-check-") as temporary:
        root = Path(temporary)
        run = root / "experiments/train/dp3/task with spaces/2026-11-03_14-30-00_seed42"
        base = ["bash", str(script), "--host", "unused", "--repo", str(root / "repo"),
                "--data-root", str(root / "data-root"), "--run", str(run),
                "--python", "/opt/policy/bin/python", "--gpu", "0", "--dry-run"]
        for options in (["--", "dp3", "task_name=task", "workspace.output_dir=/wrong/run", "hydra.run.dir=/wrong/run"],
                        ["--resume", "--", "--max-updates", "1"]):
            result = subprocess.run([*base, *options], capture_output=True, text=True, check=True)
            command = shlex.split(result.stdout.split("Command: ", 1)[1])
            if "--resume" in options:
                assert command[3] == "dexmani_policy.training.common.resume"
                assert command[command.index("--run") + 1] == str(run)
            else:
                assert command[3] == "dexmani_policy.train"
                overrides = dict(value.split("=", 1) for value in command[5:])
                assert overrides["workspace.output_dir"] == overrides["hydra.run.dir"] == f'"{run}"'
        assert not list(root.iterdir())


if __name__ == "__main__":
    check_launcher()
    print("PASS launcher fresh/resume output binding and dry-run")
    check_sync_down()
    print("PASS sync_down first/incremental download, filters, symlinks and dry-run")
