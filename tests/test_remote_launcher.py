import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class RemoteLauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.log = self.root / "calls.jsonl"
        self.script = Path(__file__).resolve().parents[1] / "scripts/remote/train_remote.sh"
        stub = self.root / "remote_stub"
        stub.write_text("#!" + sys.executable + "\n" + '''import json, os, sys
from pathlib import Path
name = Path(sys.argv[0]).name
with open(os.environ["DEX_TEST_CALLS"], "a") as stream:
    stream.write(json.dumps([name, *sys.argv[1:]]) + "\\n")
if name == "rsync":
    sys.exit(0)
command = sys.argv[-1]
if "--query-gpu=" in command:
    if os.environ.get("DEX_TEST_FAILURE") == "gpu":
        sys.exit("GPU query failed")
    print("0, Fake GPU, 0 MiB, 24000 MiB, 0 %\\n2, Fake GPU, 0 MiB, 24000 MiB, 0 %")
elif command == "df -h /data_ssd":
    if os.environ.get("DEX_TEST_FAILURE") == "disk":
        sys.exit("disk query failed")
    print("synthetic disk status")
elif "pwd -P" in command or "printf" in command:
    print("/tmp/dexmani-synthetic-source")
''')
        stub.chmod(0o755)
        for name in ("ssh", "rsync"):
            (self.root / name).symlink_to(stub)

    def run_launcher(self, *arguments, failure=""):
        self.log.unlink(missing_ok=True)
        result = subprocess.run(
            ["/bin/bash", str(self.script), *arguments],
            env={**os.environ, "PATH": str(self.root) + os.pathsep + os.environ["PATH"],
                 "DEX_TEST_CALLS": str(self.log), "DEX_TEST_FAILURE": failure},
            text=True, capture_output=True, timeout=10,
        )
        calls = [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []
        return result, calls

    def test_single_gpu_query_preserves_arguments_and_validates_actual_indices(self):
        override = "+note=space ' quote $dollar; literal"
        result, calls = self.run_launcher("--fg", "--gpus", "2", "dp3", "pour", override)
        self.assertEqual(result.returncode, 0, result.stderr)
        queries = [call for call in calls if "--query-gpu=" in call[-1]]
        self.assertEqual(len(queries), 1)
        command = shlex.split(calls[-1][-1])[-1]
        arguments = shlex.split(command)
        self.assertIn("CUDA_VISIBLE_DEVICES=2", arguments)
        self.assertIn(override, arguments)
        self.assertIn("task_name=pour", arguments)
        result, calls = self.run_launcher("--gpus", "1", "dp3", "pour")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("GPU id 1 is unavailable", result.stderr)
        self.assertFalse(any("tmux -L" in call[-1] for call in calls))

    def test_resource_query_errors_prevent_launch(self):
        for stage in ("gpu", "disk"):
            with self.subTest(stage=stage):
                result, calls = self.run_launcher("--gpus", "0", "dp3", "pour", failure=stage)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("query failed", result.stderr)
                self.assertFalse(any("tmux -L" in call[-1] for call in calls))

    def test_uuid_failure_is_not_retried_or_hidden(self):
        python = self.root / "python3"
        python.write_text("#!/bin/sh\necho 'UUID generation failed' >&2\nexit 19\n")
        python.chmod(0o755)
        result, calls = self.run_launcher("dp3", "pour")
        self.assertEqual(result.returncode, 19)
        self.assertIn("UUID generation failed", result.stderr)
        self.assertEqual(calls, [])

    def test_dry_run_only_previews_sync_and_does_not_launch(self):
        result, calls = self.run_launcher("--dry-run", "--gpus", "0", "ddp/sat", "pour")
        self.assertEqual(result.returncode, 0, result.stderr)
        sync = next(call for call in calls if call[0] == "rsync")
        self.assertIn("--dry-run", sync)
        self.assertFalse(any(
            "mkdir" in call[-1] or "tmux -L" in call[-1] or "--query-gpu=" in call[-1]
            for call in calls
        ))


if __name__ == "__main__":
    unittest.main()
