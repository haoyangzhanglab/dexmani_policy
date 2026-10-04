import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf
from dexmani_policy.deployment import runtime
from dexmani_policy.training.run_identity import resolve_resume_source
from dexmani_policy.training.checkpoint import CheckpointStore
from dexmani_policy.smoke_test import load_config


class LaunchInfraTests(unittest.TestCase):
    def test_identity_across_launches(self):
        code='from dexmani_policy.utils.config import register_resolvers; from omegaconf import OmegaConf; register_resolvers(); print(OmegaConf.create({"x":"${run_id:}"}).x)'
        values=[subprocess.check_output([sys.executable,'-c',code],text=True).strip() for _ in range(2)]
        self.assertNotEqual(*values)
        for name in ('dp','dp3','sat','maniflow','ddp/dp'):
            cfg=load_config(name)
            self.assertNotIn('${',cfg.workspace.wandb_cfg.name)

    def test_variable_depth_inspection_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for selector in ('dp/t/run','ddp/dp/t/run'):
                run=root/selector; (run/'checkpoints').mkdir(parents=True)
                cfg=load_config('dp3'); OmegaConf.save(cfg,run/'config.yaml')
                (run/'checkpoints/latest.pt').write_bytes(b'fixture')
                with patch.object(runtime,'_EXPERIMENTS_ROOT',root):
                    self.assertEqual(runtime.resolve_experiment(selector),run)
                    self.assertEqual(runtime.inspect_policy(selector,checkpoint='latest').experiment_dir,run)
                self.assertEqual(resolve_resume_source(run),str(run/'checkpoints/latest.pt'))
                self.assertEqual(CheckpointStore(run/'checkpoints').resolve_path('latest'),run/'checkpoints/latest.pt')
            with patch.object(runtime,'_EXPERIMENTS_ROOT',root):
                self.assertEqual(len(runtime.list_experiments()),2)
                with self.assertRaises((ValueError,FileNotFoundError)): runtime.resolve_experiment('../outside')
            project=Path(__file__).resolve().parents[1]
            # Relative resume is anchored to the project even after cwd changes.
            relative=os.path.relpath(root/'dp/t/run',project)
            previous=Path.cwd()
            try:
                other_cwd=root/'other_cwd'; other_cwd.mkdir()
                os.chdir(other_cwd)
                self.assertEqual(resolve_resume_source(relative),str(root/'dp/t/run/checkpoints/latest.pt'))
            finally: os.chdir(previous)
            with tempfile.TemporaryDirectory(dir=project) as local_tmp:
                source=Path(local_tmp)/'source.pt'; source.write_bytes(b'fixture')
                real_expanduser = Path.expanduser
                def expand_fixture_user(path):
                    if path.parts and path.parts[0] == '~':
                        return Path(local_tmp).joinpath(*path.parts[1:])
                    return real_expanduser(path)
                with patch.object(Path, 'expanduser', expand_fixture_user):
                    self.assertEqual(resolve_resume_source('~/source.pt'),str(source))
                self.assertEqual(resolve_resume_source(source.relative_to(project)),str(source))
                self.assertEqual(resolve_resume_source(source),str(source))

    def test_remote_launch_uses_unique_sessions_and_preserves_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); bin_dir=root/'bin'; bin_dir.mkdir()
            ssh=bin_dir/'ssh'
            ssh.write_text('#!'+sys.executable+'\n'+r'''
import json,os,shlex,sys,subprocess,re
from pathlib import Path
root=Path(os.environ['FIXTURE_ROOT'])
command=sys.argv[-1]
with (root/'commands.jsonl').open('a') as f: f.write(json.dumps(command)+'\n')
if 'tmux new-session' not in command: sys.exit(0)
remote=shlex.split(command)[2]
parts=shlex.split(remote)
assert parts[:2]==['tmux','new-session'],parts
session=parts[parts.index('-s')+1]
assert re.fullmatch(r'dex_[a-zA-Z0-9_.-]+',session)
if os.environ.get('LAUNCH_FAIL'): sys.exit(17)
marker=root/session
if marker.exists(): sys.exit(1)
marker.write_text('live')
body=parts[-1].replace('$HOME/ZHY/dexmani_policy',str(root))
body=re.sub(r'\{ .*?; _rc=', '{ (exit '+os.environ.get('TRAIN_RC','0')+'); _rc=',body,count=1)
result=subprocess.run(['bash','-c',body])
with (root/'launches.jsonl').open('a') as f: f.write(json.dumps({'session':session,'body':body,'exit':result.returncode})+'\n')
sys.exit(0)
'''); ssh.chmod(0o755)
            # rsync is a no-op: remote sync/preflight must never reach a server.
            rsync=bin_dir/'rsync'; rsync.write_text('#!/bin/sh\nexit 0\n'); rsync.chmod(0o755)
            env=dict(os.environ,PATH=str(bin_dir)+':'+os.environ['PATH'],FIXTURE_ROOT=str(root),TRAIN_RC='7')
            # A client without the Linux UUID node must still support dry-run.
            cat=bin_dir/'cat'
            cat.write_text('#!/bin/sh\ncase "$*" in */proc/*) exit 91;; esac\nexec /bin/cat "$@"\n')
            cat.chmod(0o755)
            self.assertNotIn('/proc', Path('scripts/remote/train_remote.sh').read_text())
            dry=subprocess.run(['bash','scripts/remote/train_remote.sh','--dry-run','dp','a+b'],
                               env=env,check=True,capture_output=True,text=True)
            self.assertIn('=== DRY RUN ===',dry.stdout)
            self.assertFalse((root/'commands.jsonl').exists())
            command=['bash','scripts/remote/train_remote.sh','dp','a+b']
            for _ in range(2): subprocess.run(command,env=env,check=True,capture_output=True,text=True)
            launches=[json.loads(line) for line in (root/'launches.jsonl').read_text().splitlines()]
            self.assertEqual(len(launches),2); self.assertNotEqual(launches[0]['session'],launches[1]['session'])
            for item in launches:
                self.assertEqual(item['exit'],7)
                log=root/'logs'/(item['session']+'.log'); before=log.read_bytes()
                self.assertIn(b'exit 7',before)
                # Reusing the exact generated body refuses to truncate the log.
                rerun=subprocess.run(['bash','-c',item['body']],capture_output=True)
                self.assertNotEqual(rerun.returncode,0); self.assertEqual(log.read_bytes(),before)
            self.assertNotIn('kill-session',(root/'commands.jsonl').read_text())
            result=subprocess.run(command,env=dict(env,LAUNCH_FAIL='1'),capture_output=True)
            self.assertEqual(result.returncode,17)

    def test_remote_local_python_fallback_and_error(self):
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = Path(tmp)
            # An isolated PATH models a client with no interpreter candidates.
            for name in ('bash', 'dirname', 'date', 'basename'):
                (bin_dir / name).symlink_to(shutil.which(name))
            rsync = bin_dir / 'rsync'
            rsync.write_text('#!/bin/sh\nexit 0\n'); rsync.chmod(0o755)
            env = dict(os.environ, PATH=str(bin_dir))
            command = [str(bin_dir/'bash'), 'scripts/remote/train_remote.sh', '--dry-run', 'dp', 't']
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('working local python3 or python', result.stderr)
            broken = bin_dir/'python3'
            broken.write_text('#!/bin/sh\nexit 23\n'); broken.chmod(0o755)
            (bin_dir/'python').symlink_to(sys.executable)
            result = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
            self.assertIn('=== DRY RUN ===', result.stdout)
            self.assertRegex(result.stdout, r'Session: +dex_dp_t_.*_[0-9a-f]{32}')


if __name__=='__main__': unittest.main()
