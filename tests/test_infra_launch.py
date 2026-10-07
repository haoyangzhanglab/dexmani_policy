import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from yaml import YAMLError
from dexmani_policy.deployment import runtime
from dexmani_policy.training.run_identity import resolve_resume_source
from dexmani_policy.training.checkpoint import CheckpointStore
from dexmani_policy.smoke_test import load_config
from dexmani_policy.training.resume import resolve_training_config
from dexmani_policy.utils.config import resolve_input_recipe
from scripts.remote import resolve_remote_datasets as remote_datasets


class RemoteResumeRecipeTests(unittest.TestCase):
    def setUp(self):
        self.project = Path(__file__).resolve().parents[1]
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        # Exercise the actual compose entry with a temporary primary config.
        configs = tempfile.TemporaryDirectory(prefix='_remote_test_', dir=self.project/'dexmani_policy/configs')
        self.addCleanup(configs.cleanup)
        self.config_file = Path(configs.name)/'recipe.yaml'
        self.config_name = self.config_file.parent.name + '/recipe'
        self.source = self.root/'source'
        (self.source/'checkpoints').mkdir(parents=True)
        self.checkpoint = self.source/'checkpoints/latest.pt'
        self.checkpoint.write_bytes(b'deliberately not a deserializable checkpoint')
        self.saved_data = self.root/'saved_data'; self.saved_data.mkdir()
        self.default_data = self.root/'missing_default'
        self.incoming = OmegaConf.create({
            'task_name': 'current_default', 'dataset': {'zarr_path': str(self.default_data)},
            'agent': {'architecture': 'current'},
            'training': {'seed': 999, 'num_gpus': 2, 'use_compile': True,
                         'loop': {'total_train_steps': 100, 'log_interval_steps': 10}},
            'dataloader': {'batch_size': 64, 'num_workers': 8},
            'workspace': {'output_dir': '${hydra:runtime.output_dir}', 'wandb_cfg': {'mode': 'offline'}},
        })
        self.save_incoming()
        self.saved = OmegaConf.create(OmegaConf.to_container(self.incoming, resolve=False))
        self.saved.task_name = 'saved_task'
        self.saved.dataset.zarr_path = str(self.saved_data)
        self.saved.agent.architecture = 'saved'
        self.saved.training.seed = 12
        self.saved.workspace.output_dir = str(self.source)
        self.saved.workspace.claim_token = 'previous-claim'
        self.saved.max_updates = 2
        self.save_source()

    def save_source(self):
        OmegaConf.save(self.saved, self.source/'config.yaml')

    def save_incoming(self):
        self.config_file.write_text('# @package _global_\n' + OmegaConf.to_yaml(self.incoming))

    def compose(self, overrides):
        with initialize_config_dir(version_base=None, config_dir=str(self.project/'dexmani_policy/configs')):
            return compose(config_name=self.config_name, overrides=overrides)

    def resume_overrides(self, *extra, checkpoint=False):
        return ['+resume_from=' + str(self.checkpoint if checkpoint else self.source), *extra]

    def check(self, overrides):
        # main() is the real --check entry; only argv is supplied by the harness.
        with patch.object(sys, 'argv', ['resolve_remote_datasets.py', '--check',
                                       '--config-name='+self.config_name, *overrides]):
            remote_datasets.main()

    def test_task_assertions_and_recipe_protection(self):
        for task in ('saved_task', 'b+a'):
            self.saved.task_name = task; self.save_source()
            for explicit in ([], ['task_name='+task], ['task_name='+task, 'task_name='+task]):
                with self.subTest(task=task, explicit=explicit):
                    overrides = self.resume_overrides(*explicit)
                    cfg = self.compose(overrides)
                    before = OmegaConf.to_container(cfg, resolve=False)
                    actual = resolve_input_recipe(cfg, overrides=overrides)
                    self.assertEqual(actual.task_name, task)
                    self.assertEqual(actual.agent.architecture, 'saved')
                    self.assertEqual(actual.training.seed, 12)
                    self.assertEqual(actual.dataset.zarr_path, str(self.saved_data))
                    self.assertEqual(OmegaConf.to_container(cfg, resolve=False), before)
            for invalid in ('other', 'a+b', 'true', '12', 'null', '[b,a]', '{a:b}', '"bad/task"'):
                with self.subTest(invalid=invalid):
                    overrides = self.resume_overrides('task_name='+invalid)
                    with self.assertRaisesRegex(ValueError, 'requested=.*saved='):
                        resolve_input_recipe(self.compose(overrides), overrides=overrides)
            for tasks in (['task_name=wrong', 'task_name='+task], ['task_name='+task, 'task_name=wrong']):
                overrides = self.resume_overrides(*tasks)
                with self.assertRaisesRegex(ValueError, 'requested=.*saved='):
                    resolve_input_recipe(self.compose(overrides), overrides=overrides)
        del self.saved.task_name; self.save_source()
        overrides = self.resume_overrides('task_name=saved_task')
        with self.assertRaisesRegex(ValueError, 'requested=.*saved=None'):
            resolve_input_recipe(self.compose(overrides), overrides=overrides)
        for forbidden in ('agent.architecture=other', 'training.seed=12', 'dataloader.batch_size=64',
                          'training.num_gpus=2', 'training.loop.total_train_steps=100'):
            overrides = self.resume_overrides(forbidden)
            with self.subTest(forbidden=forbidden), self.assertRaisesRegex(ValueError, 'forbidden override'):
                resolve_input_recipe(self.compose(overrides), overrides=overrides)
        cfg = self.compose(self.resume_overrides())
        for unsupported in ('~task_name', 'task_name=a,b', '~dataloader.num_workers'):
            with self.subTest(unsupported=unsupported), self.assertRaisesRegex(ValueError, 'Unsupported resume override'):
                resolve_input_recipe(cfg, overrides=[unsupported])

    def test_preflight_saved_paths_and_training_agree(self):
        for checkpoint in (False, True):
            overrides = self.resume_overrides('task_name=saved_task', checkpoint=checkpoint)
            self.check(overrides)  # saved exists; current default does not
            self.assertEqual(remote_datasets.resolve_dataset_paths(self.config_name, overrides), [self.saved_data])
            cfg = self.compose(overrides)
            cfg.workspace.output_dir = str(self.root/'destination')
            actual = resolve_training_config(cfg, overrides=overrides)
            self.assertEqual(Path(actual.dataset.zarr_path), self.saved_data)
            self.assertEqual(actual.workspace.output_dir, str(self.root/'destination'))
            self.assertNotIn('claim_token', actual.workspace)
            self.assertNotIn('max_updates', actual)
        self.default_data.mkdir()
        self.saved_data.rmdir()
        with self.assertRaisesRegex(SystemExit, str(self.saved_data)):
            self.check(overrides)  # current exists; saved does not
        migrated = self.root/'migrated'; migrated.mkdir()
        overrides += ['dataset.zarr_path='+str(migrated), '+max_updates=4', 'dataloader.num_workers=0']
        self.check(overrides)
        cfg = self.compose(overrides); cfg.workspace.output_dir = str(self.root/'destination')
        actual = resolve_training_config(cfg, overrides=overrides)
        self.assertEqual(actual.max_updates, 4)
        self.assertEqual(actual.dataloader.num_workers, 0)
        self.assertEqual(remote_datasets.resolve_dataset_paths(self.config_name, overrides), [Path(actual.dataset.zarr_path)])
        # Fresh recipes keep current defaults and overrides, without a runtime lookup.
        for overrides in ([], ['dataset.zarr_path='+str(migrated)]):
            cfg = self.compose(overrides)
            with patch.object(HydraConfig, 'get', side_effect=AssertionError('unexpected runtime lookup')):
                actual = resolve_training_config(cfg, overrides=overrides)
                self.check(overrides)
            self.assertEqual(OmegaConf.to_container(actual, resolve=False), OmegaConf.to_container(cfg, resolve=False))

    def test_multitask_saved_children_and_migration(self):
        child = self.root/'second'; child.mkdir()
        migrated = self.root/'migrated'; migrated.mkdir()
        self.saved.task_name = 'b+a'
        self.saved.dataset = {'datasets': [{'zarr_path': str(child)}, {'zarr_path': str(self.saved_data)}]}
        self.save_source()
        # Current children differ; saved order and paths remain authoritative.
        self.incoming.dataset = {'datasets': [{'zarr_path': str(self.default_data)}, {'zarr_path': '/absent'}]}
        self.save_incoming()
        for extra, expected in (([], [child, self.saved_data]),
                                (['dataset.datasets.1.zarr_path='+str(migrated)], [child, migrated])):
            overrides = self.resume_overrides('task_name=b+a', *extra)
            self.check(overrides)
            cfg = self.compose(overrides); cfg.workspace.output_dir = str(self.root/'destination')
            actual = resolve_training_config(cfg, overrides=overrides)
            self.assertEqual([Path(c.zarr_path) for c in actual.dataset.datasets], expected)
            self.assertEqual(remote_datasets.resolve_dataset_paths(self.config_name, overrides), expected)

    def test_no_runtime_imports_weights_or_source_mutation(self):
        overrides = self.resume_overrides('task_name=saved_task')
        before = {p.relative_to(self.root): p.read_bytes() if p.is_file() else None for p in self.root.rglob('*')}
        cfg = self.compose(overrides)
        self.assertEqual(OmegaConf.to_container(cfg, resolve=False)['workspace']['output_dir'], '${hydra:runtime.output_dir}')
        self.assertFalse(HydraConfig.initialized())
        # A clean interpreter rejects runtime dependencies even during import.
        code = r'''
import importlib.abc, sys
class NoTrainingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'zarr', 'wandb'} or fullname.startswith((
            'dexmani_policy.agents', 'dexmani_policy.datasets', 'dexmani_policy.workspace',
            'dexmani_policy.training.resume', 'dexmani_policy.training.builder')):
            raise AssertionError('Preflight imported runtime dependency: '+fullname)
sys.meta_path.insert(0, NoTrainingImports())
from hydra.core.hydra_config import HydraConfig
from unittest.mock import patch
from scripts.remote.resolve_remote_datasets import main
with patch.object(HydraConfig, 'get', side_effect=AssertionError('No Hydra runtime')):
    main()
'''
        command = [sys.executable, '-B', '-c', code, '--check', '--config-name='+self.config_name, *overrides]
        result = subprocess.run(command, cwd=self.project, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['dataset_paths'], [str(self.saved_data)])
        cfg.workspace.output_dir = str(self.root/'destination')
        resolve_training_config(cfg, overrides=overrides)
        after = {p.relative_to(self.root): p.read_bytes() if p.is_file() else None for p in self.root.rglob('*')}
        self.assertEqual(before, after)

    def test_broken_sources_fail_without_fallback(self):
        overrides = self.resume_overrides()
        config = self.source/'config.yaml'
        for contents in (None, 'broken: [', '[]', '{}'):
            with self.subTest(contents=contents):
                if contents is None: config.unlink()
                else: config.write_text(contents)
                with self.assertRaises((ValueError, YAMLError)):
                    remote_datasets.resolve_dataset_paths(self.config_name, overrides)
        self.save_source(); self.checkpoint.unlink()
        with self.assertRaises(FileNotFoundError):
            remote_datasets.resolve_dataset_paths(self.config_name, overrides)

    def test_training_wrapper_binds_real_hydra_runtime(self):
        source_before = (self.source/'config.yaml').read_bytes()
        code = '''
import json
import hydra
from hydra.core.hydra_config import HydraConfig
from dexmani_policy.training.resume import resolve_training_config
@hydra.main(version_base=None, config_path=%r, config_name='recipe')
def main(cfg):
    result = resolve_training_config(cfg)
    assert result.workspace.output_dir == HydraConfig.get().runtime.output_dir
    assert 'claim_token' not in result.workspace
    print(json.dumps({'task': result.task_name, 'output': result.workspace.output_dir,
                      'budget': result.get('max_updates')}))
main()
''' % str(self.config_file.parent)
        for budget in (None, 4):
            destination = self.root/('new_run_'+str(budget))
            overrides = self.resume_overrides('task_name=saved_task', 'hydra.run.dir='+str(destination))
            if budget is not None: overrides.append('+max_updates='+str(budget))
            result = subprocess.run([sys.executable, '-B', '-c', code, *overrides], cwd=self.project,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout),
                             {'task': 'saved_task', 'output': str(destination), 'budget': budget})
            self.assertFalse((destination/'.training_run.json').exists())
        self.assertEqual((self.source/'config.yaml').read_bytes(), source_before)

    def test_wrapper_actual_overrides_gate_launch(self):
        bin_dir = self.root/'bin'; bin_dir.mkdir()
        ssh = bin_dir/'ssh'
        ssh.write_text('#!'+sys.executable+'\n'+r'''
import json, os, shlex, subprocess, sys
from pathlib import Path
root = Path(os.environ['FIXTURE_ROOT'])
command = sys.argv[-1]
with (root/'commands.jsonl').open('a') as f: f.write(json.dumps(command)+'\n')
if not command.startswith('bash -lc '): sys.exit(0)
parts = shlex.split(shlex.split(command)[2])
script = 'scripts/remote/resolve_remote_datasets.py'
if script in parts:
    args = parts[parts.index(script)+1:]
    (root/'preflight.json').write_text(json.dumps(args))
    # Run the real parser and --check locally; never execute remote shell text.
    sys.exit(subprocess.run([sys.executable, '-B', '-m', 'scripts.remote.resolve_remote_datasets', *args]).returncode)
entry = 'dexmani_policy/train.py'
assert entry in parts, parts
(root/'launch.json').write_text(json.dumps(parts[parts.index(entry)+1:]))
''')
        ssh.chmod(0o755)
        rsync = bin_dir/'rsync'; rsync.write_text('#!/bin/sh\nexit 0\n'); rsync.chmod(0o755)
        env = dict(os.environ, PATH=str(bin_dir)+':'+os.environ['PATH'], FIXTURE_ROOT=str(self.root))
        base = ['bash', 'scripts/remote/train_remote.sh']
        extra = ['+resume_from='+str(self.source), 'workspace.output_dir='+str(self.root/'destination')]
        migrated = self.root/'migrated'; migrated.mkdir()
        cases = [('saved_task', [], True), ('saved_task', ['dataset.zarr_path='+str(migrated)], True),
                 ('wrong', [], False), ('wrong', ['task_name=saved_task'], False)]
        for task, additional, allowed in cases:
            with self.subTest(task=task, overrides=additional):
                launch = self.root/'launch.json'; launch.unlink(missing_ok=True)
                result = subprocess.run(base+['--fg', self.config_name, task, *extra, *additional],
                                        env=env, cwd=self.project, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode == 0, allowed, result.stdout+result.stderr)
                self.assertEqual(launch.exists(), allowed)
                if allowed:
                    args = json.loads(launch.read_text())
                    self.assertEqual(json.loads((self.root/'preflight.json').read_text()), ['--check', *args])
                    self.assertIn('task_name='+task, args)
                    actual = resolve_training_config(self.compose(args[1:]), overrides=args[1:])
                    expected = migrated if additional else self.saved_data
                    self.assertEqual(Path(actual.dataset.zarr_path), expected)
        self.saved_data.rmdir(); self.default_data.mkdir()
        result = subprocess.run(base+['--fg', self.config_name, 'saved_task', *extra],
                                env=env, cwd=self.project, capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(str(self.saved_data), result.stderr)
        self.assertFalse((self.root/'launch.json').exists())
        # sync_code.sh still runs in dry-run; its rsync is isolated above too.
        commands = (self.root/'commands.jsonl').read_bytes()
        subprocess.run(base+['--dry-run', self.config_name, 'saved_task', *extra],
                       env=env, cwd=self.project, check=True, capture_output=True, timeout=30)
        self.assertEqual((self.root/'commands.jsonl').read_bytes(), commands)


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
