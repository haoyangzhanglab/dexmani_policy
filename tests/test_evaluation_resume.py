import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import errno
import os

from dexmani_policy.env_runner.common.metrics import compute_evaluation_statistics
from dexmani_policy.evaluation.artifacts import (
    code_version, training_source_evidence, validate_runtime, verify_selection_record,
)
from dexmani_policy.evaluation.config import make_request, resolve_evaluation_request, resolve_tasks
from dexmani_policy.evaluation.selection import publish_best, select_best_checkpoint
from tests.helpers import checkpoint_fixture


def selection_fixture(root):
    candidates, references = [], []
    runtime = {
        'policy_code': {'content_sha256': 'a' * 64, 'commit': None, 'dirty': None},
        'simulator_code': {'content_sha256': 'b' * 64, 'commit': None, 'dirty': None},
        'torch': 'fixture', 'cuda': None, 'parameter_dtypes': ['torch.float32'],
        'numerics': {'float32_matmul_precision': 'highest'},
    }
    for step in (20, 40, 60, 80, 100):
        checkpoint, _, info, _ = checkpoint_fixture(root, step=step, pct=step)
        path = info.checkpoint_path
        candidates.append({'checkpoint': str(path.relative_to(root)), 'global_step': step,
                           'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    inference = {'use_ema': True, 'inference_steps': 2, 'action_steps': 3}
    cfg = copy.deepcopy(checkpoint.effective_config)
    cfg['eval'].update(inference, selection_episodes=2, episodes=1)
    cfg['eval']['video']['enabled'] = False
    cfg['env_runner']['task_name'] = 'fixture'
    request = make_request(cfg, info.checkpoint_path, resolve_tasks(cfg), {'fixture': [3]},
        source={'phase': 'test', 'heldout_from_selection': True}).to_dict()
    for i, candidate in enumerate(candidates):
        episodes = [dict(task_name='fixture', seed=seed, success=bool(i),
                         steps=2 if i else None, total_steps=4) for seed in (1, 2)]
        result = {
            'status': 'completed', 'global_step': candidate['global_step'],
            'checkpoint_sha256': candidate['checkpoint_sha256'],
            'request': {**request, 'checkpoint': str(root / candidate['checkpoint']),
                        'task_seeds': {'fixture': [1, 2]},
                        'source': {'phase': 'selection', 'heldout_from_selection': False}},
            'runtime': runtime, 'episodes': episodes,
            'statistics': compute_evaluation_statistics(episodes),
        }
        ref = f'candidate_{i}.json'
        (root / ref).write_text(json.dumps(result))
        references.append(ref)
    record = {
        'status': 'completed', 'experiment_dir': str(root),
        'seed_pools': {'fixture': [1, 2, 3]}, 'selection_task_seeds': {'fixture': [1, 2]},
        'test_request': request, 'candidates': candidates, 'candidate_results': references,
        'ckpt_relpath': candidates[-1]['checkpoint'], 'global_step': 100,
        'checkpoint_sha256': candidates[-1]['checkpoint_sha256'],
        'inference': inference, 'runtime': runtime,
    }
    path = root / 'selection_result.json'
    path.write_text(json.dumps(record))
    return path, record


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        # Only the external simulator's source identity is synthetic.
        source = patch('dexmani_policy.evaluation.artifacts.simulator_code_version',
                       return_value={'content_sha256': 'b' * 64, 'commit': None, 'dirty': None})
        source.start()
        self.addCleanup(source.stop)
        absent = patch.dict('sys.modules', {'dexmani_sim': None})
        absent.start()
        self.addCleanup(absent.stop)

    def test_selection_publication_and_frozen_rollout(self):
        from dexmani_policy.evaluation.pipeline import run_pipeline
        from dexmani_policy.evaluation.rollout import evaluate_policy

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for step in (20, 40, 60, 80, 100):
                checkpoint, _, _, obs = checkpoint_fixture(root, step=step, pct=step)
            cfg = copy.deepcopy(checkpoint.effective_config)
            cfg['eval'].update(selection_episodes=2, episodes=1)
            cfg['eval']['video']['enabled'] = False
            cfg['env_runner']['eval_seeds'] = [420001, 420002, 420003]
            frame = {key: value[-1].copy() for key, value in obs.items()}
            episodes, closed = [], []
            testcase = self

            class Environment:
                def reset(self, *, seed):
                    episodes.append(seed)
                    self.action_cnt = 0
                    return frame, {}

                def step(self, action):
                    testcase.assertEqual(action.shape, (19,))
                    self.action_cnt += 1
                    done = self.action_cnt == 2
                    return frame, 0, done, False, {'success_condition': done, 'success': done}

                def close(self):
                    closed.append(self)

            # Wrappers count real verification calls; no internal result is replaced.
            with patch('dexmani_policy.env_runner.sim_runner.SimRunner.make_env',
                       side_effect=Environment), \
                 patch('dexmani_policy.evaluation.selection.verify_selection_record',
                       wraps=verify_selection_record) as publication, \
                 patch('dexmani_policy.evaluation.config.verify_selection_record',
                       wraps=verify_selection_record) as frozen:
                result_path = run_pipeline(root, cfg)
                self.assertEqual(publication.call_count, 1)
                self.assertEqual(frozen.call_count, 0)
                pointer = json.loads((root / 'best_ckpt.json').read_text())
                request = resolve_evaluation_request(None, root,
                    selection_record=root / pointer['selection_result'])
                self.assertEqual(frozen.call_count, 1)
                repeated = evaluate_policy(request, output_root=root / 'independent_test')
                ordinary = resolve_evaluation_request(cfg, root)
                ordinary_result = evaluate_policy(ordinary, output_root=root / 'ordinary')
            result = json.loads(result_path.read_text())
            pointer = json.loads((root / 'best_ckpt.json').read_text())
            record, checkpoint_path = verify_selection_record(root, root / pointer['selection_result'])
            self.assertEqual(record['global_step'], 100)
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['request']['checkpoint'], str(checkpoint_path))
            self.assertEqual(result['checkpoint_sha256'], record['checkpoint_sha256'])
            self.assertTrue(result['request']['source']['heldout_from_selection'])
            for start in range(2, 10, 2):
                self.assertEqual(episodes[:2], episodes[start:start + 2])
            self.assertNotIn(episodes[10], episodes[:2])
            self.assertEqual(episodes[10:], [episodes[10]] * 3)
            self.assertEqual(len(episodes), 13)
            self.assertEqual(len(closed), 8)
            self.assertEqual(json.loads(repeated.read_text())['request'], result['request'])
            self.assertFalse(json.loads(ordinary_result.read_text())['request']['source']['heldout_from_selection'])

    def test_runtime_identity_and_data_guarantee_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, record = selection_fixture(Path(tmp))
            runtime = record['runtime']
            relocated = copy.deepcopy(runtime)
            relocated['device'] = 'cuda:3'
            relocated['policy_code']['commit'] = 'display-only'
            validate_runtime(runtime, relocated)
            for key in ('torch', 'cuda', 'parameter_dtypes', 'numerics'):
                changed = copy.deepcopy(runtime)
                changed[key] = 'different'
                with self.assertRaises(ValueError):
                    validate_runtime(runtime, changed)
            cfg = {'data_recipe': [{'source_evidence': {'task_name': 'a', 'episode_seeds': [1, 2]},
                                    'split_manifest': {'train_indices': [0]}}]}
            evidence = training_source_evidence(cfg, {'test': {'a': [1], 'b': [1]}})
            self.assertEqual(evidence['same_task_seed_overlap'], {'test': {'a': [1]}})
            self.assertEqual(evidence['training_test_independence'], 'unverified')
            self.assertEqual(training_source_evidence({}, {})['training_test_independence'], 'unverified')

    def test_selection_rejects_incomplete_set_before_episodes(self):
        for steps in ((), (20, 40)):
            with self.subTest(steps=steps), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                checkpoint, _, info, _ = checkpoint_fixture(root, step=100, pct=100)
                info.checkpoint_path.unlink()
                for step in steps:
                    checkpoint_fixture(root, step=step, pct=step)
                cfg = checkpoint.effective_config
                cfg['eval'].update(selection_episodes=1, episodes=1)
                cfg['env_runner']['eval_seeds'] = [1, 2]
                missing = [step for step in (20, 40, 60, 80, 100) if step not in steps]
                with patch('dexmani_policy.env_runner.sim_runner.SimRunner.make_env') as env:
                    with self.assertRaises(ValueError) as error:
                        select_best_checkpoint(root, cfg)
                    self.assertIn(f'Missing selection milestone steps: {missing}', str(error.exception))
                    env.assert_not_called()
                self.assertFalse((root / 'best_ckpt.json').exists())

    def test_best_identity_and_load_time_replacement(self):
        from dexmani_policy.common.checkpoint import load_training_checkpoint, save_training_checkpoint
        from dexmani_policy.evaluation.rollout import evaluate_policy

        for entry in ('best_before_request', 'best_after_request', 'frozen_after_request'):
            with self.subTest(entry=entry), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                path, record = selection_fixture(root)
                _, checkpoint = publish_best(root, path)
                cfg = record['test_request']['config']
                cfg['env_runner']['eval_seeds'] = [1, 2, 3]
                cfg['eval']['video']['enabled'] = False
                if entry == 'frozen_after_request':
                    request = resolve_evaluation_request(None, root, selection_record=path)
                elif entry == 'best_after_request':
                    request = resolve_evaluation_request(cfg, root)
                replaced = load_training_checkpoint(checkpoint)
                next(iter(replaced.ema_model_state.values())).add_(.25)
                save_training_checkpoint(checkpoint, replaced)
                self.assertEqual(load_training_checkpoint(checkpoint).global_step, record['global_step'])
                if entry == 'best_before_request':
                    request = resolve_evaluation_request(cfg, root)
                with patch('dexmani_policy.env_runner.sim_runner.SimRunner.make_env') as env:
                    with self.assertRaisesRegex(ValueError, 'Checkpoint content changed'):
                        evaluate_policy(request, output_root=root / 'replaced')
                    env.assert_not_called()
                failure = json.loads(next((root / 'replaced').glob('*/result_details.json')).read_text())
                self.assertEqual(failure['status'], 'failed')
                self.assertEqual(failure['episodes'], [])

    def test_selection_rejects_conflicting_milestones_before_episodes(self):
        for damage in ('duplicate', 'percentage'):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                _, record = selection_fixture(root)
                candidate = root / record['candidates'][0]['checkpoint']
                if damage == 'duplicate':
                    shutil.copyfile(candidate, candidate.with_name('epoch=1-step=20-milestone=20pct.pt'))
                else:
                    candidate.rename(candidate.with_name('epoch=0-step=20-milestone=40pct.pt'))
                cfg = record['test_request']['config']
                cfg['env_runner']['eval_seeds'] = [1, 2, 3]
                with patch('dexmani_policy.env_runner.sim_runner.SimRunner.make_env') as env:
                    with self.assertRaises(ValueError):
                        select_best_checkpoint(root, cfg)
                    env.assert_not_called()

    def test_best_requires_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, record = selection_fixture(root)
            publish_best(root, path)
            pointer = root / 'best_ckpt.json'
            info = json.loads(pointer.read_text())
            for digest in (None, 'invalid'):
                if digest is None:
                    info.pop('checkpoint_sha256')
                else:
                    info['checkpoint_sha256'] = digest
                pointer.write_text(json.dumps(info))
                with self.assertRaises((KeyError, ValueError)):
                    resolve_evaluation_request(record['test_request']['config'], root)

    def test_full_evidence_required_by_frozen_entry(self):
        for damage in ('missing_result', 'statistics', 'winner', 'duplicate_episode',
                       'missing_episode', 'overlap', 'replaced_weights', 'runtime',
                       'missing_identity', 'partial_candidates'):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                path, record = selection_fixture(root)
                candidate_path = root / record['candidate_results'][0]
                result = json.loads(candidate_path.read_text())
                if damage == 'missing_result':
                    candidate_path.unlink()
                elif damage == 'statistics':
                    result['statistics']['n_success'] = 99
                elif damage == 'winner':
                    record.update(ckpt_relpath=record['candidates'][0]['checkpoint'], global_step=20,
                                  checkpoint_sha256=record['candidates'][0]['checkpoint_sha256'])
                    record['test_request']['checkpoint'] = str(root / record['ckpt_relpath'])
                elif damage == 'duplicate_episode':
                    result['episodes'][1] = result['episodes'][0]
                elif damage == 'missing_episode':
                    result['episodes'].pop()
                elif damage == 'overlap':
                    record['test_request']['task_seeds'] = {'fixture': [1]}
                elif damage == 'replaced_weights':
                    shutil.copyfile(root / record['candidates'][0]['checkpoint'], root / record['ckpt_relpath'])
                elif damage == 'runtime':
                    result['runtime']['torch'] = 'different'
                elif damage == 'partial_candidates':
                    record['candidates'].pop(0)
                    record['candidate_results'].pop(0)
                else:
                    record.pop('checkpoint_sha256')
                if damage in {'statistics', 'duplicate_episode', 'missing_episode', 'runtime'}:
                    candidate_path.write_text(json.dumps(result))
                path.write_text(json.dumps(record))
                for resolve in (verify_selection_record,
                                lambda root, path: resolve_evaluation_request(None, root, selection_record=path)):
                    with self.assertRaises((ValueError, RuntimeError, FileNotFoundError, KeyError)):
                        resolve(root, path)

    def test_verified_publication_and_relocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'run'
            root.mkdir()
            path, record = selection_fixture(root)
            publish_best(root, path)
            before = (root / 'best_ckpt.json').read_bytes()
            result_path = root / record['candidate_results'][0]
            result = result_path.read_bytes()
            result_path.unlink()
            with self.assertRaises(FileNotFoundError):
                publish_best(root, path)
            self.assertEqual(before, (root / 'best_ckpt.json').read_bytes())
            result_path.write_bytes(result)
            moved = root.rename(Path(tmp) / 'moved')
            verified, checkpoint = verify_selection_record(moved, moved / path.name)
            request = resolve_evaluation_request(None, moved, selection_record=moved / path.name)
            self.assertEqual(request.checkpoint, str(checkpoint))
            result = subprocess.run([
                __import__('sys').executable, '-B', '-m', 'dexmani_policy.evaluation.selection',
                '--experiment-dir', str(moved), '--selection-record', str(moved / path.name),
                '--publish-best',
            ], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_snapshot_identity_is_content_not_parent_git(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def git(*args):
                return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.STDOUT)
            git('init', '-q')
            (root / '.gitignore').write_text('outputs/\n')
            (root / 'main.py').write_text('VALUE = 1\n')
            git('add', '.')
            git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                'commit', '-qm', 'fixture')
            snapshot = root / 'outputs' / 'snapshot'
            snapshot.mkdir(parents=True)
            (snapshot / 'main.py').write_text('VALUE = 2\n')
            version = code_version(snapshot)
            self.assertIsNone(version['commit'])
            self.assertNotEqual(code_version(root)['content_sha256'], version['content_sha256'])
            (snapshot / 'logs').mkdir()
            (snapshot / 'logs' / 'run.py').write_text('ignored log\n')
            self.assertEqual(version, code_version(snapshot))
            (snapshot / 'main.py').write_text('VALUE = 3\n')
            self.assertNotEqual(version['content_sha256'], code_version(snapshot)['content_sha256'])
            other = root / 'independent'
            shutil.copytree(snapshot, other)
            self.assertEqual(code_version(other)['content_sha256'], code_version(snapshot)['content_sha256'])
            (snapshot / 'dexmani_policy').symlink_to(root, target_is_directory=True)
            self.assertEqual(code_version(other)['content_sha256'], code_version(snapshot)['content_sha256'])


class ResumeArchiveTests(unittest.TestCase):
    def test_prefix_fork_repeated_resume_and_move(self):
        from dexmani_policy.common.experiments import archive_resume_milestones, discover_milestone_checkpoints
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent, first, second = (root / name for name in ('parent', 'first', 'second'))
            for step in (20, 40, 60):
                checkpoint_fixture(parent, step=step, pct=step)
            checkpoint, _, source, _ = checkpoint_fixture(parent, step=50, pct=None)
            archive_resume_milestones(source.checkpoint_path, first, checkpoint)
            self.assertEqual([m.global_step for m in discover_milestone_checkpoints(first)], [20, 40])
            checkpoint_fixture(first, step=60, pct=60, resume_from=str(source.checkpoint_path))
            later, _, later_info, _ = checkpoint_fixture(first, step=70, pct=None, resume_from=str(source.checkpoint_path))
            shutil.rmtree(parent)
            archive_resume_milestones(later_info.checkpoint_path, second, later)
            shutil.rmtree(first)
            moved = second.rename(root / 'moved')
            self.assertEqual([m.global_step for m in discover_milestone_checkpoints(moved)], [20, 40, 60])
            self.assertTrue(all(not m.path.is_symlink() for m in discover_milestone_checkpoints(moved)))

    def test_source_allows_missing_but_rejects_conflicting_prefix(self):
        from dexmani_policy.common.experiments import archive_resume_milestones, discover_milestone_checkpoints
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent, child = root / 'parent', root / 'child'
            checkpoint_fixture(parent, step=20, pct=20)
            _, _, old_info, _ = checkpoint_fixture(parent, step=30, pct=None)
            checkpoint_fixture(child, step=40, pct=40, resume_from=str(old_info.checkpoint_path))
            current, _, info, _ = checkpoint_fixture(child, step=50, pct=None, resume_from=str(old_info.checkpoint_path))
            # Preserve only available candidates; never search accessible ancestors.
            archive_resume_milestones(info.checkpoint_path, root / 'missing', current)
            self.assertEqual([m.global_step for m in discover_milestone_checkpoints(root / 'missing')], [40])
            _, _, prefix, _ = checkpoint_fixture(child, step=20, pct=20)
            shutil.copyfile(prefix.checkpoint_path, child / 'checkpoints/epoch=1-step=20-milestone=20pct.pt')
            with self.assertRaisesRegex(ValueError, 'Conflicting'):
                archive_resume_milestones(info.checkpoint_path, root / 'conflict', current)

    def test_source_without_any_milestones(self):
        from dexmani_policy.common.experiments import archive_resume_milestones, discover_milestone_checkpoints
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint, _, info, _ = checkpoint_fixture(root / 'source', step=50, pct=None)
            archive_resume_milestones(info.checkpoint_path, root / 'resumed', checkpoint)
            self.assertEqual(discover_milestone_checkpoints(root / 'resumed'), [])

    def test_archive_rejects_state_and_recipe_mismatch_before_writing(self):
        from dexmani_policy.common.checkpoint import save_training_checkpoint
        from dexmani_policy.common.experiments import archive_resume_milestones

        for damage in ('step', 'recipe', 'percentage'):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                checkpoint, _, info, _ = checkpoint_fixture(root / 'parent')
                changed = copy.deepcopy(checkpoint)
                if damage == 'step':
                    changed.global_step += 1
                elif damage == 'recipe':
                    changed.effective_config['agent']['horizon'] += 1
                else:
                    info.checkpoint_path = info.checkpoint_path.rename(
                        info.checkpoint_path.with_name('epoch=0-step=20-milestone=40pct.pt'))
                save_training_checkpoint(info.checkpoint_path, changed)
                with self.assertRaises(ValueError):
                    archive_resume_milestones(info.checkpoint_path, root / 'new', checkpoint)
                self.assertFalse((root / 'new').exists())

    def test_cross_filesystem_only_falls_back_for_exdev(self):
        from dexmani_policy.common.experiments import archive_resume_milestones, file_sha256
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint, _, info, _ = checkpoint_fixture(root / 'parent')
            link = os.link
            def cross_device(src, dst, **kwargs):
                if Path(src) == info.checkpoint_path:
                    raise OSError(errno.EXDEV, 'fixture cross device')
                return link(src, dst, **kwargs)
            with patch('os.link', side_effect=cross_device):
                archive_resume_milestones(info.checkpoint_path, root / 'new', checkpoint)
            copy = root / 'new/checkpoints' / info.checkpoint_path.name
            self.assertEqual(file_sha256(copy), file_sha256(info.checkpoint_path))
            self.assertNotEqual(copy.stat().st_ino, info.checkpoint_path.stat().st_ino)
            with patch('os.link', side_effect=PermissionError('fixture denied')):
                with self.assertRaises(PermissionError):
                    archive_resume_milestones(info.checkpoint_path, root / 'denied', checkpoint)


if __name__ == '__main__':
    unittest.main()
