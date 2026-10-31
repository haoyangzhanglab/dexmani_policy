"""Selection orchestration tests; these do not claim simulation backend coverage."""

import fcntl
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from omegaconf import open_dict

from dexmani_policy.common.checkpoint import load_checkpoint, save_checkpoint
from dexmani_policy.common.io_util import write_json
from dexmani_policy.deployment.metadata import inspect_policy
from dexmani_policy.evaluation.selection import candidate_score, run_pipeline
from dexmani_policy.evaluation.rollout import create_run_directory, evaluate_policy
from dexmani_policy.test_tools import test_training_resume
from dexmani_policy.train import run_training


class EvaluationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_explicit_subset_budget_ranking_record_and_test_separation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / 'run'
            cfg = test_training_resume.TrainingResumeChecks().make_config(root, run)
            with open_dict(cfg):
                cfg.env_runner.eval_seeds = list(range(20))
                cfg.eval.use_ema = False
            run_training(cfg, max_updates=1)
            state = load_checkpoint(run / 'checkpoints/latest.pt')
            candidates = [save_checkpoint(run / f'checkpoints/candidate-{step}.pt',
                                         replace(state, global_step=step)) for step in (2, 6, 4)]
            calls = []

            def score(checkpoint, **settings):
                calls.append((Path(checkpoint), settings))
                step = load_checkpoint(checkpoint).global_step
                stats = {'success_rate': 0.5, 'avg_steps': 10 if step == 2 else 8}
                path = settings['output_dir'] / 'result_details.json'
                path.parent.mkdir(parents=True)
                result = {'global_step': step, 'statistics': stats}
                write_json(path, result)
                return result, path

            with patch('dexmani_policy.evaluation.selection.evaluate_policy', side_effect=score):
                path = run_pipeline(candidates, weights='ema', inference_steps=3, action_steps=2,
                                    selection_episodes=2, test_episodes=3, device=torch.device('cpu'),
                                    output_dir=run / 'comparison')
            record = json.loads(path.read_text())
            self.assertEqual(record['global_step'], 6)
            self.assertEqual(record['device'], 'cpu')
            self.assertEqual(len(record['candidates']), 3)
            self.assertEqual(len(calls), 4)
            for _, settings in calls[:3]:
                self.assertEqual(settings['seeds'], record['selection_seeds'])
                self.assertEqual(settings['inference_steps'], 3)
                self.assertEqual(settings['action_steps'], 2)
                self.assertEqual(settings['weights'], 'ema')
                self.assertFalse(settings['video'])
            self.assertEqual(calls[-1][0], candidates[1])
            self.assertEqual(calls[-1][1]['weights'], 'ema')
            self.assertEqual(calls[-1][1]['seeds'], record['test_seeds'])
            self.assertFalse(set(record['selection_seeds']) & set(record['test_seeds']))
            with patch('dexmani_policy.evaluation.selection.evaluate_policy', side_effect=score):
                with self.assertRaises(FileExistsError):
                    run_pipeline(candidates, selection_episodes=2, test_episodes=3,
                                 output_dir=run / 'comparison')
            with (run / '.pipeline.lock').open('w') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError, 'Another eval pipeline'):
                    run_pipeline(candidates, output_dir=run / 'locked')
                self.assertFalse((run / 'locked').exists())

            def fail_test(checkpoint, **settings):
                if settings['output_dir'].name == 'test':
                    raise RuntimeError('held-out failure')
                return score(checkpoint, **settings)

            with patch('dexmani_policy.evaluation.selection.evaluate_policy', side_effect=fail_test):
                with self.assertRaisesRegex(RuntimeError, 'held-out failure'):
                    run_pipeline(candidates, weights='ema', inference_steps=3, action_steps=2,
                                 selection_episodes=2, test_episodes=3,
                                 output_dir=run / 'failed-test')
            failed = json.loads((run / 'failed-test/selection_result.json').read_text())
            self.assertEqual(failed['status'], 'failed')
            self.assertIn('held-out failure', failed['error'])
            self.assertEqual(failed['global_step'], 6)
            published = (run / 'best_ckpt.json').read_bytes()

            def fail_best(path, record, **kwargs):
                if Path(path).name == 'best_ckpt.json':
                    raise RuntimeError('best publication failure')
                return write_json(path, record, **kwargs)

            with patch('dexmani_policy.evaluation.selection.evaluate_policy', side_effect=score), \
                 patch('dexmani_policy.evaluation.selection.write_json', side_effect=fail_best):
                with self.assertRaisesRegex(RuntimeError, 'best publication failure'):
                    run_pipeline(candidates, selection_episodes=2, test_episodes=3,
                                 output_dir=run / 'failed-publication')
            failed = json.loads((run / 'failed-publication/selection_result.json').read_text())
            self.assertEqual(failed['status'], 'failed')
            self.assertIn('best publication failure', failed['error'])
            self.assertEqual((run / 'best_ckpt.json').read_bytes(), published)
            (run / 'config.yaml').unlink()
            info = inspect_policy(run, weights='raw')
            self.assertEqual(info.checkpoint_path, candidates[1])
            self.assertEqual(info.inference_steps, 3)
            self.assertEqual(info.future_steps, 3)
            with patch('dexmani_policy.evaluation.rollout.restore_policy', side_effect=RuntimeError('load failure')):
                with self.assertRaisesRegex(RuntimeError, 'load failure'):
                    evaluate_policy(candidates[0], weights='raw', seeds=[1], device='cpu',
                                    output_dir=run / 'failure')
            failure = json.loads((run / 'failure/result_details.json').read_text())
            self.assertEqual(failure['status'], 'failed')
            self.assertIn('load failure', failure['error'])

    def test_none_steps_loses_tie_and_later_global_step_wins(self):
        def candidate(rate, steps, step):
            return {'statistics': {'success_rate': rate, 'avg_steps': steps}, 'global_step': step}
        values = [candidate(0.5, None, 9), candidate(0.5, 8, 4), candidate(0.5, 8, 6)]
        self.assertIs(max(values, key=candidate_score), values[2])

    def test_output_isolation_and_atomic_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = create_run_directory(root)
            second = create_run_directory(root)
            self.assertNotEqual(first, second)
            self.assertEqual(create_run_directory(root, second), second)
            path = first / 'result.json'
            write_json(path, {'value': 1}, overwrite=False)
            original = path.read_bytes()
            with self.assertRaises(ValueError):
                write_json(path, {'value': float('nan')})
            with self.assertRaises(FileExistsError):
                write_json(path, {'value': 2}, overwrite=False)
            self.assertEqual(path.read_bytes(), original)
            self.assertFalse(list(first.glob('.dexmani-publish-*')))
            with self.assertRaises(FileExistsError):
                create_run_directory(root, first)

    def test_failure_details_write_does_not_replace_original_exception(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / 'run'
            cfg = test_training_resume.TrainingResumeChecks().make_config(root, run)
            run_training(cfg, max_updates=1)
            original = RuntimeError('original restore failure')
            stderr = io.StringIO()
            with patch('dexmani_policy.evaluation.rollout.restore_policy', side_effect=original), \
                 patch('dexmani_policy.common.io_util.write_json', side_effect=OSError('details write failure')), \
                 redirect_stderr(stderr):
                with self.assertRaises(RuntimeError) as caught:
                    evaluate_policy(run / 'checkpoints/latest.pt', weights='raw', seeds=[1],
                                    device='cpu', output_dir=run / 'failed-write')
            self.assertIs(caught.exception, original)
            self.assertIn('details write failure', stderr.getvalue())

    def test_inspect_selectors_and_checkpoint_path_limits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / 'experiments/nested/run'
            cfg = test_training_resume.TrainingResumeChecks().make_config(root, run)
            run_training(cfg, max_updates=1)
            latest = (run / 'checkpoints/latest.pt').resolve()
            state = load_checkpoint(latest)
            later = save_checkpoint(run / 'checkpoints/epoch=0002-step=00000002-milestone=20pct.pt',
                                    replace(state, global_step=2))
            first = save_checkpoint(run / 'checkpoints/epoch=0001-step=00000009-milestone=20pct.pt',
                                    replace(state, global_step=9))
            write_json(run / 'best_ckpt.json', {
                'ckpt_relpath': str(later.relative_to(run)),
                'inference': {'use_ema': True, 'inference_steps': 3},
            })
            outside = save_checkpoint(run / 'outside.pt', state)
            (run / 'checkpoints/escape.pt').symlink_to(outside)
            with patch('dexmani_policy.deployment.metadata.PROJECT_ROOT', root):
                best = inspect_policy('nested/run')
                self.assertEqual(best.checkpoint_path, later)
                self.assertEqual(best.weights, 'ema')
                self.assertEqual(best.inference_steps, 3)
                self.assertEqual(inspect_policy('experiments/nested/run', 'latest').checkpoint_path,
                                 latest)
                self.assertEqual(inspect_policy(run, '020pct').checkpoint_path, first)
                self.assertEqual(inspect_policy(run, later.name).checkpoint_path, later)
                self.assertEqual(inspect_policy(run, later).checkpoint_path, later)
                with self.assertRaisesRegex(FileNotFoundError, 'No 40% milestone'):
                    inspect_policy(run, '40pct')
                for selector in ('../outside.pt', 'escape.pt', outside):
                    with self.subTest(selector=selector):
                        with self.assertRaisesRegex(ValueError, 'inside'):
                            inspect_policy(run, selector)


if __name__ == '__main__':
    unittest.main()
