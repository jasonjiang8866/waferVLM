import copy
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from wafervlm import confirmation as c
from wafervlm.core import LABELS, PROMPT, metrics, write_json, write_jsonl, sha256_file
from wafervlm.experiment import seal_output
from wafervlm.modeling import MODEL_ID, MODEL_REVISION
from wafervlm.resume import REQUIRED


class ConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / 'data'
        self.data.mkdir()
        write_json(self.data / 'dataset.json', {})

    def checkpoint(self, seed=42, step=100, root=None):
        root = root or self.root / f'train-{seed}-{step}'
        cp = root / f'checkpoints/checkpoint-{step}'
        cp.mkdir(parents=True)
        for name in REQUIRED:
            (cp / name).write_text('{}')
        write_json(cp / 'trainer_state.json', {'global_step': step})
        write_json(root / 'run.json', dict(model_id=MODEL_ID, model_revision=MODEL_REVISION,
                  dataset_sha256=sha256_file(self.data / 'dataset.json'), labels=list(LABELS),
                  prompt=PROMPT, config=c.training_config(seed, step), resume=None))
        seal_output(root)
        return cp

    def test_recipe_is_frozen_and_training_seed_is_not_split_seed(self):
        cfg = c.recipe()
        self.assertEqual(cfg['new_seeds'], [42, 123])
        self.assertEqual(cfg['seeds'], [17, 42, 123])
        self.assertEqual(cfg['split_seed'], 17)
        self.assertEqual(cfg['stages'], [100, 150, 200, 250])
        self.assertEqual(cfg['endpoints'], [200, 250])
        self.assertEqual((cfg['repeats'], cfg['evaluation_seed']), (3, 17))
        self.assertFalse(cfg['holdout_access'])
        for seed in (42, 123):
            self.assertEqual(c.training_config(seed, 100)['learning_rate'], .0002)
            self.assertEqual(c.training_config(seed, 150)['learning_rate'], .00005)
        with self.assertRaises(ValueError):
            c.train_seed(self.data, self.root/'bad', 17)

    def test_resume_requires_correct_seed_lr_hash_and_complete_rng(self):
        cp = self.checkpoint()
        self.assertEqual(c.inspect_checkpoint(cp, self.data, 42, 100)['step'], 100)
        with self.assertRaisesRegex(ValueError, 'seed'):
            c.inspect_checkpoint(cp, self.data, 123, 100)
        (cp/'rng_state.pth').write_text('tampered')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            c.inspect_checkpoint(cp, self.data, 42, 100)
        (cp/'rng_state.pth').unlink()
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            c.inspect_checkpoint(cp, self.data, 42, 100)

    def test_dataset_drift_rejected_before_process(self):
        with patch.object(c, 'run_process') as run, self.assertRaisesRegex(ValueError, 'dataset'):
            c.train_seed(self.data, self.root/'out', 42)
        run.assert_not_called()

    def fake_training(self, args, deadline, log):
        # Parse value flags explicitly; the intervention is a boolean flag.
        flags = {args[i]: args[i+1] for i in range(4, len(args)-1, 2)}
        self.checkpoint(int(flags['--seed']), int(flags['--steps']), Path(flags['--output']))

    def test_fresh_seed_and_recovery_commands_keep_single_lr_transition(self):
        for seed in (42, 123):
            with patch.object(c, 'check_dataset'), patch.object(c, 'verify_training', return_value={}), \
                    patch.object(c, 'run_process', side_effect=self.fake_training) as run:
                c.train_seed(self.data, self.root/f'out{seed}', seed)
            calls = [x.args[0] for x in run.call_args_list]
            self.assertEqual([a[a.index('--steps')+1] for a in calls], ['100','150','200','250'])
            self.assertNotIn('--resume-from', calls[0])
            self.assertEqual(sum('--lower-learning-rate' in a for a in calls), 1)
            self.assertIn('--lower-learning-rate', calls[1])
            self.assertTrue(all(a[a.index('--seed')+1] == str(seed) for a in calls))
        cp = self.checkpoint(42, 100)
        with patch.object(c, 'check_dataset'), patch.object(c, 'verify_training', return_value={}), \
                patch.object(c, 'run_process', side_effect=self.fake_training) as run:
            c.train_seed(self.data, self.root/'recovered', 42, cp)
        self.assertEqual(len(run.call_args_list), 3)
        self.assertIn('--lower-learning-rate', run.call_args_list[0].args[0])

    def test_failed_stage_is_sealed_and_cannot_be_reported_completed(self):
        def fail(args, *rest):
            self.fake_training(args, *rest)
            raise subprocess.TimeoutExpired(args, 1)
        with patch.object(c, 'check_dataset'), patch.object(c, 'run_process', side_effect=fail), \
                self.assertRaises(subprocess.TimeoutExpired):
            c.train_seed(self.data, self.root/'failed', 123)
        status = json.loads((self.root/'failed/status.json').read_text())
        self.assertEqual(status['status'], 'failed')
        c.verify_tree(self.root/'failed')
        self.assertFalse((self.root/'failed/endpoints.json').exists())

    def test_matrix_requires_all_six_and_frozen_hashes(self):
        spec = [{'seed': s, 'step': t, 'checkpoint': '/fixture', 'files': {'x': 'hash'}, 'run_sha256': 'run'}
                for s in (17,42,123) for t in (200,250)]
        with patch.object(c, 'inspect_checkpoint', side_effect=lambda cp, d, s, t: {**spec[0], 'seed':s, 'step':t}):
            self.assertEqual(len(c.prepare_endpoints(spec, self.data)), 6)
            for bad in (spec[:-1], spec+[spec[0]], [spec[0]]*6):
                with self.assertRaises(ValueError): c.prepare_endpoints(bad, self.data)
            bad = copy.deepcopy(spec); bad[-1]['files']['x'] = 'changed'
            with self.assertRaisesRegex(ValueError, 'changed'):
                c.prepare_endpoints(bad, self.data)

    def repeat_fixture(self):
        cfg = c.recipe()
        reference = [dict(id=str(i), label=LABELS[i//20], image_sha256=f'{i:064x}',
                          prediction=LABELS[i//20], response=LABELS[i//20], truncated=False,
                          seconds=.4, generated_tokens=1) for i in range(160)]
        records = []
        for seed in (17,42,123):
            for step in (200,250):
                for repeat in range(1,4):
                    out = self.root/f'eval-{seed}-{step}-{repeat}';out.mkdir()
                    pred = copy.deepcopy(reference)
                    if repeat == 1: pred[0].update(prediction=None, response='invalid')
                    write_jsonl(out/'predictions.jsonl', pred)
                    write_json(out/'metrics.json', {**metrics(pred), 'answer_token_loss': .1,
                        'answer_tokens': 320, 'mean_generation_seconds': .4, 'peak_allocated_bytes': 10})
                    write_json(out/'run.json', dict(
                        model_id=MODEL_ID, model_revision=MODEL_REVISION, dataset_sha256=cfg['dataset_sha256'],
                        validation_sha256=cfg['validation_sha256'], adapter_sha256='weights',
                        source_sha='a'*40, image_digest='sha256:'+'b'*64, packages={'torch':'2.10.0'},
                        gpu='RTX 4090', cuda='12.8', driver='580', host='host-a', processor_identity={'hash':'processor'},
                        base_weights_sha256={'model.safetensors':'hash'},
                        generation=dict(do_sample=False, num_beams=1, max_new_tokens=16, use_cache=True),
                        process_instance=f'{seed}-{step}-{repeat}', prompt=PROMPT, labels=list(LABELS),
                        config=dict(split='validation', device='cuda', max_new_tokens=16, seed=17,
                            do_sample=False, enable_thinking=False, include_loss=True, adapter=True)))
                    seal_output(out)
                    records.append(dict(seed=seed, step=step, repeat=repeat, output=str(out), adapter_sha256='weights'))
        return records, reference

    def test_first_pass_headline_not_best_and_fixed_eight_class_metrics(self):
        records, reference = self.repeat_fixture()
        report = c.summarize_repeats(records, reference)
        self.assertEqual(report['pairs'][0]['headline']['accuracy'], 159/160)
        self.assertEqual(report['pairs'][0]['passes'][1]['score']['accuracy'], 1)
        self.assertEqual(report['pairs'][0]['passes'][1]['changed_prediction_ids'], ['0'])
        self.assertTrue(report['same_environment'])
        self.assertEqual(report['paired_deltas'][0]['role'], 'exploratory')
        self.assertEqual(metrics([dict(label='center', prediction='center')])['macro_f1'], 1/8)

    def test_changed_metrics_settings_process_order_and_weights_rejected(self):
        records, reference = self.repeat_fixture()
        out = Path(records[-1]['output'])
        original = json.loads((out/'run.json').read_text())
        for key, val in [('process_instance', '17-200-1'), ('adapter_sha256','other'), ('validation_sha256','other')]:
            write_json(out/'run.json', {**original, key:val});seal_output(out)
            with self.subTest(key=key), self.assertRaises(ValueError): c.summarize_repeats(records, reference)
        write_json(out/'run.json', original)
        score = json.loads((out/'metrics.json').read_text());score['macro_f1'] = .123
        write_json(out/'metrics.json', score);seal_output(out)
        with self.assertRaisesRegex(ValueError, 'metrics'): c.summarize_repeats(records, reference)

    def test_changed_hardware_is_reported_as_limitation(self):
        records, reference = self.repeat_fixture()
        out = Path(records[-1]['output'])
        run = json.loads((out/'run.json').read_text());run['gpu'] = 'RTX 3090'
        write_json(out/'run.json', run);seal_output(out)
        self.assertFalse(c.summarize_repeats(records, reference)['same_environment'])

    def test_unlisted_file_and_hash_corruption_rejected(self):
        cp = self.checkpoint();root = cp.parent.parent
        (root/'unlisted.txt').write_text('x')
        with self.assertRaises(ValueError): c.verify_tree(root)
        (root/'unlisted.txt').unlink();(cp/'optimizer.pt').write_text('x')
        with self.assertRaises(ValueError): c.verify_tree(root)

    def test_training_acceptance_rejects_lr_gradient_and_frozen_parameter_drift(self):
        cp = self.checkpoint();root=cp.parent.parent
        report = dict(start_step=0,end_step=100,executed_steps=100,gradient_checks=100,
                      nonzero_gradient_checks=100,learning_rate_intervention=None,
                      learning_rate_trace=[dict(step=i,rates=[.0002,.0002]) for i in range(1,101)],
                      train_loss=.2,peak_allocated_bytes=10,frozen_parameters_verified=True)
        write_json(root/'training.json',report);seal_output(root)
        c.verify_training(root,self.data,42,0,100)
        for key,val in [('gradient_checks',99),('nonzero_gradient_checks',0),('frozen_parameters_verified',False),
                        ('learning_rate_trace',[dict(step=i,rates=[.00005]) for i in range(1,101)])]:
            write_json(root/'training.json',{**report,key:val});seal_output(root)
            with self.subTest(key=key),self.assertRaises(ValueError):c.verify_training(root,self.data,42,0,100)

    def test_acceptance_binds_restored_optimizer_rng_scheduler_and_parent_metadata(self):
        parent=self.checkpoint(42,100)
        cp=self.checkpoint(42,150);root=cp.parent.parent
        run=json.loads((root/'run.json').read_text())
        run['resume']=c.check_resume(parent,self.data,MODEL_REVISION,c.training_config(42,150),lower_learning_rate=True)
        write_json(root/'run.json',run)
        report=dict(start_step=100,end_step=150,executed_steps=50,gradient_checks=50,nonzero_gradient_checks=50,
                    optimizer_steps_at_resume=[100],scheduler_step_at_resume=100,train_loss=.2,
                    peak_allocated_bytes=10,frozen_parameters_verified=True,
                    learning_rate_intervention={'from':.0002,'to':.00005,'scheduler_last_epoch':100},
                    learning_rate_trace=[dict(step=i,rates=[.00005,.00005]) for i in range(101,151)])
        write_json(root/'training.json',report);seal_output(root)
        c.verify_training(root,self.data,42,100,150,parent)
        for key in ('rng_sha256','optimizer_sha256','scheduler_sha256','parent_run_sha256','checkpoint_state_sha256'):
            altered=copy.deepcopy(run);altered['resume'][key]='bad'
            write_json(root/'run.json',altered);seal_output(root)
            with self.subTest(key=key),self.assertRaises(ValueError):c.verify_training(root,self.data,42,100,150,parent)

    def test_timeout_reaps_real_process_group(self):
        pidfile = self.root/'pid'
        code = 'import os,time,pathlib;pathlib.Path('+repr(str(pidfile))+').write_text(str(os.getpid()));time.sleep(30)'
        with self.assertRaises(subprocess.TimeoutExpired):
            c.run_process([sys.executable,'-c',code], time.monotonic()+.5, self.root/'process.log')
        with self.assertRaises(ProcessLookupError): os.kill(int(pidfile.read_text()), 0)

    def test_subprocess_receives_no_account_or_result_credentials(self):
        with patch.dict(os.environ, {k:'sensitive' for k in ('RUNPOD_API_KEY','GH_TOKEN','GITHUB_TOKEN','WAFER_RESULT_TOKEN')}):
            c.run_process([sys.executable,'-c', "import os;assert not any(k in os.environ for k in ['RUNPOD_API_KEY','GH_TOKEN','GITHUB_TOKEN','WAFER_RESULT_TOKEN'])"], time.monotonic()+5, self.root/'env.log')


if __name__ == '__main__':
    unittest.main()
