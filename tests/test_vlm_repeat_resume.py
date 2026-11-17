import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import test_confirmation as fixture_module
from runpod import vlm_repeat_resume as resume, vlm_controller as controller
from wafervlm import confirmation as c
from wafervlm.core import sha256_file, write_json, write_jsonl


class RepeatContinuation(unittest.TestCase):
    def setUp(self):
        f = self.fixture = fixture_module.ConfirmationTests()
        f.setUp(); self.addCleanup(f.doCleanups)
        records, self.reference = f.repeat_fixture()
        self.previous = f.root / 'previous'; self.previous.mkdir()
        self.outputs, self.rows = {}, []
        for seed in (17, 42, 123):
            for step in (200, 250):
                cp = f.checkpoint(seed, step)
                self.rows.append(dict(seed=seed, step=step, checkpoint=str(cp),
                                      files={'adapter_model.safetensors': sha256_file(cp / 'adapter_model.safetensors')}))
        self.receipt = dict(reused=[], new_outputs=[])
        for index, r in enumerate(records):
            name = f"seed{r['seed']}-step{r['step']}-repeat{r['repeat']}"
            path = Path(r['output'])
            run = json.loads((path / 'run.json').read_text())
            run['adapter_sha256'] = self.rows[0]['files']['adapter_model.safetensors']
            write_json(path / 'run.json', run); c.seal_output(path)
            self.outputs[name] = path
            if index < 14:
                shutil.copytree(path, self.previous / name)
                self.receipt['reused'].append(dict(output=name))
            else:
                self.receipt['new_outputs'].append(name)
        write_jsonl(f.data / 'validation.jsonl', self.reference)
        self.out = f.root / 'resumed'
        self.endpoints = f.root / 'endpoints.json'; write_json(self.endpoints, self.rows)

    def run_resume(self, fail=False):
        executed = []

        def evaluate(args, deadline, log):
            dest = Path(args[args.index('--output') + 1])
            executed.append(dest.name)
            self.assertIn(dest.name, self.receipt['new_outputs'])
            if fail:
                raise subprocess.TimeoutExpired(args, 1)
            shutil.copytree(self.outputs[dest.name], dest)

        with patch.object(resume, 'verify_source', return_value=(self.previous, self.receipt)), \
             patch.object(c, 'check_dataset', return_value=self.reference), \
             patch.object(c, 'prepare_endpoints', return_value=self.rows), \
             patch.object(c, 'run_process', side_effect=evaluate) as original:
            try:
                report = resume.resume_validation(self.fixture.data, self.endpoints, self.out, 'source', 1200)
            finally:
                self.assertIs(c.run_process, original)
        return report, executed

    def test_only_four_new_processes_and_all_18_passes_recomputed(self):
        report, executed = self.run_resume()
        self.assertEqual(executed, self.receipt['new_outputs'])
        self.assertEqual(len(json.loads((self.out / 'passes.json').read_text())), 18)
        self.assertEqual(len(report['environments']), 18)
        for old in self.receipt['reused']:
            name = old['output']
            self.assertEqual(sha256_file(self.out / name / 'checksums.json'),
                             sha256_file(self.previous / name / 'checksums.json'))
        c.verify_tree(self.out)

    def test_second_timeout_stays_failed_and_preserves_completed_prefix(self):
        with self.assertRaises(subprocess.TimeoutExpired):
            self.run_resume(fail=True)
        status = json.loads((self.out / 'status.json').read_text())
        self.assertEqual(status, dict(status='failed', error_type='TimeoutExpired', completed_passes=14))
        self.assertFalse((self.out / 'summary.json').exists())
        c.verify_tree(self.out)

    def test_sixteen_verified_passes_execute_only_the_final_two_processes(self):
        for name in self.receipt['new_outputs'][:2]:
            shutil.copytree(self.outputs[name], self.previous / name)
            self.receipt['reused'].append(dict(output=name))
        self.receipt['new_outputs'] = self.receipt['new_outputs'][2:]
        report, executed = self.run_resume()
        self.assertEqual(executed, ['seed123-step250-repeat2', 'seed123-step250-repeat3'])
        self.assertEqual(len(report['environments']), 18)
        self.assertEqual(len(json.loads((self.out/'passes.json').read_text())), 18)

    def test_finish_requires_reconciled_16_pass_receipt_and_preserves_aggregate_caps(self):
        from runpod.vlm_repeat_capacity import PIN
        pin = resume.finish_pin()
        ledger = dict(usd=9.472521519981784, seconds=44333.58444801331,
                      resume_source_run_id=pin['run_id'], resume_capacity_run_id=PIN['run_id'])
        with patch.dict(os.environ, WAFER_VLM_PHASE='repeat', WAFER_VLM_RESUME='1',
                        WAFER_VLM_RESUME_RETRY='1', WAFER_VLM_FINISH='1'):
            q = controller.quote(1.19, ledger)
            self.assertEqual(q['work_seconds'],2400)
            self.assertEqual(q['cleanup_seconds'],600)
            self.assertLess(q['planned_total_usd'],10.48)
            self.assertLess(q['planned_total_seconds'],790*60)
            self.assertEqual(resume.transfer_pin()['artifact_id'],pin['result_artifact_id'])
            for change in (dict(ledger,resume_source_run_id=resume.PIN['run_id']),
                           dict(ledger,resume_capacity_run_id=0),dict(ledger,seconds=64000)):
                with self.assertRaises(ValueError):controller.quote(1.19,change)

    def test_untrusted_partial_rejected_before_any_inference(self):
        bad = self.fixture.root / 'bad.zip'; bad.write_bytes(b'not the reviewed artifact')
        with patch.object(c, 'run_process') as process, self.assertRaisesRegex(ValueError, 'checksum'):
            resume.resume_validation(self.fixture.data, self.endpoints, self.out, bad, 1200)
        process.assert_not_called()

    def test_resume_reserves_30_plus_10_minutes_within_existing_total(self):
        ledger = dict(usd=8.979243733782864, seconds=41834.144988689426,
                      repeat_seconds=3391.886188030242, resume_source_run_id=resume.PIN['run_id'])
        with patch.dict(os.environ, WAFER_VLM_PHASE='repeat', WAFER_VLM_RESUME='1'):
            quote = controller.quote(.89, ledger)
            self.assertEqual(quote['work_seconds'], 1800)
            self.assertEqual(quote['cleanup_seconds'], 600)
            self.assertLess(quote['planned_total_usd'] - ledger['usd'], .60)
            for changed in (dict(ledger, resume_source_run_id=0), dict(ledger, seconds=65000), dict(ledger, usd=29)):
                with self.assertRaises(ValueError):
                    controller.quote(.89, changed)

    def test_capacity_recovery_deducts_failed_elapsed_time_without_resetting_caps(self):
        from runpod.vlm_repeat_capacity import PIN
        ledger=dict(usd=9.015288746456742,seconds=41978.496886882785,
                    resume_source_run_id=resume.PIN['run_id'],resume_capacity_run_id=PIN['run_id'],
                    resume_retry_seconds=144.35189819335938)
        with patch.dict(os.environ,WAFER_VLM_PHASE='repeat',WAFER_VLM_RESUME='1',WAFER_VLM_RESUME_RETRY='1'):
            q=controller.quote(.69,ledger)
            self.assertEqual(q['work_seconds'],2555)
            self.assertLess(q['planned_total_usd'],9.63)
            self.assertLess(q['planned_total_seconds'],41834.144988689426+3300)
            for change in (dict(ledger,resume_retry_seconds=1600),dict(ledger,resume_capacity_run_id=0),
                           dict(ledger,resume_retry_seconds=float('nan'))):
                with self.assertRaises(ValueError):controller.quote(.69,change)


if __name__ == '__main__':
    unittest.main()
