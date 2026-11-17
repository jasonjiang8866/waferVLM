import argparse
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from runpod import storage_import as m

SOURCE = 'a' * 40
PREFIX = m.identity('123', '1')
CPU = {'id': 'cpu3c', 'compute_usd_per_hour': 0.06, 'availability': 'HIGH'}


def pod():
    p = m.payload(PREFIX, SOURCE, 'import', 'gsb_one', CPU)
    p.update(id='owned', cost=0.06, status='RUNNING')
    return p


class CleanupAPI:
    def __init__(self, rows):
        self.rows, self.deleted = rows, []

    def listing(self, path, key):
        return self.rows

    def call(self, path, *, method, body=None):
        assert method == 'DELETE'
        self.deleted.append(path)
        self.rows = [p for p in self.rows if path != '/pods/' + p['id']]


class StorageImportTests(unittest.TestCase):
    def test_missing_approval_blocks_before_api(self):
        args = argparse.Namespace(confirm_paid=False, approved_budget_usd=1)
        with self.assertRaises(ValueError):
            m.execute(None, args)
        for value in (0, .49, float('nan'), 2):
            args.confirm_paid, args.approved_budget_usd = True, value
            with self.assertRaises(ValueError):
                m.execute(None, args)

    def test_volume_selection_never_guesses_or_creates(self):
        self.assertEqual(m.select_volume([{'id': 'one'}], ''), 'one')
        self.assertEqual(m.select_volume([{'id': 'one'}, {'id': 'two'}], 'two'), 'two')
        for rows, request in [([], ''), ([{'id': 'one'}, {'id': 'two'}], ''), ([{'id': 'one'}], 'missing')]:
            with self.assertRaises(ValueError):
                m.select_volume(rows, request)

    def test_actual_mount_price_and_owner_are_checked(self):
        m.verify_pod(pod(), PREFIX, SOURCE, 'import', 'gsb_one')
        for field, value in [('cost', .11), ('mounts', {'global': [{'volumeId': 'wrong', 'path': m.MOUNT}]}),
                             ('image', 'unreviewed'), ('env', {})]:
            p = pod(); p[field] = value
            with self.assertRaises(ValueError):
                m.verify_pod(p, PREFIX, SOURCE, 'import', 'gsb_one')

    def test_cleanup_ignores_other_runs_and_spoofed_names(self):
        other = pod(); other['id'] = 'other'; other['env'] = {}
        api = CleanupAPI([pod(), other])
        result = m.cleanup(api, PREFIX, SOURCE)
        self.assertEqual(api.deleted, ['/pods/owned'])
        self.assertEqual(api.rows, [other])
        self.assertEqual(result['status'], 'cleanup_verified')

    def test_receipt_integrity_rejects_corruption_and_missing_files(self):
        selected, _ = m.plan(m.DATASETS)
        rows = []
        for d in selected:
            rows.append({'id': d['id'], 'version': d['version'], 'volume_id': 'gsb_one',
                         'mount_path': m.MOUNT, 'dataset_sha256': m.dataset_digest(d),
                         'files': [{**{k: f[k] for k in ('name', 'bytes', 'sha256')},
                                    'publisher_md5': f['md5']} for f in d['files']]})
        result = {'status': 'readback_verified', 'receipts': rows}
        m.validate_result(result, selected, 'verify', 'gsb_one')
        bad = copy.deepcopy(result); bad['receipts'][0]['files'][0]['sha256'] = 'wrong'
        with self.assertRaises(ValueError): m.validate_result(bad, selected, 'verify', 'gsb_one')
        bad = copy.deepcopy(result); bad['receipts'].pop()
        with self.assertRaises(ValueError): m.validate_result(bad, selected, 'verify', 'gsb_one')

    def test_secret_free_bundle_and_payload(self):
        with patch.dict(os.environ, {'RUNPOD_API_KEY': 'SECRET'}):
            self.assertNotIn('SECRET', json.dumps(pod()))
            self.assertLess(len(m.bundle()), 16000)
        self.assertFalse(pod()['ports'])
        self.assertNotIn('gpu', pod())

    def test_new_pod_exec_404_can_recover_without_recreating_or_writing(self):
        class API:
            def call(self, path):
                self.last_path = path
                return pod()
        api, diagnostics = API(), {}
        with patch.object(m, 'exec_python', side_effect=[m.RemoteHTTPError(404, []), 'ready']) as probe, patch.object(m.time, 'sleep'):
            result = m.wait_ready(api, 'owned', 'import', PREFIX, SOURCE, 'gsb_one', float('inf'), diagnostics)
        self.assertEqual(result['id'], 'owned')
        self.assertEqual(diagnostics['readiness_http_statuses'], [404])
        self.assertEqual([c.args[2] for c in probe.call_args_list], ["print('ready')"] * 2)
        self.assertEqual(api.last_path, '/pods/owned')

    def test_exec_readiness_is_bounded_and_never_retries_auth_errors(self):
        class API:
            def call(self, path): return pod()
        for code in (401, 403):
            with patch.object(m, 'exec_python', side_effect=m.RemoteHTTPError(code, [])) as probe:
                with self.assertRaises(m.RemoteHTTPError):
                    m.wait_ready(API(), 'owned', 'import', PREFIX, SOURCE, 'gsb_one', float('inf'), {})
                self.assertEqual(probe.call_count, 1)
        with patch.object(m.time, 'monotonic', side_effect=[0, 0, 121]), patch.object(m.time, 'sleep'), patch.object(m, 'exec_python', side_effect=m.RemoteHTTPError(404, [])) as probe:
            with self.assertRaises(TimeoutError):
                m.wait_ready(API(), 'owned', 'import', PREFIX, SOURCE, 'gsb_one', float('inf'), {})
            self.assertEqual(probe.call_count, 1)

    def test_two_sequential_stages_and_cleanup_failure_cannot_be_accepted(self):
        class API:
            def listing(self, path, key):
                return [{'id': 'gsb_one'}] if path == '/global-volumes' else []
        for cleanup_error in (False, True):
            with self.subTest(cleanup_error=cleanup_error), tempfile.TemporaryDirectory() as tmp:
                args = argparse.Namespace(confirm_paid=True, approved_budget_usd=.5, run_id='123',
                                          attempt='1', volume_id='', output=Path(tmp))
                clean_results = [{'status': 'cleanup_verified'}, {'status': 'cleanup_verified'},
                                 RuntimeError('cleanup unavailable') if cleanup_error else {'status': 'cleanup_verified'}]
                with patch.dict(os.environ, {'GITHUB_SHA': SOURCE}), patch.object(m, 'observe', return_value={'cpu_candidates': [CPU]}), patch.object(m, 'budget'), patch.object(m, 'stage', return_value={'checked': True}) as stages, patch.object(m, 'cleanup', side_effect=clean_results):
                    if cleanup_error:
                        with self.assertRaises(RuntimeError): m.execute(API(), args)
                    else:
                        self.assertEqual(m.execute(API(), args)['status'], 'accepted')
                    self.assertEqual([call.args[1] for call in stages.call_args_list], ['import', 'verify'])
                report = json.loads((Path(tmp) / 'import.json').read_text())
                self.assertEqual(report['status'], 'cleanup_unverified' if cleanup_error else 'accepted')

    def test_create_timeout_still_reconciles_cleanup_and_records_failure(self):
        class API:
            def listing(self, path, key):
                return [{'id': 'gsb_one'}] if path == '/global-volumes' else []
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(confirm_paid=True, approved_budget_usd=.5, run_id='123',
                                      attempt='1', volume_id='', output=Path(tmp))
            with patch.dict(os.environ, {'GITHUB_SHA': SOURCE}), patch.object(m, 'observe', return_value={'cpu_candidates': [CPU]}), patch.object(m, 'budget'), patch.object(m, 'stage', side_effect=TimeoutError('unknown create')), patch.object(m, 'cleanup', return_value={'status': 'cleanup_verified'}) as clean:
                with self.assertRaises(TimeoutError): m.execute(API(), args)
                clean.assert_called_once()
            result = json.loads((Path(tmp) / 'import.json').read_text())
            self.assertEqual(result['status'], 'partial_or_failed')
            self.assertEqual(result['cleanup']['status'], 'cleanup_verified')


if __name__ == '__main__':
    unittest.main()
