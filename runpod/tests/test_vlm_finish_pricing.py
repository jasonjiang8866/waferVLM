import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from runpod import vlm_controller as c, vlm_ledger as ledger, vlm_repeat_inputs as inputs


class FinishPricing(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        self.ledger = dict(usd=9.528627960290688, seconds=44477.96864977837,
                           finish_capacity_run_id=c.FINISH_CAPACITY_PIN['run_id'],
                           finish_capacity_seconds=144.38420176506042,
                           finish_rejected_placements=[dict(gpu='NVIDIA GeForce RTX 5090', region='FAILED-1')],
                           resume_source_run_id=999, resume_capacity_run_id=c.CAPACITY_PIN['run_id'],
                           attempts=[dict(seed=s, accepted=True) for s in (42, 123)])
        self.env = dict(WAFER_VLM_PHASE='repeat', WAFER_VLM_RESUME='1',
                        WAFER_VLM_RESUME_RETRY='1', WAFER_VLM_FINISH='1',
                        GITHUB_REPOSITORY=c.REPOSITORY, GITHUB_RUN_ID='1234')
        for p in (patch.dict(os.environ, self.env), patch.object(c, 'OUT', self.out),
                  patch.object(c, 'active_pin', return_value={'run_id': 999})):
            p.start(); self.addCleanup(p.stop)

    def preflight(self, model='5090', rate=1.39, extra=None):
        api = Mock(); api.listing.return_value = []
        available = {model: rate, **(extra or {})}

        def catalog(url):
            selected = next((m for m in available if m in url), None)
            return dict(price={'secure': available[selected] if selected else .5},
                        dataCenters=[dict(id='TEST-1', availability='HIGH')] if selected else [])

        api.call.side_effect = catalog
        with patch.object(c, 'provenance', return_value={}), \
             patch.object(ledger, 'reconcile', return_value=self.ledger), \
             patch.object(inputs, 'preflight_inputs', return_value={}), \
             patch.object(c, 'budget', return_value={}):
            try:
                return c.preflight(api)
            finally:
                self.assertTrue(all(len(call.args) == 1 for call in api.call.call_args_list))

    def test_live_5090_quote_fits_only_the_final_two_envelope(self):
        result = self.preflight()
        self.assertEqual(result['gpu'], 'NVIDIA GeForce RTX 5090')
        cost = result['cost']
        self.assertEqual((cost['work_seconds'], cost['cleanup_seconds']), (2255, 600))
        self.assertAlmostEqual(cost['planned_total_usd'] - self.ledger['usd'], (1.39 + c.DISK_HOURLY) * 2855 / 3600)
        self.assertLess(cost['planned_total_usd'], 10.64)
        self.assertLess(cost['planned_total_seconds'], 790 * 60)
        self.assertEqual(cost['approved_usd'], 30)
        self.assertEqual(cost['approved_total_seconds'], 1100 * 60)

    def test_final_slice_keeps_4090_then_5090_then_3090_preference(self):
        self.assertEqual(self.preflight(extra={'3090': .5})['gpu'], 'NVIDIA GeForce RTX 5090')
        self.assertEqual(self.preflight(extra={'4090': .89, '3090': .5})['gpu'], 'NVIDIA GeForce RTX 4090')

    def test_price_above_139_is_rejected_but_capacity_observation_survives(self):
        with self.assertRaisesRegex(RuntimeError, 'No approved GPU'):
            self.preflight(rate=1.40)
        rows = json.loads((self.out / 'preflight.json').read_text())['catalog']
        row = next(r for r in rows if '5090' in r['gpu'])
        self.assertEqual(row['status'], 'quote_rejected')
        self.assertEqual(row['rate'], 1.40)
        self.assertEqual(row['approved_rate_cap'], 1.39)
        self.assertEqual(row['regions'], ['TEST-1'])

    def test_other_model_caps_remain_enforced(self):
        for model, rate in [('4090', .90), ('3090', .51)]:
            with self.subTest(model=model), self.assertRaisesRegex(RuntimeError, 'No approved GPU'):
                self.preflight(model, rate)

    def test_recently_rejected_pair_is_excluded_without_mutating_provider(self):
        self.ledger['finish_rejected_placements'][0]['region'] = 'TEST-1'
        with self.assertRaisesRegex(RuntimeError, 'No approved GPU'):
            self.preflight()
        rows = json.loads((self.out / 'preflight.json').read_text())['catalog']
        row = next(r for r in rows if '5090' in r['gpu'])
        self.assertEqual(row['status'], 'rejected_placement_excluded')
        self.assertEqual(row['observed_regions'], ['TEST-1'])
        self.assertEqual(row['excluded_regions'], ['TEST-1'])
        self.assertEqual(row['regions'], [])
        # A different authorized GPU in that region remains eligible.
        self.assertEqual(self.preflight(extra={'4090': .89})['gpu'], 'NVIDIA GeForce RTX 4090')

    def test_missing_or_malformed_rejection_evidence_blocks(self):
        for value in (None, [], [{'gpu': 'unknown', 'region': 'TEST-1'}],
                      [{'gpu': 'NVIDIA GeForce RTX 5090', 'region': []}]):
            self.ledger['finish_rejected_placements'] = value
            with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, 'No approved GPU'):
                self.preflight()

    def test_training_and_other_continuations_keep_the_original_ceiling(self):
        for change in [dict(WAFER_VLM_PHASE='training'), dict(WAFER_VLM_RESUME='0'),
                       dict(WAFER_VLM_RESUME_RETRY='0'), dict(WAFER_VLM_FINISH='0')]:
            with self.subTest(change=change), patch.dict(os.environ, change):
                self.assertEqual(c.gpu_price_caps()['NVIDIA GeForce RTX 5090'], 1.19)
                with self.assertRaises(ValueError): c.quote(1.39, self.ledger)
        self.assertEqual(c.GPU_CAPS['NVIDIA GeForce RTX 5090'], 1.19)

    def test_changed_rate_cannot_bypass_receipts_or_campaign_caps(self):
        for change in [dict(usd=29), dict(seconds=64000), dict(resume_source_run_id=0),
                       dict(resume_capacity_run_id=0), dict(finish_capacity_run_id=0),
                       dict(finish_capacity_seconds=1201), dict(finish_capacity_seconds=float('nan'))]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                c.quote(1.39, dict(self.ledger, **change))
        for rate in [float('nan'), float('inf'), 0, 1.40]:
            with self.subTest(rate=rate), self.assertRaises(ValueError): c.quote(rate, self.ledger)


if __name__ == '__main__':
    unittest.main()
