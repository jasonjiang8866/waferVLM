import unittest
from runpod.confirmation_budget import estimate
from runpod import controller


class ConfirmationBudgetTests(unittest.TestCase):
    def test_envelope_keeps_cleanup_storage_and_consumed_marker(self):
        for rate in (.5,.69,.89,1.19):
            result=estimate(rate)
            self.assertEqual(result['additional_minutes'],460)
            self.assertAlmostEqual(result['planned_total_minutes'],951.911)
            self.assertLess(result['planned_total_usd'],30)
            self.assertGreater(result['additional_usd'],rate*460/60)
            self.assertFalse(result['paid_launch_approved'])
            self.assertFalse(result['marker_reconciled_in_controller'])
        self.assertNotIn('42682266526',controller.RECONCILED_MARKERS)

    def test_invalid_planning_quote_rejected(self):
        for rate in (True,0,-1,1.2,float('inf'),float('nan')):
            with self.assertRaises(ValueError):estimate(rate)
