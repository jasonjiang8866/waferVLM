"""Q2 allocation order and matched-pair exposure invariants."""
import json
import random
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from qwen_q2_pilot import pair_epoch


class MatchedEpoch(unittest.TestCase):
    def test_short_last_update_and_one_to_one_exposure(self):
        pairs = json.loads((ROOT / "docs/evidence/qwen-validation-area/training-pairs.json").read_text())
        self.assertEqual(Counter(p["peer_label"] for p in pairs), {"center": 13, "scratch": 12, "edge-loc": 12})
        ids = [p[k] for p in pairs for k in ("anchor", "peer")] + [str(i) for i in range(716)]
        for seed in (0, 17, 42):
            control = random.Random(seed).sample(ids, k=len(ids))
            matched = pair_epoch(control, pairs)
            self.assertCountEqual(control, matched)
            updates = set()
            for p in pairs:
                a, b = map(matched.index, (p["anchor"], p["peer"]))
                self.assertEqual(a + 1, b)
                self.assertEqual(a // 8, b // 8)
                updates.add(a // 8)
            self.assertEqual(len(updates), 37)
            remainder = set(str(i) for i in range(716))
            self.assertEqual([i for i in control if i in remainder], [i for i in matched if i in remainder])
        with self.assertRaises(ValueError):
            pair_epoch(ids, pairs + pairs[:1])
