"""Exercise the exact Q2 startup command on an empty ephemeral filesystem."""
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from runpod import q2_pilot as pilot


class WorkerBootstrap(unittest.TestCase):
    def test_no_workspace_or_network_needed_to_start_worker(self):
        sha = "c" * 40
        program = pilot.bootstrap(sha)
        self.assertEqual(len(program.splitlines()), 1)
        self.assertLess(len(shlex.quote(program)), 60 * 1024)
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            # Move only the absolute ephemeral root into this test's isolated directory.
            workspace = base / "ephemeral/workspace"
            program = program.replace("/workspace/q2-source", str(workspace / "q2-source"))
            stub = base / "sitecustomize.py"
            stub.write_text("""import pathlib, runpy, sys
def observe(path, run_name):
    assert pathlib.Path(path).is_file() and run_name == '__main__'
    assert pathlib.Path(sys.path[0]).resolve() == pathlib.Path.cwd()
    assert (pathlib.Path(sys.path[1])/'wafervlm/experiment.py').is_file()
    print('worker entry reached')
runpy.run_path = observe
""")
            cmd = "exec " + shlex.quote(sys.executable) + " -u -c " + shlex.quote(program)
            result = subprocess.run(["/bin/bash", "-lc", cmd], capture_output=True, text=True,
                                    env={**os.environ, "WAFER_CONTROLLER_SHA": sha, "PYTHONPATH": tmp}, check=True)
            self.assertIn("worker entry reached", result.stdout)
            for name in pilot.FILES:
                self.assertEqual((workspace / "q2-source" / name).read_bytes(), (pilot.ROOT / name).read_bytes())


class RecoveryBudget(unittest.TestCase):
    def test_prior_charge_and_cleanup_are_inside_both_caps(self):
        base = pilot.PRIOR_MINUTES * 60
        ledger = {"usd": pilot.PRIOR_USD + .31, "seconds": base + 1230.2}
        quote = pilot.prices(.89, .01, ledger)
        self.assertEqual(quote["work_seconds"], 4169)
        self.assertLessEqual(ledger["seconds"] - base + quote["work_seconds"] + quote["cleanup_seconds"], 6000)
        self.assertAlmostEqual(quote["planned_pilot_usd"], .31 + .9 * 4769 / 3600)
        for cost, elapsed in ((4.99, 1000), (.31, 5400), (0, 0)):
            with self.subTest(cost=cost, elapsed=elapsed), self.assertRaises(ValueError):
                pilot.prices(.89, .01, {"usd": pilot.PRIOR_USD + cost, "seconds": base + elapsed})

    def test_only_two_exact_marker_owners_are_accepted(self):
        sha = "c" * 40
        old = {"marker_id": 11, "run_id": 91, "controller_sha": "a" * 40}
        def marker(i, run, sha):
            return dict(id=i, name=pilot.MARKER, expired=False, workflow_run=dict(id=run, head_sha=sha))
        items = [marker(11, 91, old["controller_sha"]), marker(12, 92, sha)]
        listing = {"artifacts": items, "total_count": 2}
        with patch.dict(os.environ, GH_TOKEN="test", AUTHORIZATION_ARTIFACT_ID="12", EXPECTED_SHA=sha), \
                patch.object(pilot, "previous_attempt", return_value=old), \
                patch.object(pilot, "request", return_value=listing):
            pilot.guard("92")
            items[0]["workflow_run"]["head_sha"] = "d" * 40
            with self.assertRaises(RuntimeError):
                pilot.guard("92")
            listing["total_count"] = 3
            with self.assertRaises(RuntimeError):
                pilot.guard("92")


if __name__ == "__main__":
    unittest.main()
