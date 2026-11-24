"""Exercise the operator's receipt with a local gh stub and the actual shell/jq pipeline."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/qwen_q2_dispatch.sh"
SHA = "1" * 40
URL = "https://github.com/jasonjiang8866/waferVLM/actions/runs/17"


class DispatchReceipt(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.receipt = self.root / "receipt"
        self.calls = self.root / "calls"
        self.page = self.root / "page"
        self.env = dict(os.environ, PATH=f"{self.root}:{os.environ['PATH']}",
                        CALLS=str(self.calls), PAGE=str(self.page))
        gh = self.root / "gh"
        gh.write_text("""#!/bin/bash
set -eu
echo "$*" >> "$CALLS"
[[ "$1" == auth ]] && exit 0
cat "$PAGE"
""")
        gh.chmod(0o700)

    def invoke(self, mode, *args):
        return subprocess.run(["bash", SCRIPT, mode, self.receipt, *args], env=self.env,
                              capture_output=True, text=True)

    def test_submit_returns_exact_url_and_cannot_repeat(self):
        self.page.write_text(json.dumps({"workflow_run_id": 17, "html_url": URL}))
        run = self.invoke("submit", "main", SHA, "--paid")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(run.stdout.strip(), URL)
        request = json.loads((self.receipt / "request.json").read_text())
        self.assertIs(request["inputs"]["confirm_paid"], True)
        self.assertNotEqual(self.invoke("submit", "main", SHA, "--paid").returncode, 0)
        self.assertEqual(self.calls.read_text().count("-X POST"), 1)

    def test_empty_response_recovers_only_the_correlated_run(self):
        self.page.write_text("")
        self.assertNotEqual(self.invoke("submit", "test-branch", SHA).returncode, 0)
        body = json.loads((self.receipt / "request.json").read_text())
        self.assertIs(body["inputs"]["confirm_paid"], False)
        title = f'Qwen Q2 preflight [{body["inputs"]["request_id"]}]'
        row = dict(id=17, html_url=URL, display_title=title, status="completed", conclusion="success")
        for rows, expected in (([], False), ([row, row], False),
                               ([dict(row, display_title="unrelated"), row], True)):
            self.page.write_text(json.dumps({"workflow_runs": rows}))
            reply = self.invoke("find")
            self.assertEqual(reply.returncode == 0, expected, reply.stderr)
            if expected:
                self.assertEqual(json.loads(reply.stdout)["id"], 17)
        calls = self.calls.read_text()
        self.assertEqual(calls.count("-X POST"), 1)
        self.assertIn(f"branch=test-branch -f head_sha={SHA}", calls)

    def test_paid_development_ref_fails_before_network_access(self):
        self.assertNotEqual(self.invoke("submit", "test-branch", SHA, "--paid").returncode, 0)
        self.assertFalse(self.calls.exists())
