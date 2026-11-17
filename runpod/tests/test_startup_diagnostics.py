import collections
import io
import json
import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from runpod import startup_diagnostics as d


class Response(io.BytesIO):
    headers = SimpleNamespace(get_content_type=lambda: "text/event-stream")


class StartupDiagnosticsTests(unittest.TestCase):
    def test_only_fixed_categories_leave_log_parser(self):
        counts = collections.Counter()
        d.add_frame(json.dumps({"source": "system", "line":
            "pulling private-image TOKEN=secret-value https://private/path"}), counts)
        self.assertEqual(dict(counts), {"system:lines": 1, "system:image_pull": 1})
        self.assertNotIn("secret", json.dumps(counts))
        self.assertFalse(d.add_frame(b'[]', counts))
        self.assertFalse(d.add_frame(b'{"source":"secret","line":"error"}', counts))

    def test_runtime_never_serializes_env_ip_or_unrecognized_status(self):
        result = d.runtime_summary({"status": "secret-value", "env": {"KEY": "secret"},
            "runtime": {"uptime": 3, "cpu": {"util": 12}, "memory": {"util": float("nan")},
                        "gpus": [{"util": 0}, {"util": -1}], "ip": "private"}})
        self.assertEqual(result, {"status": "unknown", "runtime_reported": True,
            "uptime_seconds": 3, "cpu_util_percent": 12, "gpu_util_percent": [0]})

    def test_sse_read_uses_documented_get_and_bounds_frame_shape(self):
        data = b'data: {"source":"container","line":"ModuleNotFoundError secret"}\n\n'
        with patch.object(d, "Runpod", return_value=SimpleNamespace(key="test-only")), \
             patch.object(d.urllib.request, "build_opener") as build:
            build.return_value.open.return_value = Response(data)
            result = d.read_logs("owned-pod")
            req = build.return_value.open.call_args.args[0]
            self.assertEqual(req.get_method(), "GET")
            self.assertEqual(req.full_url, d.API + "/pods/owned-pod/logs?tail=80")
            self.assertEqual(result["categories"]["container:python_import"], 1)
            self.assertNotIn("secret", json.dumps(result))

    def test_oversized_line_and_invalid_id_do_not_escape(self):
        with patch.object(d, "Runpod", return_value=SimpleNamespace(key="test-only")), \
             patch.object(d.urllib.request, "build_opener") as build:
            build.return_value.open.return_value = Response(b"x" * 9000)
            self.assertEqual(d.read_logs("owned")["status"], "byte_limit")
            build.reset_mock()
            self.assertEqual(d.read_logs("../another")["status"], "invalid_id")
            build.assert_not_called()

    def test_timeout_is_evidence_unavailable_and_not_root_cause(self):
        with patch.object(d.subprocess, "run", side_effect=subprocess.TimeoutExpired("diag", 12)):
            result = d.capture("owned", {"status": "RUNNING", "runtime": None}, {"kind": "http", "status": 502})
        self.assertFalse(result["root_cause_verified"])
        self.assertEqual(result["logs"], {"status": "diagnostic_unavailable"})
        self.assertFalse(result["runtime"]["runtime_reported"])

    def test_child_failure_does_not_expose_stderr(self):
        with patch.object(d.subprocess, "run", return_value=SimpleNamespace(
                returncode=1, stdout=b"", stderr=b"PRIVATE ENV")) as run:
            result = d.capture("owned", {"status": "RUNNING"})
            self.assertEqual(run.call_args.kwargs["timeout"], 12)
        self.assertEqual(result["logs"]["status"], "diagnostic_process_failed")
        self.assertNotIn("PRIVATE", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
