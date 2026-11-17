"""Bounded startup evidence; never persist provider log text or environments.

The official runpod/runpodctl internal/api/logs.go documents the REST v2 SSE
route. Categories are observations, not a determination of the root cause.
"""

import collections
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

from runpod.provider import API, NoRedirect, Runpod, finite

PATTERNS = {
    "image_pull": r"pulling|fetching image|downloading|extracting",
    "image_ready": r"downloaded newer image|image is up to date|pull complete",
    "container_start": r"start(?:ing)? container|container started",
    "container_stop": r"stop(?:ping)? container|container exited",
    "image_auth": r"unauthorized|authentication required|pull access denied",
    "missing_executable": r"no such file or directory|executable file not found",
    "exec_format": r"exec format error",
    "disk_full": r"no space left on device",
    "out_of_memory": r"out of memory|oomkilled|oom killed",
    "python_import": r"modulenotfounderror|importerror",
    "python_syntax": r"syntaxerror|indentationerror",
    "cuda_error": r"cuda error|cuda initialization|driver version is insufficient",
    "network_error": r"connection refused|connection timed out|network is unreachable",
    "unclassified_error": r"error|failed|exception|traceback",
}


def runtime_summary(pod):
    """Explicit numeric/enum allowlist; no IDs, IPs, env, commands or raw errors."""
    status = pod.get("status")
    result = {"status": status if status in {
        "PROVISIONING", "STARTING", "RUNNING", "EXITED", "ERROR", "TERMINATED"
    } else "unknown", "runtime_reported": isinstance(pod.get("runtime"), dict)}
    runtime = pod.get("runtime")
    if isinstance(runtime, dict):
        if finite(runtime.get("uptime")):
            result["uptime_seconds"] = runtime["uptime"]
        for key in ("cpu", "memory"):
            item = runtime.get(key)
            if isinstance(item, dict) and finite(item.get("util")) and item["util"] <= 100:
                result[key + "_util_percent"] = item["util"]
        gpus = runtime.get("gpus")
        if isinstance(gpus, list):
            result["gpu_util_percent"] = [g["util"] for g in gpus[:8]
                if isinstance(g, dict) and finite(g.get("util")) and g["util"] <= 100]
    return result


def add_frame(data, counts):
    try:
        entry = json.loads(data)
    except (ValueError, UnicodeError):
        return False
    if (not isinstance(entry, dict) or entry.get("source") not in {"system", "container"}
            or not isinstance(entry.get("line"), str)):
        return False
    prefix = entry["source"] + ":"
    counts[prefix + "lines"] += 1
    for label, pattern in PATTERNS.items():
        if re.search(pattern, entry["line"].lower()):
            counts[prefix + label] += 1
    return True


def read_logs(pod_id):
    """One GET; idle/socket, byte, frame and process-level time bounds."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", pod_id):
        return {"status": "invalid_id"}
    req = urllib.request.Request(API + "/pods/" + pod_id + "/logs?tail=80",
        headers={"Authorization": "Bearer " + Runpod().key,
                 "Accept": "text/event-stream"})
    counts = collections.Counter()
    frames, received, pending = 0, 0, []
    status = "empty"
    try:
        with urllib.request.build_opener(NoRedirect).open(req, timeout=3) as response:
            if response.headers.get_content_type() != "text/event-stream":
                return {"status": "unexpected_content_type"}
            deadline = time.monotonic() + 6
            while time.monotonic() < deadline:
                line = response.readline(8193)
                received += len(line)
                if len(line) > 8192 or received > 65536:
                    status = "byte_limit"
                    break
                if not line:
                    status = "stream_ended"
                    break
                if line.strip() == b"":
                    if pending:
                        frames += 1
                        if not add_frame(b"\n".join(pending), counts):
                            counts["invalid_frames"] += 1
                        pending = []
                    if frames >= 160:
                        status = "frame_limit"
                        break
                elif line.startswith(b"data:"):
                    pending.append(line[5:].strip())
            else:
                status = "time_limit"
    except urllib.error.HTTPError as exc:
        return {"status": "http_error", "http_status": exc.code}
    except (TimeoutError, urllib.error.URLError, OSError):
        status = "idle_or_transport_timeout" if frames else "no_response_or_no_logs"
    return {"status": status, "frames": frames, "categories": dict(counts)}


def capture(pod_id, pod, proxy_error=None):
    result = {"runtime": runtime_summary(pod), "root_cause_verified": False}
    if isinstance(proxy_error, dict):
        # Caller creates this from fixed exceptions, never from response text.
        result["last_proxy_error"] = proxy_error
    try:
        proc = subprocess.run([sys.executable, "-m", "runpod.startup_diagnostics", pod_id],
            capture_output=True, timeout=12, check=False)
        if proc.returncode or len(proc.stdout) > 16000:
            result["logs"] = {"status": "diagnostic_process_failed"}
        else:
            result["logs"] = json.loads(proc.stdout)
    except (subprocess.TimeoutExpired, ValueError, OSError):
        result["logs"] = {"status": "diagnostic_unavailable"}
    return result


if __name__ == "__main__":
    try:
        result = read_logs(sys.argv[1])
    except Exception:
        result = {"status": "diagnostic_unavailable"}
    print(json.dumps(result, allow_nan=False))
