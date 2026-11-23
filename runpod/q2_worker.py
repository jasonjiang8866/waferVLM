"""Run both Q2 arms on one GPU and expose their portable evidence through the existing worker."""
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from runpod import vlm_worker as server


def copy_evidence(source, name, result):
    if result["status"] == "accepted" or source.is_file():
        dst = server.OUT / name
        dst.parent.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(source, dst)


def work():
    result = dict(status="rejected", source_sha=os.environ["WAFER_SOURCE_SHA"],
                  controller_sha=os.environ["WAFER_CONTROLLER_SHA"], config_sha256=os.environ["WAFER_CONFIG_SHA"],
                  profiles=["q2-pair-pilot"])
    root = Path.cwd()
    output = Path("/workspace/q2-training")
    dataset = Path("/workspace/q2-data")
    try:
        server.OUT.mkdir(parents=True, exist_ok=False)
        env = dict(os.environ, WAFERVLM_SOURCE_SHA=result["controller_sha"])
        for name in ("RUNPOD_API_KEY","GH_TOKEN","GITHUB_TOKEN","WAFER_RESULT_TOKEN"):
            env.pop(name, None)
        for phase, args in (("data", ["scripts/prepare_public_data.py", "--trust-pickle", "--cache",
                                      "/workspace/q2-cache", "--output", str(dataset)]),
                            ("training", ["analysis/qwen_q2_pilot.py", "run", str(dataset), str(output)])):
            server.RESULT["phase"] = phase
            with (server.OUT / (phase + ".log")).open("w") as log:
                subprocess.run([sys.executable, "-u", *args], check=True, cwd=root, env=env,
                               stdout=log, stderr=subprocess.STDOUT,
                               timeout=max(1, float(os.environ["WAFER_DEADLINE"]) - time.time() - 150))
        result["status"] = "accepted"
    except Exception as exc:
        result.update(error_type=type(exc).__name__, phase=server.RESULT.get("phase"))
    finally:
        # Preserve completed stages even if the later arm runs out of time.
        try:
            for name in ("train.jsonl", "validation.jsonl"):
                copy_evidence(dataset / name, name, result)
            for name in ("pairs.json", "comparison.json"):
                copy_evidence(output / name, name, result)
            for arm in ("control", "matched"):
                for name in ("train/epoch-order.json", "train/training.json", "train/run.json",
                             "train/adapter/adapter_model.safetensors", "train/adapter/adapter_config.json",
                             "validation/run.json", "validation/metrics.json", "validation/predictions.jsonl"):
                    copy_evidence(output / arm / name, f"{arm}/{name}", result)
        except OSError as exc:
            result.update(status="rejected", error_type=type(exc).__name__, phase="retrieval")
        server.write("worker.json", result)
        with tarfile.open(server.ARCHIVE, "w:gz", compresslevel=1) as archive:
            archive.add(server.OUT, arcname="results")
        result["archive"] = dict(size=server.ARCHIVE.stat().st_size, sha256=server.digest(server.ARCHIVE))
        server.RESULT = result


if __name__ == "__main__":
    server.main(work)
