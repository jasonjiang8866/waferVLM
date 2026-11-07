"""Bounded worker workload; provisioning/billing belongs to an external controller."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from wafervlm.core import audit_dataset, write_json
from wafervlm.experiment import seal_output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--profile", choices=("smoke", "baseline"), default="smoke")
    p.add_argument("--timeout-seconds", type=int, default=1800)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = p.parse_args()
    if not 1 <= args.timeout_seconds <= 7200:
        p.error("timeout-seconds must be between 1 and 7200")
    audit_dataset(args.dataset)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    config_path = Path(__file__).resolve().parents[1] / "configs" / f"{args.profile}.json"
    config = json.loads(config_path.read_text())
    split = config.pop("evaluation_split")
    write_json(root / "workload.json", dict(profile=args.profile, config=config, split=split,
                                           timeout_seconds=args.timeout_seconds))
    started = time.monotonic()

    def run(*parts):
        remaining = args.timeout_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Worker deadline exceeded")
        subprocess.run([sys.executable, "-m", "wafervlm.cli", *map(str, parts)], check=True, timeout=remaining)

    try:
        common = ["--dataset", args.dataset, "--device", args.device]
        run("evaluate", *common, "--split", split, "--output", root / "base")
        options = [x for k, v in config.items() for x in ("--" + k.replace("_", "-"), str(v))]
        run("train", *common, *options, "--output", root / "train")
        if split == "test":
            run("evaluate", *common, "--split", "validation", "--adapter", root / "train/adapter", "--output", root / "validation")
        run("evaluate", *common, "--split", split, "--adapter", root / "train/adapter", "--output", root / "tuned")
        run("compare", "--baseline", root / "base", "--tuned", root / "tuned", "--output", root / "comparison")
        write_json(root / "status.json", {"status": "completed", "quality_accepted": False,
            "note": "Review comparison.json. Completion alone is not evidence of quality improvement.",
            "elapsed_seconds": time.monotonic() - started})
    except BaseException as exc:
        write_json(root / "status.json", {"status": "failed", "error_type": type(exc).__name__,
                                         "elapsed_seconds": time.monotonic() - started})
        raise
    finally:
        seal_output(root)


if __name__ == "__main__":
    main()
