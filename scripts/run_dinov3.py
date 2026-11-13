"""Bounded DINOv3 experiments. No RunPod provisioning or holdout evaluation."""
import argparse
import json
import signal
import sys
import time
from pathlib import Path

from wafervlm.dinov3 import evaluate, prepare_weights, recipe, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight", help="Check official gated weights on CPU before renting a GPU")
    preflight.add_argument("--output", required=True)
    training = commands.add_parser("train")
    training.add_argument("--mode", choices=("linear", "lora"), required=True)
    training.add_argument("--seed", type=int, choices=(17, 42, 123), default=17)
    training.add_argument("--learning-rate", type=float)
    training.add_argument("--epochs", type=int, default=recipe()["epochs"])
    evaluation = commands.add_parser("evaluate", help="Reload selected weights; validation only, batch size one")
    evaluation.add_argument("--trained", required=True)
    for sub in (training, evaluation):
        sub.add_argument("--dataset", required=True)
        sub.add_argument("--output", required=True)
        sub.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        sub.add_argument("--timeout-seconds", type=int, default=recipe()["max_work_seconds"])
        sub.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "preflight":
        print(json.dumps(prepare_weights(**args), indent=2))
        return
    cap = args["timeout_seconds"]
    if not 1 <= cap <= recipe()["max_work_seconds"]:
        parser.error("timeout exceeds the declared per-process work cap")
    if not args.pop("child"):
        from wafervlm.confirmation import run_process
        destination = Path(args["output"])
        log = destination.parent / (destination.name + ".log")
        if destination.exists() or log.exists():
            raise ValueError("Refusing to replace previous results/logs")
        destination.parent.mkdir(parents=True, exist_ok=True)
        def interrupted(signum, frame):
            raise KeyboardInterrupt("Stop child and preserve partial results")
        signal.signal(signal.SIGTERM, interrupted)
        # The outer process enforces the cap even during loading, inference or a stuck CUDA call.
        try:
            run_process([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--child"],
                        time.monotonic() + cap, log)
        except BaseException as exc:
            if destination.is_dir():
                from wafervlm.core import write_json
                from wafervlm.experiment import seal_output
                write_json(destination / "status.json", {"status": "failed", "error_type": type(exc).__name__})
                seal_output(destination)
            raise
    else:
        if command == "evaluate":
            args.pop("timeout_seconds")
        result = (train if command == "train" else evaluate)(**args)
        print(json.dumps(result, default=str, indent=2))


if __name__ == "__main__":
    main()
