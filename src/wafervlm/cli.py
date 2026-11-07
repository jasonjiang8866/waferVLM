import argparse
import json

from .core import audit_dataset
from .modeling import MODEL_REVISION


def main():
    parser = argparse.ArgumentParser(description="Qwen3.5-0.8B wafer-map experiment")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--trust-pickle", action="store_true")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--train-cap", type=int, default=100)
    p.add_argument("--eval-cap", type=int, default=20)
    p.add_argument("--conflict-policy", choices=("error", "exclude"), default="error")
    p = commands.add_parser("audit")
    p.add_argument("--dataset", required=True)
    for name in ("train", "evaluate"):
        p = commands.add_parser(name)
        p.add_argument("--dataset", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--revision", default=MODEL_REVISION)
        p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        if name == "train":
            p.add_argument("--steps", type=int, default=100)
            p.add_argument("--seed", type=int, default=17)
            p.add_argument("--accumulation", type=int, default=8)
            p.add_argument("--rank", type=int, default=8)
            p.add_argument("--learning-rate", type=float, default=0.0002)
            p.add_argument("--max-length", type=int, default=1024)
        else:
            p.add_argument("--split", choices=("validation", "test"), default="test")
            p.add_argument("--adapter")
            p.add_argument("--max-new-tokens", type=int, default=16)
    p = commands.add_parser("compare")
    p.add_argument("--baseline", required=True)
    p.add_argument("--tuned", required=True)
    p.add_argument("--output", required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "prepare":
        from .data import prepare
        args["expected_sha256"] = args.pop("sha256")
        result = prepare(**args)
    elif command == "audit":
        result = {k: len(v) for k, v in audit_dataset(args["dataset"]).items()}
    else:
        from . import experiment
        result = getattr(experiment, command)(**args)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
