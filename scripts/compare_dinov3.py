"""Render a paired Qwen/DINOv3 validation report or summarize six fixed-recipe runs."""
import argparse
import json

from wafervlm.vision_comparison import compare, summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    one = commands.add_parser("compare")
    one.add_argument("--dinov3-result", required=True)
    many = commands.add_parser("summarize")
    many.add_argument("--results", nargs=6, required=True)
    for sub in (one, many):
        sub.add_argument("--dataset", required=True)
        sub.add_argument("--output", required=True)
    args = vars(p.parse_args())
    command = args.pop("command")
    print(json.dumps((compare if command == "compare" else summarize)(**args), indent=2))


if __name__ == "__main__":
    main()
