"""Reproduce the pinned Kaggle WM-811K subset; does not train or call RunPod."""

import argparse
import json
from pathlib import Path

from wafervlm.acquisition import ARCHIVE_SHA256, PICKLE_SHA256, SOURCE_PAGE, SOURCE_URL, acquire
from wafervlm.core import sha256_file, write_json
from wafervlm.data import prepare


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", default="data/source")
    p.add_argument("--output", required=True)
    p.add_argument("--trust-pickle", action="store_true")
    args = p.parse_args()
    if not args.trust_pickle:
        p.error("Review the pinned public mirror's provenance/terms, then pass --trust-pickle")
    if Path(args.output).exists():
        p.error("Output already exists; use a new directory")
    source = acquire(args.cache)
    report = prepare(source, args.output, PICKLE_SHA256, trust_pickle=True,
                     seed=17, train_cap=100, eval_cap=20, conflict_policy="exclude")
    evidence = dict(source_page=SOURCE_PAGE, download_url=SOURCE_URL,
                    archive_sha256=ARCHIVE_SHA256, pickle_sha256=PICKLE_SHA256,
                    dataset_sha256=sha256_file(Path(args.output) / "dataset.json"),
                    dataset=report, audit="passed", pretrained_model_executed=False)
    write_json(Path(args.output) / "acquisition.json", evidence)
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
