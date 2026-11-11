"""Frozen training and 18 fresh-process validation passes. Never evaluates holdout."""
import argparse
import json
import signal
from pathlib import Path

from wafervlm.confirmation import (inspect_checkpoint, prepare_endpoints, recipe,
                                  repeat_validation, summarize_repeats, train_seed, verify_tree)
from wafervlm.core import read_jsonl, sha256_file, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    train = sub.add_parser('train')
    train.add_argument('--seed', required=True, type=int, choices=(42, 123))
    train.add_argument('--resume-from')
    repeat = sub.add_parser('repeat')
    repeat.add_argument('--endpoints', required=True)
    freeze = sub.add_parser('freeze', help='Hash-bind all six full checkpoints before evaluation')
    analyze = sub.add_parser('analyze', help='Recompute a retrieved validation package without images or GPU')
    analyze.add_argument('--results', required=True)
    analyze.add_argument('--output', required=True)
    for seed in recipe()['seeds']:
        for step in recipe()['endpoints']:
            freeze.add_argument(f'--seed{seed}-step{step}', required=True)
    for p in (train, repeat, freeze):
        p.add_argument('--dataset', required=True)
        p.add_argument('--output', required=True)
    for p in (train, repeat):
        p.add_argument('--timeout-seconds', type=int)
    args = vars(parser.parse_args())
    mode = args.pop('mode')
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Confirmation interrupted; reap child and preserve evidence')
    signal.signal(signal.SIGTERM, interrupted)
    if mode == 'train':
        train_seed(**args)
    elif mode == 'repeat':
        args['endpoints'] = json.loads(Path(args['endpoints']).read_text())
        repeat_validation(**args)
    elif mode == 'freeze':
        output, dataset = args.pop('output'), args.pop('dataset')
        if Path(output).exists():
            raise ValueError('Refusing to replace a frozen endpoint matrix')
        endpoints = [inspect_checkpoint(args[f'seed{s}_step{t}'], dataset, s, t)
                     for s in recipe()['seeds'] for t in recipe()['endpoints']]
        write_json(output, prepare_endpoints(endpoints, dataset))
    else:
        root = Path(args['results'])
        verify_tree(root)
        if sha256_file(root/'validation-reference.jsonl') != recipe()['validation_sha256']:
            raise ValueError('Validation reference identity mismatch')
        report = summarize_repeats(json.loads((root/'passes.json').read_text()),
                                   read_jsonl(root/'validation-reference.jsonl'), base=root)
        if Path(args['output']).exists() or Path(args['output']).resolve().is_relative_to(root.resolve()):
            raise ValueError('Analysis output must be a new file outside the immutable result package')
        write_json(args['output'], report)


if __name__ == '__main__':
    main()
