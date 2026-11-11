"""Stage both released seed-17 full states from pinned ZIPs; no inference or data download."""
import argparse
import json
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from wafervlm.core import sha256_file, write_json
from wafervlm.confirmation import recipe, verify_tree
from wafervlm.resume import REQUIRED

RELEASES = {
    200: dict(tag='checkpoint-200-v1', run=42502913001,
              zip_sha='b039ca64d37b2d343a0f94da5e26e741e3b463ee48af2ab5877659c555841490',
              inner_sha='eb83e251e3a27a59ae643d89f7f98a7c620b6b225fd7e9926a4190c6f7226508',
              prefix='results/continuation/step200/'),
    250: dict(tag='continuation-300-v1', run=42682266526,
              zip_sha='32024461ade6dcd34398e96a78b537c5a274e9da636ebda9a655b6618e32a250',
              inner_sha='79aa2a1cca663e3023ef40f1b20592f52de80529b589e71dbb3ec301614bec93',
              prefix='results/extension/step250/'),
}


def prepare(archive, output, step):
    spec = RELEASES[step]
    if sha256_file(archive) != spec['zip_sha']:
        raise ValueError('Released ZIP checksum mismatch')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory() as tmp:
        inner = Path(tmp)/'experiment.tar.gz'
        with zipfile.ZipFile(archive) as z:
            entries = [m for m in z.infolist() if m.filename == 'experiment.tar.gz']
            if len(entries) != 1 or entries[0].file_size > 2*1024**3:
                raise ValueError('Invalid inner archive')
            with z.open(entries[0]) as source, inner.open('wb') as target:
                shutil.copyfileobj(source, target)
        if sha256_file(inner) != spec['inner_sha']:
            raise ValueError('Released inner checksum mismatch')
        total = 0
        seen = set()
        with tarfile.open(inner, 'r|gz') as tar:
            for member in tar:
                if not member.name.startswith(spec['prefix']) or member.isdir():
                    continue
                rel = PurePosixPath(member.name[len(spec['prefix']):])
                total += member.size
                if (not member.isfile() or rel.is_absolute() or '..' in rel.parts
                        or str(rel) in seen or total > 1024**3 or member.size > 100_000_000):
                    raise ValueError('Unsafe released checkpoint member')
                seen.add(str(rel))
                dest = output/rel;dest.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, dest.open('wb') as target:
                    shutil.copyfileobj(source, target)
    verify_tree(output)
    cp = output/f'checkpoints/checkpoint-{step}'
    if (not REQUIRED <= {p.name for p in cp.iterdir()}
            or json.loads((cp/'trainer_state.json').read_text())['global_step'] != step
            or sha256_file(cp/'adapter_model.safetensors') != recipe()['seed17_adapter_sha256'][str(step)]):
        raise ValueError('Incomplete or changed released state')
    return {'seed':17, 'step':step, 'release':spec['tag'], 'zip_sha256':spec['zip_sha'],
            'archive_sha256':spec['inner_sha'], 'checkpoint':str(cp), 'files_verified':len(seen)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step200-archive', type=Path)
    parser.add_argument('--step250-archive', type=Path)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--output', required=True, type=Path)
    args=parser.parse_args()
    if not args.download and (args.step200_archive is None or args.step250_archive is None):
        parser.error('Provide both archive paths or --download')
    args.output.mkdir(parents=True, exist_ok=False)
    evidence=[]
    for step, spec in RELEASES.items():
        archive=getattr(args,f'step{step}_archive')
        with tempfile.TemporaryDirectory() as tmp:
            if args.download:
                archive=Path(tmp)/'release.zip'
                url=f"https://github.com/jasonjiang8866/waferVLM/releases/download/{spec['tag']}/wafervlm-run-{spec['run']}.zip"
                with urllib.request.urlopen(url,timeout=60) as source, archive.open('wb') as target:
                    shutil.copyfileobj(source,target)
            evidence.append(prepare(archive,args.output/f'seed17-step{step}',step))
    write_json(args.output/'released-inputs.json',evidence)
    print(json.dumps(evidence,indent=2))


if __name__ == '__main__':
    main()
