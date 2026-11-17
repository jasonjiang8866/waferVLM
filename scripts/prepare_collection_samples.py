"""Reproduce the byte-identical, attributed Carinthia-S preview subset."""

import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

from collect_wafer_data import ROOT, digest, plan, verify_file


def main():
    selected, _ = plan(["carinthia-s"])
    dataset = selected[0]
    archive_path = ROOT / "data/collection" / dataset["id"] / dataset["version"] / "data.zip"
    verify_file(archive_path, next(f for f in dataset["files"] if f["name"] == "data.zip"))
    root = ROOT / "data_sources/samples/carinthia-s"
    manifest = json.loads((root / "manifest.json").read_text())
    if digest(archive_path) != manifest["archive_sha256"]:
        raise ValueError("Preview archive identity changed")
    with zipfile.ZipFile(archive_path) as archive:
        rows = list(csv.DictReader(io.StringIO(archive.read("data/carinthia-s.csv").decode()), delimiter=";"))
        chosen = []
        for label in sorted({r["label"] for r in rows}):
            chosen.extend(sorted([r for r in rows if r["label"] == label], key=lambda r: r["filename"])[:2])
        if [r["filename"] for r in chosen] != [s["source_id"] for s in manifest["samples"]]:
            raise ValueError("Selection changed")
        for row, sample in zip(chosen, manifest["samples"]):
            if int(row["label"]) != sample["label"]:
                raise ValueError("Label changed")
            for kind in ("image", "mask"):
                item = sample[kind]
                member = "data/" + row[kind + "_path"]
                if member != item["source_member"] or Path(item["path"]).name != item["path"]:
                    raise ValueError("Sample path changed")
                dest = root / item["path"]
                payload = archive.read(member)
                if len(payload) != item["bytes"] or hashlib.sha256(payload).hexdigest() != item["sha256"]:
                    raise ValueError("Sample bytes changed")
                if dest.exists() and digest(dest) != item["sha256"]:
                    raise ValueError("Refusing to overwrite modified preview")
                dest.write_bytes(payload)
    print("Reproduced 12 original SEM images and 12 masks; no transformations")


if __name__ == "__main__":
    main()
