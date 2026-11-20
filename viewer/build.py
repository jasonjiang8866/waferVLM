"""Build the public microscope from attributed Git previews. No network required."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import mimetypes
import shutil

HERE = Path(__file__).resolve().parent
SAMPLES = HERE.parent / "data_sources" / "samples"


def build(output, standalone):
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in ("carinthia-s", "silicon-wafer-pl"):
        folder = SAMPLES / dataset
        manifest = json.loads((folder / "manifest.json").read_text())
        if manifest["license"] != "CC-BY-4.0":
            raise ValueError(f"Review license for {dataset}")

        def asset(filename, checksum):
            if Path(filename).name != filename:
                raise ValueError("Expected a local sample filename")
            data = (folder / filename).read_bytes()
            if hashlib.sha256(data).hexdigest() != checksum:
                raise ValueError(f"Checksum mismatch for {filename}")
            if standalone:
                encoded = base64.b64encode(data).decode("ascii")
                return f"data:{mimetypes.guess_type(filename)[0]};base64,{encoded}"
            rel = Path("samples") / dataset / filename
            (output / rel).parent.mkdir(parents=True, exist_ok=True)
            (output / rel).write_bytes(data)
            return rel.as_posix()

        if dataset == "carinthia-s":
            for item in manifest["samples"]:
                row = dict(id=item["source_id"], label=item["label"], dataset=dataset)
                for key in ("image", "mask"):
                    row[key] = asset(item[key]["path"], item[key]["sha256"])
                rows.append(row)
        else:
            name = manifest["source_filename"]
            rows.append(dict(id=name, dataset=dataset, image=asset(name, manifest["sha256"])))
    html = (HERE / "template.html").read_text()
    payload = json.dumps(rows, separators=(",", ":")).replace("<", r"\u003c")
    explorer = html.replace("__SAMPLE_DATA__", payload)
    if standalone:
        (output / "index.html").write_text(explorer.replace("__DAILY_LINK__", ""))
        return
    (output / "explorer.html").write_text(explorer.replace("__DAILY_LINK__", '<a href="./">Daily atlas</a>'))
    collections = json.loads((HERE / "collections.json").read_text())
    ids = [item["id"] for item in collections]
    expected = {path.stem for path in (SAMPLES.parent / "cards").glob("*.md")}
    if set(ids) != expected or len(set(ids)) != len(ids):
        raise ValueError("Each data card must have exactly one daily collection")
    for item in collections:
        if item["id"] == "carinthia-s":
            images = [
                dict(src=row["image"], title=f"Class {row['label']} · {row['id']}",
                     source="https://zenodo.org/records/16895427",
                     credit="Corinna Kofler and Vahidin Hasić · CC BY 4.0",
                     license="https://creativecommons.org/licenses/by/4.0/",
                     note="Original SEM. Carinthia-S annotates the same image family as Carinthia.")
                for row in rows if row["dataset"] == item["id"]
            ]
        else:
            manifest_path = HERE / "daily-data" / (item["id"] + ".json")
            images = json.loads(manifest_path.read_text())
        if not images or len({(row["src"], tuple(row.get("crop", ()))) for row in images}) != len(images):
            raise ValueError(f"Empty or duplicate previews for {item['id']}")
        for row in images:
            if not all(row.get(key) for key in ("src", "title", "credit", "source")) or not row["source"].startswith("https://"):
                raise ValueError(f"Incomplete attribution for {item['id']}")
            src = row["src"]
            if src.startswith("https://") or (item["id"] == "carinthia-s" and (output / src).is_file()):
                continue
            rel = Path(src)
            if len(rel.parts) != 2 or rel.parts[0] != "daily-assets":
                raise ValueError(f"Invalid preview path: {src}")
            (output / rel.parent).mkdir(exist_ok=True)
            preview = HERE / rel
            if row.get("sha256") and hashlib.sha256(preview.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError(f"Preview checksum mismatch: {src}")
            shutil.copyfile(preview, output / rel)
        item["images"] = images
    daily = (HERE / "daily.html").read_text()
    catalog = json.dumps(collections, separators=(",", ":")).replace("<", r"\u003c")
    (output / "index.html").write_text(daily.replace("__DAILY_DATA__", catalog))
    for filename in ("daily.css", "daily.js"):
        shutil.copyfile(HERE / filename, output / filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--standalone", action="store_true", help="Embed previews for offline viewing")
    args = parser.parse_args()
    build(args.output, args.standalone)
