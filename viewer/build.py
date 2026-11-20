"""Build the public microscope from attributed Git previews. No network required."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import mimetypes

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
    (output / "index.html").write_text(html.replace("__SAMPLE_DATA__", payload))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--standalone", action="store_true", help="Embed previews for offline viewing")
    args = parser.parse_args()
    build(args.output, args.standalone)
