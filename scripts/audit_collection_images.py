"""Decode every Carinthia-S image/mask without extraction or label changes."""

import csv
from collections import Counter
from datetime import datetime, timezone
import io
import json
import zipfile

from PIL import Image

from collect_wafer_data import ROOT, plan, verify_file


def main():
    dataset = plan(["carinthia-s"])[0][0]
    spec = next(f for f in dataset["files"] if f["name"] == "data.zip")
    path = ROOT / "data/collection" / dataset["id"] / dataset["version"] / "data.zip"
    verification = verify_file(path, spec)
    shapes, modes = Counter(), Counter()
    grayscale_values, color_mask_ids = set(), []
    nonbinary = 0
    with zipfile.ZipFile(path) as archive:
        rows = list(csv.DictReader(io.StringIO(archive.read("data/carinthia-s.csv").decode()), delimiter=";"))
        for row in rows:
            im = Image.open(io.BytesIO(archive.read("data/" + row["image_path"])))
            mask = Image.open(io.BytesIO(archive.read("data/" + row["mask_path"])))
            im.load()
            mask.load()
            if im.size != mask.size:
                raise ValueError("Image/mask shape mismatch: " + row["filename"])
            shapes[str(im.size)] += 1
            modes[mask.mode] += 1
            if mask.mode == "L":
                values = {value for count, value in mask.getcolors(256)}
                grayscale_values.update(values)
                nonbinary += bool(values - {0, 255})
            else:
                color_mask_ids.append({"source_id": row["filename"], "mode": mask.mode})
    result = {"dataset": "carinthia-s", "verified_utc": datetime.now(timezone.utc).isoformat(),
              "image_mask_pairs_decoded": len(rows), "image_mask_dimensions_equal": True,
              "image_dimensions_counts": dict(shapes), "mask_modes": dict(modes),
              "grayscale_mask_value_range": [min(grayscale_values), max(grayscale_values)],
              "grayscale_mask_distinct_values": len(grayscale_values),
              "grayscale_masks_with_intermediate_values": nonbinary,
              "color_mask_ids": color_mask_ids,
              "class_counts": dict(sorted(Counter(row["label"] for row in rows).items())),
              "archive_bytes": verification["bytes"], "expanded_bytes": verification["expanded_bytes"],
              "archive_sha256": verification["sha256"], "global_volume_status": "not_imported"}
    output = ROOT / "data_sources/receipts/carinthia-s-image-audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
