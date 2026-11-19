"""Create a prediction-blind plate and a separate key for the Qwen loc review."""
import argparse
import csv
import hashlib
import io
import json
import string
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from qwen_validation_errors import REFERENCE, load_predictions


def review_cases(maps, endpoints):
    cases = [row for row in maps if row["label"] == "loc" and
             sum(row[e] == "loc" for e in endpoints) in (0, len(endpoints))]
    cases.sort(key=lambda row: hashlib.sha256(("loc-review-v1/" + row["id"]).encode()).digest())
    if len(cases) != 10:
        raise ValueError("Expected ten extreme loc cases in the frozen validation run")
    return dict(zip(string.ascii_uppercase[:10], cases, strict=True))


def reference_rows():
    # load_predictions verifies this manifest against the frozen experiment config.
    return {r["id"]: r for r in map(json.loads, REFERENCE.read_text().splitlines())}


def read_image(dataset, ref):
    data = (dataset / ref["image"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != ref["image_sha256"]:
        raise ValueError(f"Image hash mismatch for {ref['id']}")
    with Image.open(io.BytesIO(data)) as wafer:
        if wafer.size != (256, 256) or wafer.mode != "RGB":
            raise ValueError(f"Unexpected rendering for {ref['id']}")
        return wafer.copy()


def make_plate(cases, dataset, output):
    reference = reference_rows()
    tile, header, gap = 256, 32, 8
    sheet = Image.new("RGB", (5 * (tile + gap) + gap, 2 * (tile + header + gap) + gap), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=21)
    for index, (letter, case) in enumerate(cases.items()):
        ref = reference[case["id"]]
        wafer = read_image(dataset, ref)
        x = gap + index % 5 * (tile + gap)
        y = gap + index // 5 * (tile + header + gap)
        draw.text((x, y + 2), letter, fill="black", font=font)
        sheet.paste(wafer, (x, y + header))
    sheet.save(output)


def write_coverage(maps, endpoints, dataset, output):
    reference = reference_rows()
    passing, failing, background = (0, 90, 180), (255, 220, 0), (0, 0, 0)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["id", "correct_of_six", "yellow_pixels", "wafer_pixels", "yellow_fraction"])
        for row in maps:
            if row["label"] != "loc":
                continue
            wafer = read_image(dataset, reference[row["id"]])
            counts = {color: count for count, color in wafer.getcolors(256 * 256)}
            if counts.keys() - {passing, failing, background}:
                raise ValueError(f"Unexpected palette for {row['id']}")
            yellow = counts.get(failing, 0)
            valid = yellow + counts.get(passing, 0)
            correct = sum(row[e] == "loc" for e in endpoints)
            writer.writerow([row["id"], correct, yellow, valid, f"{yellow / valid:.6f}"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_subparsers(dest="mode", required=True)
    plate = mode.add_parser("plate", help="Render the ten maps with letter labels only")
    plate.add_argument("dataset", type=Path, help="Directory containing the lot17 images")
    plate.add_argument("output", type=Path)
    reveal = mode.add_parser("reveal", help="Write the per-case identity and predictions")
    reveal.add_argument("output", type=Path)
    coverage = mode.add_parser("coverage", help="Measure yellow area on all validation loc maps")
    coverage.add_argument("dataset", type=Path)
    coverage.add_argument("output", type=Path)
    args = parser.parse_args()
    maps, endpoints, _ = load_predictions()
    if args.mode == "coverage":
        write_coverage(maps, endpoints, args.dataset, args.output)
        return
    cases = review_cases(maps, endpoints)
    if args.mode == "plate":
        make_plate(cases, args.dataset, args.output)
    else:
        with args.output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, ["case", "id", "label", *endpoints], lineterminator="\n")
            writer.writeheader()
            for letter, row in cases.items():
                writer.writerow({"case": letter, **row})


if __name__ == "__main__":
    main()
