"""Render the released checkpoint-250 validation showcase on CPU, without inference."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tarfile
import zipfile

from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version

from wafervlm.core import (
    LABELS, PROMPT, checked_image, metrics, parse_label, read_jsonl,
    sha256_file, write_json,
)

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = 42682266526
DATASET_SHA256 = "c588b3ef59d9dba57cfc745ec20db903e6ff2dcc75504112348f8b9f1ad3ab68"
MANIFEST_SHA256 = "3243e450163721eecac5ce5f4667f9a93599674337fed46e5e40ddeb940ab54f"
RELEASE = "https://github.com/jasonjiang8866/waferVLM/releases/tag/continuation-300-v1"

# Editorial examples chosen for visible morphology, independently of correctness.
# The full matrix uses ALL validation records; its thumbnails use lowest sample ID.
ATLAS_IDS = {
    "center": "4c0cb116592b2babf3807229",
    "donut": "32f11e5f884139a8eb1a749a",
    "edge-loc": "5e90d1e702ea245b652ba450",
    "edge-ring": "0ad3c35371e7fe06efb357a2",
    "loc": "dd855c3b619057c30af18ade",
    "near-full": "8e56c17bd8a8ad08868d3973",
    "random": "8507bcec612300fe21c3be98",
    "scratch": "5f0dc094af5b30aae2baddf1",
}
HERO_IDS = ("5f0dc094af5b30aae2baddf1", "27eadb2b864e3ee3edfdda41")
DESCRIPTIONS = {
    "center": "Failures clustered near the center",
    "donut": "A ring of failures inside the wafer",
    "edge-loc": "A local cluster along the edge",
    "edge-ring": "Failures around the outer rim",
    "loc": "A localized patch of failures",
    "near-full": "Almost the whole wafer fails",
    "random": "Failures scattered across the wafer",
    "scratch": "A line or curved trace of failures",
}

BG, INK, MUTED = "#f3f6fa", "#152b42", "#53667b"
LINE, TEAL, RUST = "#d7e0e9", "#087d78", "#ad4a17"


def load_release(path):
    """Read only named evaluation files from the hash-pinned release; never extract."""
    evidence = json.loads((ROOT / "docs/evidence/continuation-42682266526.json").read_text())
    expected = evidence["assets"]["wafervlm-run-42682266526.zip"]["sha256"]
    if sha256_file(path) != expected:
        raise ValueError("Release ZIP checksum mismatch")
    prefix = "results/extension/validation-250/"
    wanted = {prefix + n for n in ("run.json", "metrics.json", "predictions.jsonl", "checksums.json")}
    files = {}
    with zipfile.ZipFile(path) as z, z.open("experiment.tar.gz") as member:
        with tarfile.open(fileobj=member, mode="r|gz") as archive:
            for item in archive:
                if item.name in wanted:
                    if not item.isfile() or item.size > 2_000_000 or item.name in files:
                        raise ValueError("Unexpected evaluation archive member")
                    files[item.name] = archive.extractfile(item).read()
    if set(files) != wanted:
        raise ValueError("Incomplete evaluation evidence")
    files = {name.removeprefix(prefix): data for name, data in files.items()}
    for name, digest in json.loads(files["checksums.json"]).items():
        if name not in files or hashlib.sha256(files[name]).hexdigest() != digest:
            raise ValueError("Evaluation evidence checksum mismatch")
    return files, evidence


def validate_predictions(rows, manifest, dataset=None):
    """Reject silent omissions, duplicates, label drift, or wrong image joins."""
    lookup = {r["id"]: r for r in manifest}
    if len(lookup) != len(manifest):
        raise ValueError("Duplicate manifest ID")
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids) or set(ids) != set(lookup):
        raise ValueError("Predictions must cover the complete validation manifest once")
    for row in rows:
        source = lookup[row["id"]]
        if source["split"] != "validation":
            raise ValueError("Showcase requires validation records")
        if any(row[k] != source[k] for k in ("label", "image_sha256")):
            raise ValueError("Prediction/image identity mismatch")
        if parse_label(row["response"], row["truncated"]) != row["prediction"]:
            raise ValueError("Raw response disagrees with parsed prediction")
        if dataset is not None:
            checked_image(dataset, source)
    return lookup


def build_cells(rows):
    """Return rows=predicted, columns=true; percentages are per true column."""
    if any(r["prediction"] not in LABELS or r["label"] not in LABELS for r in rows):
        raise ValueError("Eight-label matrix cannot silently omit invalid outputs")
    totals = Counter(r["label"] for r in rows)
    cells = []
    for predicted in LABELS:
        for true in LABELS:
            samples = sorted(
                (r["id"] for r in rows if r["label"] == true and r["prediction"] == predicted)
            )
            cells.append(dict(
                true_label=true, predicted_label=predicted, count=len(samples),
                true_class_total=totals[true],
                fraction_of_true_class=len(samples) / totals[true] if totals[true] else 0,
                sample_ids=samples, thumbnail_id=samples[0] if samples else None,
            ))
    return cells


class Canvas:
    def __init__(self, width, height, fonts):
        self.image = Image.new("RGB", (width, height), BG)
        self.draw = ImageDraw.Draw(self.image)
        self.fonts = fonts

    def text(self, xy, text, size=24, color=INK, bold=False, anchor=None):
        font = ImageFont.truetype(str(self.fonts[bool(bold)]), size)
        self.draw.text(xy, text, font=font, fill=color, anchor=anchor)

    def box(self, xy, fill="white", outline=LINE, radius=18, width=1):
        self.draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)

    def wafer(self, path, xy, size):
        with Image.open(path) as image:
            self.image.paste(image.convert("RGB").resize((size, size), Image.Resampling.NEAREST), xy)

    def legend(self, x, y, size=19):
        for label, color, advance in (
            ("Defective die", "#ffdc00", 205),
            ("Good die", "#005ab4", 165),
            ("Outside wafer", "#000000", 220),
        ):
            self.draw.rectangle((x, y+5, x+17, y+22), fill=color)
            self.text((x+27, y), label, size, MUTED)
            x += advance

    def save(self, output, name):
        self.image.save(output / name, optimize=True)


def draw_hero(output, images, lookup, fonts):
    c = Canvas(1600, 840, fonts)
    c.text((48, 30), "waferVLM  /  CHECKPOINT 250  /  VALIDATION", 19, TEAL, True)
    c.text((48, 74), "Can a small VLM read a wafer failure pattern?", 42, bold=True)
    c.text((48, 135), "Two real inputs. The same question. One correct answer and one mistake.", 25, MUTED)
    c.text((48, 194), 'Prompt excerpt: "Classify the defect pattern in this wafer map."', 23)
    c.text((48, 231), "The full prompt defines the colors and asks for exactly one of eight labels.", 21, MUTED)
    for i, identity in enumerate(HERO_IDS):
        row = lookup[identity]
        good = row["prediction"] == row["label"]
        color = TEAL if good else RUST
        x, y = 48+i*776, 292
        c.box((x, y, x+728, y+402))
        c.text((x+24, y+21), "CORRECT" if good else "MISCLASSIFIED", 19, color, True)
        c.wafer(images[identity], (x+24, y+70), 256)
        c.text((x+310, y+77), "ACTUAL MODEL OUTPUT", 17, MUTED, True)
        c.text((x+310, y+113), row["response"], 43, color, True)
        c.text((x+310, y+190), "DATASET LABEL", 17, MUTED, True)
        c.text((x+310, y+225), row["label"], 32, bold=True)
        c.text((x+24, y+352), "Sample " + identity, 17, MUTED)
    c.legend(48, 724)
    c.text((48, 779), "Recorded predictions, not a live demo. Two selected examples do not measure overall accuracy.", 20, MUTED)
    c.save(output, "inference.png")


def draw_atlas(output, images, fonts):
    c = Canvas(1600, 1150, fonts)
    c.text((48, 30), "READING A WAFER MAP", 19, TEAL, True)
    c.text((48, 72), "Eight labels, eight spatial patterns", 42, bold=True)
    c.text((48, 132), "Each colored cell is a die location. Yellow shows a failed test; blue shows a passed test.", 23, MUTED)
    c.legend(48, 178)
    for i, label in enumerate(LABELS):
        x, y = 48+(i%4)*382, 238+(i//4)*414
        c.box((x, y, x+358, y+390))
        c.text((x+20, y+16), label, 29, bold=True)
        c.wafer(images[ATLAS_IDS[label]], (x+67, y+65), 224)
        # Fixed two-line descriptions avoid unreadable tiny type in README previews.
        words = DESCRIPTIONS[label].split()
        lines, line = [], ""
        for word in words:
            trial = f"{line} {word}".strip()
            font = ImageFont.truetype(str(fonts[False]), 20)
            if c.draw.textlength(trial, font=font) > 318:
                lines.append(line)
                line = word
            else:
                line = trial
        lines.append(line)
        for n, line in enumerate(lines):
            c.text((x+20, y+307+n*26), line, 20, MUTED)
    c.text((48, 1081), "Labels shown are dataset annotations. Examples were chosen for readability, not prediction success.", 20, MUTED)
    c.text((48, 1114), "Patterns describe appearance; they do not establish a manufacturing root cause.", 20, MUTED)
    c.save(output, "label-atlas.png")


def draw_matrix(output, images, cells, score, fonts):
    c = Canvas(2120, 2240, fonts)
    c.text((58, 32), "CHECKPOINT 250  /  ALL 160 VALIDATION MAPS", 22, TEAL, True)
    c.text((58, 82), "See what the model gets right - and what it confuses", 43, bold=True)
    c.text((58, 151), f"{round(score['accuracy']*160)}/160 correct   |   Accuracy {score['accuracy']:.3%}   |   Macro-F1 {score['macro_f1']:.6f}   |   0 invalid outputs", 27)
    c.text((58, 203), "One real thumbnail per non-empty cell. Counts include every sample; percentages are within each true class.", 24, MUTED)
    left, top, width, height = 235, 365, 226, 215
    c.text((left+4*width, 269), "TRUE LABEL  /  20 samples per column", 27, bold=True, anchor="mt")
    c.text((24, 311), "PREDICTED", 21, bold=True)
    c.text((24, 338), "LABEL", 21, bold=True)
    by_key = {(x["predicted_label"], x["true_label"]): x for x in cells}
    for i, label in enumerate(LABELS):
        c.text((left+i*width+width/2, 322), label, 22, bold=True, anchor="mt")
        c.text((left-24, top+i*height+height/2), label, 22, bold=True, anchor="rm")
    for row, predicted in enumerate(LABELS):
        for col, true in enumerate(LABELS):
            cell = by_key[predicted, true]
            x, y = left+col*width, top+row*height
            correct = predicted == true
            color = TEAL if correct else RUST
            fill = "#e5f4ef" if correct else ("#fff3e9" if cell["count"] else "#edf1f5")
            c.box((x+4, y+4, x+width-4, y+height-4), fill=fill, radius=8)
            if not cell["count"]:
                c.text((x+width/2, y+height/2), "0", 29, "#7b8896", anchor="mm")
                continue
            label = f"{cell['count']}/20  ({cell['fraction_of_true_class']:.0%})"
            c.text((x+width/2, y+11), label, 23, color, True, "mt")
            c.wafer(images[cell["thumbnail_id"]], (x+49, y+45), 128)
            c.text((x+width/2, y+184), cell["thumbnail_id"][:12], 14, MUTED, anchor="mt")
    c.text((58, 2108), "Teal diagonal: correct label. Orange cell: misclassification. Gray zero: no samples in this cell.", 23, MUTED)
    c.text((58, 2149), "Thumbnail rule: lowest sample ID in each cell. A thumbnail illustrates that cell; it is not an average wafer.", 22, MUTED)
    c.text((58, 2190), "Validation was used to select this checkpoint. This is not a test score or a final-holdout evaluation.", 22, MUTED)
    c.save(output, "image-confusion-matrix.png")


def write_report(output, rows, cells):
    lookup = {r["id"]: r for r in rows}
    lines = [
        "# Checkpoint-250 visual showcase", "",
        "Real WM-811K inputs and **recorded validation predictions**, from "
        f"[run {RUN_ID}](https://github.com/jasonjiang8866/waferVLM/actions/runs/{RUN_ID}). "
        f"The model and original evidence are in the [release]({RELEASE}).", "",
        "This showcase reuses all 160 predictions: **125 correct, 35 incorrect, 0 invalid**. "
        "Accuracy is **78.125%** and macro-F1 is **0.773992**. Validation selected this "
        "checkpoint; these are not test or final-holdout results. No new inference ran.", "",
        "## From a wafer map to a label", "",
        "A wafer contains many dies (individual chips). A wafer map records their test "
        "results at each location. Yellow cells are defective dies, blue cells are good "
        "dies, and black is outside the wafer. The model identifies the spatial pattern "
        "of failures. It does not infer a process root cause.", "",
        "![Two actual validation inputs: one scratch correctly predicted as scratch, "
        "and another scratch misclassified as loc.](inference.png)", "",
        "The image quotes the first sentence of the actual prompt. The full prompt is:", "",
        "> " + PROMPT, "",
        "Greedy decoding, thinking disabled, at most 16 new tokens. "
        "The response fields shown are the original recorded model outputs.", "",
        "## Eight-label guide", "",
        "![Eight real annotated wafer maps: center, donut, edge-loc, edge-ring, loc, "
        "near-full, random, and scratch.](label-atlas.png)", "",
        "These eight examples were selected editorially for visible morphology, "
        "not by model confidence or prediction success. Real annotations can be visually "
        "ambiguous; they are retained unchanged. For transparency, the selected `loc` "
        "example is predicted `donut`, not `loc`. Normal (`none`) and unknown defects "
        "are outside the current eight-class experiment.", "",
        "| Dataset label | Appearance | Selected sample ID | Checkpoint-250 prediction |",
        "|---|---|---|---|",
    ]
    for label, identity in ATLAS_IDS.items():
        lines.append(f"| `{label}` | {DESCRIPTIONS[label]} | `{identity}` | `{lookup[identity]['prediction']}` |")
    lines += [
        "", "## Image confusion matrix", "",
        "**X / columns = true labels; Y / rows = predicted labels.** This orientation "
        "is intentional. Every true-label column contains 20 samples, so its percentages "
        "sum to 100%. The diagonal counts are per-class correct predictions (recall "
        "times 20), not precision.", "",
        "[![Image confusion matrix: all 160 validation predictions, "
        "with a real thumbnail in each non-empty cell.](image-confusion-matrix.png)]"
        "(image-confusion-matrix.png)", "",
        "Open the image at full size to inspect the thumbnails. Each non-empty cell "
        "shows the lowest sample ID in that cell; all other sample IDs are listed in "
        "[evidence.json](evidence.json). Zero-count cells contain no invented example. "
        "Original image colors and aspect ratios are preserved with nearest-neighbor "
        "scaling. Headers, captions and borders were added only after inference.", "",
        "### Accessible count table", "",
        "| Predicted \\ True | " + " | ".join(LABELS) + " |",
        "|---|" + "---:|"*len(LABELS),
    ]
    for predicted in LABELS:
        lines.append("| " + predicted + " | " + " | ".join(str(x["count"]) for x in cells if x["predicted_label"] == predicted) + " |")
    lines += [
        "", "## Provenance and reproduction", "",
        "The public source is the [qingyi WM-811K mirror](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map). "
        "The pinned dataset recipe, source hashes and data terms are documented in "
        "[the data audit](../../data-audit.md). These visual excerpts do not relicense "
        "the upstream dataset under the code's MIT license.", "",
        "The CPU-only builder verifies the original release ZIP, evaluation checksums, "
        "full validation coverage, raw-output parsing, every image hash, and recomputed "
        "metrics before drawing. [predictions.jsonl](predictions.jsonl) is an unchanged "
        "copy of the released prediction file. [evidence.json](evidence.json) records "
        "source identities, sample selections, matrix orientation and output hashes.", "",
        "From the repository root, with the release ZIP downloaded and a new output directory:", "",
        "```bash",
        "uv sync --frozen",
        "# The source recipe verifies the pinned public archive before loading its pickle.",
        "uv run --frozen python scripts/prepare_public_data.py \\",
        "  --trust-pickle --output data/wm811k-lot17",
        "# Requires DejaVuSans.ttf and DejaVuSans-Bold.ttf (fonts-dejavu-core on Debian/Ubuntu).",
        "uv run --frozen python scripts/build_showcase.py \\",
        "  --dataset data/wm811k-lot17 \\",
        "  --release-zip /path/to/wafervlm-run-42682266526.zip \\",
        "  --output outputs/checkpoint-250-showcase",
        "```", "",
        "Use `--font-dir` if DejaVu fonts are installed elsewhere. The builder does not "
        "load model weights, train, call RunPod, or evaluate test/final-holdout data. "
        "It deliberately accepts only this pinned release; a new checkpoint requires "
        "an explicit evidence and example-selection update.", "",
    ]
    (output / "README.md").write_text("\n".join(lines))


def build(dataset, release_zip, output, font_dir):
    dataset, output = Path(dataset), Path(output)
    fonts = {False: Path(font_dir)/"DejaVuSans.ttf", True: Path(font_dir)/"DejaVuSans-Bold.ttf"}
    if not all(p.is_file() for p in fonts.values()):
        raise ValueError("DejaVu Sans fonts missing; set --font-dir")
    files, provenance = load_release(release_zip)
    run = json.loads(files["run.json"])
    if (run["config"]["split"] != "validation" or run["prompt"] != PROMPT
            or run["labels"] != list(LABELS) or run["dataset_sha256"] != DATASET_SHA256):
        raise ValueError("Evaluation protocol mismatch")
    if sha256_file(dataset/"dataset.json") != DATASET_SHA256:
        raise ValueError("Dataset identity mismatch")
    if sha256_file(dataset/"validation.jsonl") != MANIFEST_SHA256:
        raise ValueError("Validation manifest identity mismatch")
    rows = [json.loads(line) for line in files["predictions.jsonl"].splitlines() if line.strip()]
    manifest = read_jsonl(dataset/"validation.jsonl")
    records = validate_predictions(rows, manifest, dataset)
    score = metrics(rows)
    saved = json.loads(files["metrics.json"])
    if len(rows) != 160 or any(score[k] != saved[k] for k in score):
        raise ValueError("Recomputed metrics differ from release")
    if Counter(r["label"] for r in rows) != Counter({label: 20 for label in LABELS}):
        raise ValueError("Expected 20 validation samples per class")
    lookup = {r["id"]: r for r in rows}
    for label, identity in ATLAS_IDS.items():
        if lookup[identity]["label"] != label:
            raise ValueError("Atlas example has the wrong dataset label")
    if [(lookup[i]["label"], lookup[i]["prediction"]) for i in HERO_IDS] != [("scratch", "scratch"), ("scratch", "loc")]:
        raise ValueError("Hero examples do not match the stated outcomes")
    cells = build_cells(rows)
    images = {identity: checked_image(dataset, row) for identity, row in records.items()}
    output.mkdir(parents=True, exist_ok=False)
    draw_hero(output, images, lookup, fonts)
    draw_atlas(output, images, fonts)
    draw_matrix(output, images, cells, score, fonts)
    write_report(output, rows, cells)
    (output/"predictions.jsonl").write_bytes(files["predictions.jsonl"])
    write_json(output/"evidence.json", dict(
        schema_version=1, checkpoint=250, split="validation", source_run_id=RUN_ID,
        source_release=RELEASE, source_sha=run["source_sha"], image_digest=run["image_digest"],
        model_id=run["model_id"], model_revision=run["model_revision"],
        release_zip_sha256=provenance["zip_sha256"], dataset_sha256=DATASET_SHA256,
        validation_manifest_sha256=MANIFEST_SHA256,
        prediction_sha256=hashlib.sha256(files["predictions.jsonl"]).hexdigest(),
        inference_executed=False, test_evaluated=False, final_holdout_evaluated=False,
        metrics=score, matrix_axes=dict(x="true_label", y="predicted_label"),
        percentage_denominator="all validation samples with this true label (column)",
        matrix_thumbnail_rule="lowest lexicographic sample ID in each non-empty cell",
        atlas_selection="editorial morphology examples, not filtered by correctness",
        atlas_ids=ATLAS_IDS, hero_ids=HERO_IDS, cells=cells,
        displayed_samples=[{**lookup[i], "source_row": records[i]["source_row"]}
                           for i in sorted(set(ATLAS_IDS.values()) | set(HERO_IDS)
                                           | {c["thumbnail_id"] for c in cells if c["thumbnail_id"]})],
        renderer=dict(pillow=pillow_version, fonts={p.name: sha256_file(p) for p in fonts.values()},
                      interpolation="nearest", input_palette="0:black, 1:blue, 2:yellow"),
        outputs={p.name: sha256_file(p) for p in sorted(output.iterdir()) if p.is_file()},
    ))
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--release-zip", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--font-dir", default="/usr/share/fonts/truetype/dejavu")
    args = parser.parse_args()
    result = build(args.dataset, args.release_zip, args.output, args.font_dir)
    print(json.dumps({k: result[k] for k in ("n", "accuracy", "macro_f1", "invalid_rate")}, indent=2))
