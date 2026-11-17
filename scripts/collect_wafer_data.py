"""Collect licensed public data; copy verified archives to an already mounted volume.

No credentials, cloud API calls, Pod allocation, archive extraction or training.
Global Volumes are object backed: write immutable files and a receipt LAST;
never rely on rename, chmod, flock, or a mutable shared index on the mount.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data_sources/catalog.json"
LIMIT = 10_000_000_000  # decimal GB; user approval required above this
CHUNK = 1024 * 1024


def token(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("Unsafe identifier or filename")
    return value


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_digest(dataset):
    # Future additions to the catalog must not invalidate an existing version.
    identity = {k: dataset[k] for k in ("id", "version", "source", "license", "files")}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def plan(ids):
    catalog = json.loads(CATALOG.read_text())
    lookup = {d["id"]: d for d in catalog["datasets"]}
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Supply a nonempty, unique dataset list")
    selected = [lookup[i] for i in ids]
    for d in selected:
        token(d["id"])
        token(d["version"])
        if d.get("license_verified") is not True or d.get("license") != "CC-BY-4.0":
            raise ValueError("Dataset license has not passed collection review")
        if not d["files"] or len({f["name"] for f in d["files"]}) != len(d["files"]):
            raise ValueError("Empty or duplicate file list")
        for f in d["files"]:
            token(f["name"])
            for field in ("bytes", "expanded_bytes_limit"):
                if type(f[field]) is not int or f[field] <= 0:
                    raise ValueError("Unknown or invalid size: review before downloading")
            if not f["url"].startswith("https://zenodo.org/api/records/"):
                raise ValueError("Unreviewed download origin")
            if not re.fullmatch(r"[0-9a-f]{32}", f["md5"]):
                raise ValueError("Missing publisher checksum")
            if f.get("sha256") and not re.fullmatch(r"[0-9a-f]{64}", f["sha256"]):
                raise ValueError("Invalid SHA256")
    download = sum(f["bytes"] for d in selected for f in d["files"])
    expanded = sum(f["expanded_bytes_limit"] for d in selected for f in d["files"])
    # Count archive AND unpacked copies, even though the volume stores archives only.
    if download + expanded > LIMIT:
        raise ValueError("Above 10 GB: explicit user approval and a reviewed larger plan required")
    return selected, {"download_bytes": download, "expanded_bytes_ceiling": expanded,
                      "peak_bytes_ceiling": download + expanded, "limit_bytes": LIMIT}


def verify_file(path, spec):
    path = Path(path)
    if path.is_symlink() or path.stat().st_size != spec["bytes"]:
        raise ValueError("File size/type mismatch: " + spec["name"])
    if digest(path, "md5") != spec["md5"]:
        raise ValueError("Publisher checksum mismatch: " + spec["name"])
    sha = digest(path)
    if spec.get("sha256") and sha != spec["sha256"]:
        raise ValueError("Pinned SHA256 mismatch: " + spec["name"])
    expanded = spec["bytes"]
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            expanded = 0
            names = set()
            for info in archive.infolist():
                name = PurePosixPath(info.filename)
                if (name.is_absolute() or ".." in name.parts or "\\" in info.filename
                        or ":" in info.filename or info.filename in names
                        or stat.S_ISLNK(info.external_attr >> 16) or info.flag_bits & 1):
                    raise ValueError("Unsafe or duplicate ZIP member")
                names.add(info.filename)
                expanded += info.file_size
                if expanded > spec["expanded_bytes_limit"]:
                    raise ValueError("ZIP expansion exceeds reviewed bound")
            if archive.testzip() is not None:
                raise ValueError("ZIP CRC verification failed")
    return {"name": spec["name"], "bytes": spec["bytes"], "sha256": sha,
            "publisher_md5": spec["md5"], "expanded_bytes": expanded}


def fetch_file(spec, path):
    path = Path(path)
    if path.exists():
        return verify_file(path, spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    # A failed download is never renamed into place. Retries start from zero.
    try:
        start, count = time.monotonic(), 0
        with urllib.request.urlopen(spec["url"], timeout=60) as response, part.open("wb") as out:
            if response.headers.get("Content-Length") not in (None, str(spec["bytes"])):
                raise ValueError("Upstream size changed")
            while chunk := response.read(CHUNK):
                count += len(chunk)
                if count > spec["bytes"] or time.monotonic() - start > 600:
                    raise ValueError("Download exceeds byte/time bound")
                out.write(chunk)
        # Verify before local rename; suffix must remain .zip for archive checks.
        if part.stat().st_size != spec["bytes"] or digest(part, "md5") != spec["md5"]:
            raise ValueError("Incomplete download or publisher checksum mismatch")
        if spec.get("sha256") and digest(part) != spec["sha256"]:
            raise ValueError("Pinned SHA256 mismatch")
        part.rename(path)  # local staging disk only
        return verify_file(path, spec)
    finally:
        part.unlink(missing_ok=True)


def acquire(selected, cache):
    result = []
    for d in selected:
        files = [fetch_file(f, cache / d["id"] / d["version"] / f["name"]) for f in d["files"]]
        result.append({"id": d["id"], "version": d["version"], "source": d["source"],
                       "license": d["license"], "files": files})
    return result


def copy_verified(source, target, spec):
    if target.exists():
        if digest(target) != digest(source):
            raise ValueError("Immutable destination already exists with different bytes")
    else:
        # Do not use copy2: Global Volumes do not implement permission bits.
        with source.open("rb") as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst, CHUNK)
    return verify_file(target, spec)


def inventory_bytes(root):
    """Count even incomplete uploads. A single writer is required operationally."""
    total = 0
    for p in root.rglob("*"):
        if p.is_symlink():
            raise ValueError("Symlinks are not supported in the collection")
        if p.is_file():
            total += p.stat().st_size
    return total


def ingest(selected, cache, mount, volume_id):
    token(volume_id)
    if mount.is_symlink() or not os.path.ismount(mount):
        raise ValueError("Destination is not a mount point; refusing a fake local import")
    mount = mount.resolve()
    if cache.resolve().is_relative_to(mount):
        raise ValueError("Stage downloads on local disk, outside the Global Volume")
    # Verify all local bytes before the first remote write.
    for d in selected:
        for f in d["files"]:
            verify_file(cache / d["id"] / d["version"] / f["name"], f)
    root = mount / "waferVLM-datasets"
    if root.is_symlink():
        raise ValueError("Collection root must not be a symlink")
    existing = inventory_bytes(root)
    # Conservative cumulative gate (existing bytes + full planned peak + receipts).
    planned = sum(f["bytes"] + f["expanded_bytes_limit"] for d in selected for f in d["files"])
    if existing + planned + 1_000_000 > LIMIT:
        raise ValueError("Cumulative collection exceeds 10 GB; approval required")
    for d in selected:
        if (root / d["id"] / d["version"]).exists():
            raise ValueError("Version directory exists; verify it or review incomplete import")
    receipts = []
    for d in selected:
        dest = root / d["id"] / d["version"]
        if dest.exists():
            raise ValueError("Version directory exists; verify it or review incomplete import")
        dest.mkdir(parents=True)
        files = [copy_verified(cache / d["id"] / d["version"] / f["name"], dest / f["name"], f)
                 for f in d["files"]]
        receipt = {"schema_version": 1, "status": "mounted_readback_verified",
                   "id": d["id"], "version": d["version"], "source": d["source"],
                   "license": d["license"], "volume_id": volume_id,
                   "mount_path": str(mount), "catalog_sha256": digest(CATALOG),
                   "dataset_sha256": dataset_digest(d),
                   "verified_utc": datetime.now(timezone.utc).isoformat(), "files": files}
        # Consumers require this last-written marker AND rehash all payloads.
        encoded = json.dumps(receipt, indent=2) + "\n"
        marker = dest / "receipt.json"
        marker.write_text(encoded)
        if marker.read_text() != encoded:
            raise ValueError("Receipt readback mismatch")
        receipts.append(receipt)
    return receipts


def verify_volume(selected, mount, volume_id):
    if mount.is_symlink() or not os.path.ismount(mount):
        raise ValueError("Destination is not a mount point")
    receipts = []
    for d in selected:
        dest = mount / "waferVLM-datasets" / d["id"] / d["version"]
        receipt = json.loads((dest / "receipt.json").read_text())
        if (receipt["volume_id"] != volume_id or receipt["id"] != d["id"]
                or receipt["version"] != d["version"]
                or receipt["status"] != "mounted_readback_verified"
                or receipt["dataset_sha256"] != dataset_digest(d)):
            raise ValueError("Receipt identity mismatch")
        files = [verify_file(dest / f["name"], f) for f in d["files"]]
        if files != receipt["files"]:
            raise ValueError("Receipt payload mismatch")
        receipts.append(receipt)
    return {"status": "readback_verified", "receipts": receipts}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["plan", "fetch", "ingest", "verify-volume"])
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--cache", type=Path, default=ROOT / "data/collection")
    p.add_argument("--mount", type=Path)
    p.add_argument("--volume-id")
    p.add_argument("--receipt", type=Path, help="Local result JSON, outside the volume")
    args = p.parse_args()
    try:
        selected, sizes = plan(args.datasets)
        result = {"status": "planned", "datasets": args.datasets, **sizes}
        if args.command == "fetch":
            result = {"status": "download_verified", "datasets": acquire(selected, args.cache), **sizes}
        elif args.command in ("ingest", "verify-volume"):
            if not args.mount or not args.volume_id:
                raise ValueError("--mount and --volume-id are required; verify these against the Pod API")
            if args.receipt and args.receipt.resolve().is_relative_to(args.mount.resolve()):
                raise ValueError("Output receipt must be outside the Global Volume")
            if args.command == "ingest":
                result = {"status": "mounted_readback_verified", "receipts": ingest(
                    selected, args.cache, args.mount, args.volume_id)}
            else:
                result = verify_volume(selected, args.mount, args.volume_id)
        result["schema_version"] = 1
        result["verified_utc"] = datetime.now(timezone.utc).isoformat()
        result["catalog_sha256"] = digest(CATALOG)
        if args.receipt:
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            args.receipt.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    except (ValueError, KeyError, OSError, zipfile.BadZipFile) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
