"""Pinned public WM-811K mirror acquisition; checksum gates precede pickle loading."""

import shutil
import urllib.request
import zipfile
from pathlib import Path

from .core import sha256_file

SOURCE_URL = "https://www.kaggle.com/api/v1/datasets/download/qingyi/wm811k-wafer-map"
SOURCE_PAGE = "https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map"
ARCHIVE_SHA256 = "0e4bd4316a110a04e475a88071f5e1c7fe4303b12ee53c851d2cae4d995a93e2"
PICKLE_SHA256 = "1d04fccb3dd3176b276878b926b20fead7e077c5751e4d353ea9741a5e7b5c65"
ARCHIVE_BYTES = 156686283
PICKLE_BYTES = 2095505977


def extract_verified(archive, destination):
    """Accept exactly the reviewed archive/member; never extract arbitrary paths."""
    archive, destination = Path(archive), Path(destination)
    if archive.stat().st_size != ARCHIVE_BYTES or sha256_file(archive) != ARCHIVE_SHA256:
        raise ValueError("WM-811K archive checksum/size mismatch")
    if destination.exists():
        if destination.stat().st_size != PICKLE_BYTES or sha256_file(destination) != PICKLE_SHA256:
            raise ValueError("Cached WM-811K pickle checksum/size mismatch")
        return destination
    partial = destination.with_suffix(".partial")
    try:
        with zipfile.ZipFile(archive) as z:
            members = z.infolist()
            if len(members) != 1 or members[0].filename != "LSWMD.pkl" or members[0].file_size != PICKLE_BYTES:
                raise ValueError("Unexpected WM-811K archive members")
            with z.open(members[0]) as src, partial.open("xb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
        if sha256_file(partial) != PICKLE_SHA256:
            raise ValueError("Extracted WM-811K pickle checksum mismatch")
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    return destination


def acquire(cache):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "wm811k.zip"
    if not archive.exists():
        partial = cache / "wm811k.zip.partial"
        try:
            request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "waferVLM/0.1"})
            with urllib.request.urlopen(request, timeout=60) as response, partial.open("xb") as dst:
                total = 0
                while block := response.read(1024 * 1024):
                    total += len(block)
                    if total > ARCHIVE_BYTES:
                        raise ValueError("WM-811K download exceeds pinned size")
                    dst.write(block)
            if total != ARCHIVE_BYTES or sha256_file(partial) != ARCHIVE_SHA256:
                raise ValueError("WM-811K download checksum/size mismatch")
            partial.replace(archive)
        finally:
            partial.unlink(missing_ok=True)
    return extract_verified(archive, cache / "LSWMD.pkl")
