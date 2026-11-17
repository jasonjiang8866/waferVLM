"""Offline checks for real collection identity and consequential ingress gates."""

import importlib.util
import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("collector", ROOT / "scripts/collect_wafer_data.py")
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.payload = self.root / "data.zip"
        with zipfile.ZipFile(self.payload, "w") as z:
            z.writestr("images/a.txt", b"known public fixture")
        self.file = self.file_spec(self.payload)
        self.dataset = {"id": "fixture", "version": "v1", "source": "https://zenodo.org/records/1",
                        "license": "CC-BY-4.0", "license_verified": True, "files": [self.file]}

    def file_spec(self, path):
        return {"name": path.name, "url": "https://zenodo.org/api/records/1/files/data.zip/content",
                "bytes": path.stat().st_size, "md5": collector.digest(path, "md5"),
                "sha256": collector.digest(path), "expanded_bytes_limit": 1024}

    def catalog(self, dataset):
        p = self.root / "catalog.json"
        p.write_text(json.dumps({"datasets": [dataset]}))
        return patch.object(collector, "CATALOG", p)

    def staged(self):
        cache = self.root / "cache"
        p = cache / "fixture/v1/data.zip"
        p.parent.mkdir(parents=True)
        p.write_bytes(self.payload.read_bytes())
        mount = self.root / "mount"
        mount.mkdir()
        return cache, mount

    def test_real_catalog_is_bounded_and_matches_primary_metadata(self):
        data = json.loads(collector.CATALOG.read_text())
        selected, size = collector.plan([d["id"] for d in data["datasets"]])
        self.assertLess(size["peak_bytes_ceiling"], collector.LIMIT)
        for d in selected:
            record = json.loads((ROOT / "data_sources/metadata" / (d["version"] + ".json")).read_text())
            self.assertEqual(record["metadata"]["license"]["id"], "cc-by-4.0")
            files = {f["key"]: f for f in record["files"]}
            for f in d["files"]:
                self.assertEqual(f["bytes"], files[f["name"]]["size"])
                self.assertEqual("md5:" + f["md5"], files[f["name"]]["checksum"])

    def test_real_samples_keep_source_hashes_and_git_budget(self):
        root = ROOT / "data_sources/samples/carinthia-s"
        m = json.loads((root / "manifest.json").read_text())
        self.assertEqual(len(m["samples"]), 12)
        self.assertEqual(sum(m["class_counts"].values()), 4591)
        for s in m["samples"]:
            for kind in ("image", "mask"):
                p = root / s[kind]["path"]
                self.assertEqual(p.stat().st_size, s[kind]["bytes"])
                self.assertEqual(collector.digest(p), s[kind]["sha256"])
        all_samples = ROOT / "data_sources/samples"
        self.assertLess(sum(p.stat().st_size for p in all_samples.rglob("*") if p.is_file()), 20_000_000)
        pl = all_samples / "silicon-wafer-pl"
        item = json.loads((pl / "manifest.json").read_text())
        self.assertEqual(collector.digest(pl / item["source_filename"]), item["sha256"])
        self.assertEqual((pl / item["source_filename"]).stat().st_size, item["bytes"])

    def test_over_10gb_is_blocked_before_network(self):
        self.dataset["files"][0]["bytes"] = collector.LIMIT
        with self.catalog(self.dataset), self.assertRaisesRegex(ValueError, "10 GB"):
            collector.plan(["fixture"])

    def test_exact_boundary_and_unknown_sizes(self):
        self.dataset["files"][0]["bytes"] = collector.LIMIT - 1024
        with self.catalog(self.dataset):
            self.assertEqual(collector.plan(["fixture"])[1]["peak_bytes_ceiling"], collector.LIMIT)
        self.dataset["files"][0]["bytes"] = None
        with self.catalog(self.dataset), self.assertRaisesRegex(ValueError, "size"):
            collector.plan(["fixture"])

    def test_unreviewed_license_duplicate_ids_and_unsafe_names(self):
        with self.catalog(self.dataset), self.assertRaises(ValueError):
            collector.plan(["fixture", "fixture"])
        self.dataset["license_verified"] = False
        with self.catalog(self.dataset), self.assertRaisesRegex(ValueError, "license"):
            collector.plan(["fixture"])
        with self.assertRaises(ValueError):
            collector.token("../outside")

    def test_corruption_and_sha_pin_rejected(self):
        self.file["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA256"):
            collector.verify_file(self.payload, self.file)
        self.payload.write_bytes(b"broken")
        with self.assertRaisesRegex(ValueError, "size"):
            collector.verify_file(self.payload, self.file)

    def test_zip_traversal_symlink_and_expansion_rejected(self):
        for name in ("../escape", "/absolute", "C:/windows", "back\\slash"):
            with zipfile.ZipFile(self.payload, "w") as z:
                z.writestr(name, "x")
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                collector.verify_file(self.payload, self.file_spec(self.payload))
        with zipfile.ZipFile(self.payload, "w") as z:
            info = zipfile.ZipInfo("link")
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            z.writestr(info, "/etc/passwd")
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            collector.verify_file(self.payload, self.file_spec(self.payload))
        with zipfile.ZipFile(self.payload, "w") as z:
            z.writestr("huge", b"x" * 2048)
        with self.assertRaisesRegex(ValueError, "expansion"):
            collector.verify_file(self.payload, self.file_spec(self.payload))

    def test_stream_overrun_does_not_publish(self):
        class Response(io.BytesIO):
            headers = {}
        path = self.root / "download/data.zip"
        with patch.object(collector.urllib.request, "urlopen", return_value=Response(b"x" * (self.file["bytes"] + 1))):
            with self.assertRaisesRegex(ValueError, "bound"):
                collector.fetch_file(self.file, path)
        self.assertFalse(path.exists())
        self.assertFalse(path.with_name("data.zip.part").exists())

    def test_plain_directory_is_not_a_volume(self):
        cache, mount = self.staged()
        with self.assertRaisesRegex(ValueError, "mount point"):
            collector.ingest([self.dataset], cache, mount, "fixture-volume")
        self.assertFalse((mount / "waferVLM-datasets").exists())

    def test_mounted_import_readback_and_corruption(self):
        cache, mount = self.staged()
        with patch.object(collector.os.path, "ismount", return_value=True):
            receipt = collector.ingest([self.dataset], cache, mount, "fixture-volume")
            self.assertEqual(receipt[0]["status"], "mounted_readback_verified")
            self.assertEqual(collector.verify_volume([self.dataset], mount, "fixture-volume")["status"], "readback_verified")
            with self.assertRaisesRegex(ValueError, "exists"):
                collector.ingest([self.dataset], cache, mount, "fixture-volume")
            with self.assertRaisesRegex(ValueError, "identity"):
                collector.verify_volume([self.dataset], mount, "wrong-volume")
            (mount / "waferVLM-datasets/fixture/v1/data.zip").write_bytes(b"broken")
            with self.assertRaises(ValueError):
                collector.verify_volume([self.dataset], mount, "fixture-volume")

    def test_partial_copy_does_not_create_receipt(self):
        cache, mount = self.staged()
        with patch.object(collector.os.path, "ismount", return_value=True), patch.object(collector, "copy_verified", side_effect=OSError("connection lost")):
            with self.assertRaises(OSError):
                collector.ingest([self.dataset], cache, mount, "fixture-volume")
        self.assertFalse((mount / "waferVLM-datasets/fixture/v1/receipt.json").exists())

    def test_cumulative_gate_counts_incomplete_bytes(self):
        cache, mount = self.staged()
        with patch.object(collector.os.path, "ismount", return_value=True), patch.object(collector, "inventory_bytes", return_value=collector.LIMIT):
            with self.assertRaisesRegex(ValueError, "Cumulative"):
                collector.ingest([self.dataset], cache, mount, "fixture-volume")
        self.assertFalse((mount / "waferVLM-datasets").exists())

    def test_catalog_growth_does_not_invalidate_existing_receipt(self):
        cache, mount = self.staged()
        with self.catalog(self.dataset), patch.object(collector.os.path, "ismount", return_value=True):
            collector.ingest([self.dataset], cache, mount, "fixture-volume")
            collector.CATALOG.write_text('{"datasets": [], "unrelated_metadata": "new"}')
            self.assertEqual(collector.verify_volume([self.dataset], mount, "fixture-volume")["status"], "readback_verified")

    def test_existing_later_dataset_blocks_whole_batch_before_copy(self):
        cache, mount = self.staged()
        other = dict(self.dataset, id="other")
        p = cache / "other/v1/data.zip"
        p.parent.mkdir(parents=True)
        p.write_bytes(self.payload.read_bytes())
        (mount / "waferVLM-datasets/other/v1").mkdir(parents=True)
        with patch.object(collector.os.path, "ismount", return_value=True):
            with self.assertRaisesRegex(ValueError, "exists"):
                collector.ingest([self.dataset, other], cache, mount, "fixture-volume")
        self.assertFalse((mount / "waferVLM-datasets/fixture").exists())


if __name__ == "__main__":
    unittest.main()
