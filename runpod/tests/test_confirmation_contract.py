import copy
import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from runpod.confirmation_contract import launch_plan
from runpod.confirmation_verify import unpack


def specification():
    return dict(seed=42, source_sha='a'*40, controller_sha='b'*40,
                image='ghcr.io/jasonjiang8866/wafervlm@sha256:'+'c'*64,
                gpu='NVIDIA GeForce RTX 4090', gpu_hourly_usd=.89,
                ledger=dict(usd=5.903039, seconds=491.911*60, attempts=[
                    dict(run_id=42682266526, marker_artifact_id=13703281873,
                         evidence_sha256='d'*64, cleanup='verified_absent')]))


class ContractTests(unittest.TestCase):
    def test_phase_bound_includes_cleanup_and_disk(self):
        p = launch_plan(specification())
        self.assertEqual(p['lifecycle_seconds'], 195*60)
        self.assertAlmostEqual(p['estimated_additional_usd'], (.89+60*.10/(28*24))*3.25)
        self.assertEqual(p['current_quote'], 'unverified')
        self.assertFalse(p['holdout_access'])

    def test_missing_digest_or_old_image_rejected(self):
        for image in (None, 'ghcr.io/jasonjiang8866/wafervlm:latest',
                      'ghcr.io/jasonjiang8866/wafervlm@sha256:b13a9807ae05698f0813e0d1ab7f25548e782f65729d2b2aec5e66de6de24cd1'):
            s=specification();s['image']=image
            with self.assertRaises(ValueError): launch_plan(s)

    def test_unreconciled_duplicate_or_stale_ledger_rejected(self):
        for mutation in ('cleanup','duplicate','missing','usd','seconds','retry'):
            s=specification();ledger=s['ledger']
            if mutation=='cleanup': ledger['attempts'][0]['cleanup']='unknown'
            if mutation=='duplicate': ledger['attempts']*=2
            if mutation=='missing': ledger['attempts']=[]
            if mutation=='usd': ledger['usd']=5.38
            if mutation=='seconds': ledger['seconds']=457*60
            if mutation=='retry': ledger['attempts'][0]['seed']=42
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):launch_plan(s)

    def test_budget_and_per_gpu_cap_not_just_global_cap(self):
        for field,value in [('gpu_hourly_usd',.90),('gpu_hourly_usd',float('nan')),
                            ('gpu','unknown'),('seed',True)]:
            s=specification();s[field]=value
            with self.assertRaises(ValueError):launch_plan(s)
        for field,value in [('usd',29),('seconds',1000*60)]:
            s=specification();s['ledger'][field]=value
            with self.assertRaises(ValueError):launch_plan(s)

    def test_seed123_waits_for_seed42_reconciliation(self):
        s=specification();s['seed']=123
        with self.assertRaises(ValueError):launch_plan(s)
        s['ledger']['attempts'].append(dict(run_id=99,marker_artifact_id=88,
            evidence_sha256='e'*64,seed=42,accepted=True,cleanup='verified_absent'))
        self.assertEqual(launch_plan(s)['seed'],123)


class ArchivePaths(unittest.TestCase):
    def test_rejects_traversal_links_duplicates_and_noncanonical_paths(self):
        for name,kind in [('results/../escape',tarfile.REGTYPE),
                          ('/results/x',tarfile.REGTYPE),('results//x',tarfile.REGTYPE),
                          ('results/link',tarfile.SYMTYPE),('results/link',tarfile.LNKTYPE),
                          ('results/x',tarfile.FIFOTYPE),('duplicate',tarfile.REGTYPE)]:
            with self.subTest(name=name,kind=kind), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);archive=root/'x.tar.gz'
                with tarfile.open(archive,'w:gz') as tar:
                    for _ in range(2 if name=='duplicate' else 1):
                        m=tarfile.TarInfo('results/x' if name=='duplicate' else name);m.type=kind
                        m.size=1 if kind==tarfile.REGTYPE else 0
                        tar.addfile(m,io.BytesIO(b'x') if m.size else None)
                with self.assertRaises(ValueError):unpack(archive,root/'out')
                self.assertFalse((root/'escape').exists())

    def test_size_and_member_limits(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);archive=root/'x.tar.gz'
            with tarfile.open(archive,'w:gz') as tar:
                m=tarfile.TarInfo('results/x');m.size=2;tar.addfile(m,io.BytesIO(b'xx'))
            for key,value in [('MAX_BYTES',1),('MAX_FILES',0)]:
                with patch('runpod.confirmation_verify.'+key,value), self.assertRaises(ValueError):
                    unpack(archive,root/key)
