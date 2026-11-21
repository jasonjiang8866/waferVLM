"""Check identity-safe joins on the real published validation and holdout rows."""
import copy
import unittest

from analysis.dinov3_confirmation_report import ROOT, paired, reference_for
from wafervlm.core import read_jsonl


class PairedReportTests(unittest.TestCase):
    def test_reordered_real_validation_retains_predeclared_four_way_counts(self):
        reference=reference_for('validation')
        qwen=read_jsonl(ROOT/'docs/showcase/checkpoint-250/predictions.jsonl')
        for file, expected in [('linear-predictions.jsonl',[112,22,13,13]),
                               ('predictions.jsonl',[117,22,8,13])]:
            dino=read_jsonl(ROOT/'docs/showcase/dinov3-seed17'/file)
            groups=paired(qwen[::-1],dino[73:]+dino[:73],reference)
            self.assertEqual(list(map(len,groups.values())),expected)
            self.assertEqual(sum(map(len,groups.values())),160)

    def test_holdout_join_uses_the_148_map_manifest(self):
        reference=reference_for('holdout')
        qwen=read_jsonl(ROOT/'docs/evidence/qwen-holdout/seed17-step250/predictions.jsonl')
        groups=paired(qwen[::-1],qwen,reference)
        self.assertEqual(sum(map(len,groups.values())),148)
        self.assertEqual(len(groups['both_correct']),sum(r['prediction']==r['label'] for r in qwen))
        self.assertEqual(groups['dino_only_correct'],[])
        self.assertEqual(groups['qwen_only_correct'],[])

    def test_mismatched_image_label_and_duplicate_id_are_rejected(self):
        reference=reference_for('validation')
        qwen=read_jsonl(ROOT/'docs/showcase/checkpoint-250/predictions.jsonl')
        for change in ({'image_sha256':'0'*64},{'label':'incorrect'},{'id':qwen[1]['id']}):
            broken=copy.deepcopy(qwen);broken[0].update(change)
            with self.assertRaises(ValueError):paired(qwen,broken,reference)
        with self.assertRaises(ValueError):paired(qwen,qwen[:-1],reference)


    def test_published_validation_evidence_has_18_verified_processes(self):
        import json
        from runpod.dinov3_study import names, verify_predictions
        from wafervlm.confirmation import verify_tree
        from wafervlm.core import sha256_file
        root=ROOT/'docs/showcase/dinov3-confirmation-validation'
        report=json.loads((root/'analysis.json').read_text())
        frozen=json.loads((ROOT/'docs/evidence/dinov3-six-endpoints.json').read_text())
        expected=json.loads((root/'checksums.json').read_text())
        self.assertEqual(expected,{str(p.relative_to(root)):sha256_file(p) for p in root.rglob('*') if p.is_file() and p.name!='checksums.json'} | {str(p.relative_to(root)):sha256_file(p) for p in (root/'predictions').rglob('checksums.json')})
        self.assertEqual(report['validation_passes'],18)
        self.assertEqual(report['holdout_predictions'],0)
        self.assertEqual(report['endpoints'],frozen['endpoints'])
        pids=set()
        for ep in frozen['endpoints']:
            baseline=None
            for repeat in (1,2,3):
                folder=root/'predictions'/ep['name']/f'repeat{repeat}'
                verify_tree(folder)
                rows=read_jsonl(folder/'predictions.jsonl')
                score=verify_predictions(rows,reference_for('validation'))
                run=json.loads((folder/'run.json').read_text())
                self.assertEqual(run['weight_sha256'],ep['weight_sha256'])
                self.assertNotIn(run['pid'],pids);pids.add(run['pid'])
                if repeat==1:
                    baseline=rows
                    self.assertEqual(score,report['models'][ep['name']])
                else:
                    changed=[r['id'] for r,b in zip(rows,baseline) if r['prediction']!=b['prediction']]
                    self.assertEqual(changed,report['passes'][ep['name']][repeat-1]['label_changes'])
        self.assertEqual(len(pids),18)

    def test_published_holdout_has_six_frozen_models_and_888_paired_rows(self):
        import json
        from runpod.dinov3_study import names, verify_predictions, plan
        from runpod.vlm_holdout import bootstrap
        from wafervlm.confirmation import verify_tree
        from wafervlm.core import sha256_file
        root=ROOT/'docs/showcase/dinov3-confirmation-holdout'
        report=json.loads((root/'analysis.json').read_text())
        frozen=json.loads((ROOT/'docs/evidence/dinov3-six-endpoints.json').read_text())
        self.assertEqual(json.loads((root/'checksums.json').read_text()),
            {str(p.relative_to(root)):sha256_file(p) for p in root.rglob('*') if p.is_file() and p!=root/'checksums.json'})
        self.assertEqual(report['validation_passes'],0)
        self.assertEqual(report['holdout_predictions'],888)
        self.assertEqual(report['endpoints'],frozen['endpoints'])
        reference=reference_for('holdout')
        self.assertEqual((len(reference),len({r['lot_group'] for r in reference})),(148,143))
        expected={root/'predictions'/n/'repeat1'/'predictions.jsonl' for n in names()}
        self.assertEqual(set((root/'predictions').rglob('predictions.jsonl')),expected)
        predictions={};pids=set();count=0
        qwen=read_jsonl(ROOT/'docs/evidence/qwen-holdout/seed17-step250/predictions.jsonl')
        for ep in frozen['endpoints']:
            n=ep['name'];folder=root/'predictions'/n/'repeat1'
            verify_tree(folder)
            rows=read_jsonl(folder/'predictions.jsonl');predictions[n]=rows;count+=len(rows)
            self.assertEqual(verify_predictions(rows,reference),report['models'][n])
            run=json.loads((folder/'run.json').read_text())
            self.assertEqual(run['weight_sha256'],ep['weight_sha256'])
            self.assertEqual(run['manifest_sha256'],plan()['holdout_manifest_sha256'])
            self.assertEqual(run['controller_sha'],report['execution']['controller_sha'])
            self.assertNotIn(run['pid'],pids);pids.add(run['pid'])
            self.assertEqual(paired(qwen[::-1],rows[::-1],reference),report['qwen_paired'][n])
        self.assertEqual((len(pids),count),(6,888))
        for seed in (17,42,123):
            self.assertEqual(bootstrap(reference,predictions[f'seed{seed}-linear'],predictions[f'seed{seed}-lora']),report['lot_group_percentile_95'][str(seed)])
        self.assertEqual(bootstrap(reference,qwen,predictions['seed17-lora']),report['seed17_lora_minus_qwen']['percentile_95'])


if __name__=='__main__':
    unittest.main()
