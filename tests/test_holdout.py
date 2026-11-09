import importlib.util
import unittest
from pathlib import Path

spec=importlib.util.spec_from_file_location('reserve_holdout',Path(__file__).resolve().parents[1]/'scripts/reserve_holdout.py')
reserve=importlib.util.module_from_spec(spec);spec.loader.exec_module(reserve)


class HoldoutTests(unittest.TestCase):
    def test_excludes_whole_duplicate_connected_lots_and_render_collisions(self):
        def row(id,lot,map,label='center'):
            return dict(id=id,lot=lot,map_sha256=map,label=label)
        records=[row('a','A','map-a'),row('a-copy','B','map-a'),row('b','B','map-b'),
                 row('c','C','map-c'),row('d','D','map-d'),row('e','E','conflict'),
                 row('f','F','conflict','donut'),row('g','G','map-g','near-full')]
        dev=[{**records[0],'image_sha256':'same-render'}]
        def digest(r):return 'same-render' if r['id']=='c' else 'image-'+r['id']
        selected,report=reserve.select_holdout(records,dev,digest)
        self.assertEqual([r['id'] for r in selected],['d','g'])
        self.assertEqual(report['quarantined_rows'],2)
        self.assertEqual(report['render_collisions_skipped'],1)
        self.assertFalse(report['all_eight_classes_available'])
        self.assertEqual((selected,report),reserve.select_holdout(records,dev,digest))
