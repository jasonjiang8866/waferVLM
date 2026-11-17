import json
import unittest

from runpod.storage_preflight import observe


class FakeAPI:
    def __init__(self):
        self.calls = []

    def listing(self, path, key):
        self.calls.append((path, key))
        if path == '/global-volumes':
            return [{'id': 'gsb_1', 'name': 'wafervlm-public-datasets', 'createdByUserId': 'PRIVATE'},
                    {'id': 'gsb_2', 'name': 'PRIVATE'}]
        return [{'id': 'pod1', 'name': 'wafervlm-training', 'status': 'RUNNING',
                 'gpu': {'count': 1}, 'env': {'TOKEN': 'SECRET'},
                 'mounts': {'global': [{'volumeId': 'gsb_1', 'path': '/data', 'extra': 'SECRET'}]}},
                {'id': 'pod2', 'name': 'PRIVATE', 'env': {'TOKEN': 'SECRET'}}]

    def call(self, path):
        self.calls.append(path)
        return {'cpus': [
            {'id': 'cpuA', 'price': {'securePerVcpu': 0.01}, 'ramGbPerVcpu': 2,
             'vcpu': {'min': 2, 'max': 32}, 'availability': 'HIGH',
             'dataCenters': [{'id': 'DC1', 'availability': 'HIGH', 'extra': 'SECRET'}]},
            {'id': 'bad', 'price': {'securePerVcpu': float('nan')}, 'ramGbPerVcpu': 2,
             'vcpu': {'min': 2, 'max': 32}},
        ]}


class StoragePreflightTests(unittest.TestCase):
    def test_get_only_allowlist_and_quote(self):
        api = FakeAPI()
        report = observe(api)
        encoded = json.dumps(report)
        self.assertNotIn('SECRET', encoded)
        self.assertNotIn('PRIVATE', encoded)
        self.assertEqual(report['global_volume_count'], 2)
        self.assertEqual(report['pod_count'], 2)
        self.assertEqual(len(report['cpu_candidates']), 1)
        self.assertEqual(report['cpu_candidates'][0]['compute_usd_per_hour'], 0.02)
        self.assertFalse(report['resources_created'])
        self.assertFalse(report['data_imported'])
        self.assertEqual(api.calls, [('/global-volumes', 'globalVolumes'), ('/pods', 'pods'),
                                    '/catalog/cpus?include=AVAILABILITY&product=POD&vcpuCount=2'])

    def test_invalid_catalog_blocks(self):
        api = FakeAPI()
        api.call = lambda path: {'unexpected': []}
        with self.assertRaises(ValueError):
            observe(api)


if __name__ == '__main__':
    unittest.main()
