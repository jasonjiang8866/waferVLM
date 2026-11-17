"""GET-only storage inventory; secret stays on the Actions runner."""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from runpod.provider import Runpod, finite


def project_resource(row):
    return 'wafer' in str(row.get('name', '')).lower()


def observe(api):
    volumes = api.listing('/global-volumes', 'globalVolumes')
    pods = api.listing('/pods', 'pods')
    result = api.call('/catalog/cpus?include=AVAILABILITY&product=POD&vcpuCount=2')
    if not isinstance(result, dict) or not isinstance(result.get('cpus'), list):
        raise ValueError('Invalid CPU catalog')
    candidates = []
    for cpu in result['cpus']:
        price = cpu.get('price', {}).get('securePerVcpu')
        ram = cpu.get('ramGbPerVcpu')
        bounds = cpu.get('vcpu', {})
        if not (finite(price, 0.000001) and finite(ram, 0.001)
                and finite(bounds.get('min'), 1) and finite(bounds.get('max'), 1)
                and bounds['min'] <= 2 <= bounds['max']):
            continue
        candidates.append({
            'id': cpu['id'], 'vcpu_count': 2, 'memory_gb': 2 * ram,
            'compute_usd_per_hour': round(2 * price, 8),
            'availability': cpu.get('availability', 'UNKNOWN'),
            'data_centers': [{k: dc.get(k) for k in ('id', 'availability')}
                             for dc in cpu.get('dataCenters', [])],
        })
    candidates.sort(key=lambda item: (item['compute_usd_per_hour'], item['id']))
    relevant_volumes = [{k: row.get(k) for k in ('id', 'name')}
                        for row in volumes if project_resource(row)]
    relevant_pods = []
    for pod in pods:
        if project_resource(pod):
            mount = pod.get('mounts', {}).get('global') or {}
            relevant_pods.append({
                **{k: pod.get(k) for k in ('id', 'name', 'status', 'cost', 'dataCenterId')},
                'kind': 'gpu' if pod.get('gpu') else 'cpu',
                'global_mount': {k: mount.get(k) for k in ('id', 'path')},
            })
    return {
        'schema_version': 1, 'observed_at': datetime.now(timezone.utc).isoformat(),
        'source_sha': os.environ.get('GITHUB_SHA'),
        'run_id': os.environ.get('GITHUB_RUN_ID'),
        'mode': 'read-only', 'resources_created': False, 'data_imported': False,
        'global_volume_count': len(volumes), 'project_global_volumes': relevant_volumes,
        'pod_count': len(pods), 'project_pods': relevant_pods,
        'cpu_candidates': candidates,
        'limitations': [
            'Catalog availability is not a reservation or successful allocation.',
            'Compute quote excludes storage and request fees.',
            'Other projects are counted but their resource details are omitted.',
            'No existing Pod may be reused without checking its workload and mount.',
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('storage-observation'))
    args = parser.parse_args()
    report = observe(Runpod())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'inventory.json').write_text(json.dumps(report, indent=2) + '\n')
    # Only the explicit allowlist above is printable; never dump raw API responses.
    print(json.dumps(report, indent=2))
    summary = ('## Wafer dataset storage preflight\n\n'
               f"Global Volumes: {report['global_volume_count']}; "
               f"wafer-related: {len(report['project_global_volumes'])}.\n\n"
               f"2-vCPU catalog candidates: {len(report['cpu_candidates'])}.\n\n"
               'GET-only; no resources created, no data imported. '
               'See sanitized inventory.json for quotes and availability.\n')
    (args.output / 'summary.md').write_text(summary)


if __name__ == '__main__':
    main()
