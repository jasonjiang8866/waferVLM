"""Offline planning estimate only; never modifies the consumed authorization marker."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def estimate(rate=.89):
    if type(rate) not in (int,float) or not 0 < rate <= 1.19:
        raise ValueError('Invalid planning GPU rate')
    cfg=json.loads((ROOT/'configs/seed-confirmation.json').read_text())
    evidence=json.loads((ROOT/'docs/evidence/continuation-42682266526.json').read_text())
    prior=evidence['cleanup']
    # Round history UP to the values already frozen in the protocol.
    usd=5.903039;minutes=491.911
    if prior['estimated_total_usd'] > usd or prior['estimated_total_seconds']/60 > minutes:
        raise ValueError('Prior ledger evidence exceeds the frozen planning reserve')
    cap=(2*cfg['train_work_seconds']+cfg['evaluation_work_seconds']+3*cfg['cleanup_seconds'])/60
    disk_rate=60*.10/(28*24)
    incremental=(rate+disk_rate)*cap/60
    return dict(status='planned_not_launched',approved_usd=30,approved_minutes=1100,
                prior_usd=usd,prior_minutes=minutes,remaining_usd=30-usd,remaining_minutes=1100-minutes,
                additional_minutes=cap,gpu_rate_assumption=rate,disk_hourly_assumption=disk_rate,
                additional_usd=incremental,planned_total_usd=usd+incremental,
                planned_total_minutes=minutes+cap,remaining_after_plan_usd=30-usd-incremental,
                remaining_after_plan_minutes=1100-minutes-cap,
                current_inventory='unverified',current_quote='unverified',invoice=False,
                prior_marker_run=42682266526,prior_marker_artifact=13703281873,
                marker_reconciled_in_controller=False,paid_launch_approved=False)


if __name__ == '__main__':
    print(json.dumps({'rtx4090_ceiling':estimate(.89),'highest_existing_ceiling':estimate(1.19)},indent=2))
