"""A partial live allocation must still launch available methods and keep the rest pending."""
import json
import subprocess

import pytest

from scripts import run_batch as batch


def test_plan_preserves_all_methods_when_quota_covers_only_one(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, 'OUT', tmp_path)
    monkeypatch.setattr(batch, 'quota', lambda: {'owner': 8, **{a: 30 for a in batch.EXCLUDE}})
    monkeypatch.setattr(batch, 'inventory', lambda a: {'gpu_left': 8, 'busy': []})
    monkeypatch.setattr(batch, 'sh', lambda *a, **kw: 'abc123')
    batch.plan()
    plan = json.loads((tmp_path / 'plan.json').read_text())
    assert len(plan['rows']) == 1 and len(plan['deferred']) == 12
    assert {r['method'] for r in plan['rows'] + plan['deferred']} == {m for m, _, _ in batch.METHODS}
    assert plan['rows'][0]['account'] == 'owner'
    assert sum(r['max_hours'] for r in plan['rows']) * batch.MARGIN <= 8


@pytest.mark.parametrize('readable', [False, True])
def test_deferred_launch_requires_readable_free_account(tmp_path, monkeypatch, readable):
    monkeypatch.setattr(batch, 'OUT', tmp_path)
    monkeypatch.setattr(batch, 'quota', lambda: {'owner': 8, **{a: 30 for a in batch.EXCLUDE}})
    def inventory(account):
        assert account == 'owner'
        if not readable:
            raise subprocess.CalledProcessError(1, ['inventory'])
        return {'gpu_left': 8, 'busy': []}
    monkeypatch.setattr(batch, 'inventory', inventory)
    pushed = []
    monkeypatch.setattr(batch, 'push', lambda r, head: pushed.append((r, head)))
    pending = dict(slug='flb-fedprox', run='flb-fedprox', method='fedprox', set=[], part=1, est_hours=6.35)
    rows = {}
    batch.launch_deferred({'head': 'frozen', 'deferred': [pending]}, rows)
    assert len(pushed) == int(readable)
    if readable:
        r, head = pushed[0]
        assert head == 'frozen' and r['account'] == 'owner' and r['max_hours'] * batch.MARGIN <= 8
        assert rows[r['slug']] == r
