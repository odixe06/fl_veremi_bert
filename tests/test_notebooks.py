"""Validate emitted notebook syntax/JSON arguments and reject incomplete two-rank parents before decode."""
import ast
import glob
import json
import os
import shlex
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.gen_notebook import cells


def args(**kw):
    return SimpleNamespace(**(dict(run='flb-fedavg', slug='flb-fedavg-p2', method='fedavg', rounds=50,
                                  args='', set=['seed=42'], overlay=[], resume_kernel='', max_hours=5) | kw))


@pytest.mark.parametrize('cfg', [args(), args(resume_kernel='owner/flb-fedavg'),
                               args(method='canperfl', rounds=10,
                                    set=["init='\\\"/kaggle/input/**/flb-fedavg/state.pt\\\"'"])])
def test_emitted_notebook_arguments(cfg):
    for kind, source in cells(cfg, 'abc123'):
        if kind != 'code':
            continue
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'cmd' for t in node.targets):
                env = {'budget': 5}
                exec(compile(ast.Module(body=[node], type_ignores=[]), '<notebook>', 'exec'), env)
                argv = shlex.split(env['cmd'].split('&&', 1)[1])
                settings = dict(s.split('=', 1) for s in argv[argv.index('--set') + 1:])
                decoded = {k: json.loads(v) for k, v in settings.items()}
                assert decoded == ({'init': '/kaggle/input/**/flb-fedavg/state.pt'} if cfg.method == 'canperfl' else {'seed': 42})


@pytest.mark.parametrize('rank1_round', [None, 1, 2])
def test_resume_requires_matching_ranks_before_copy(tmp_path, rank1_round):
    parent = tmp_path / 'input/parent/flb-fedavg'
    parent.mkdir(parents=True)
    torch.save({'method': 'fedavg', 'hp': {'rounds': 50}, 'round': 2, 'gen': [torch.zeros(1), torch.zeros(1)]},
               parent / 'state.pt')
    if rank1_round is not None:
        torch.save({'round': rank1_round}, parent / 'state_rank1.pt')
    (parent / 'models').mkdir()
    (parent / 'models/r002.pt').write_bytes(b'snapshot')
    for p in parent.rglob('*'):
        if p.is_file():
            p.chmod(0o444)
    source = next(s for kind, s in cells(args(resume_kernel='owner/flb-fedavg'), 'abc123')
                  if kind == 'code' and s.startswith('prev ='))
    source = source.replace('/kaggle', str(tmp_path))
    env = dict(glob=glob, Path=Path, torch=torch, os=os, shutil=shutil)
    target = tmp_path / 'working/flb-fedavg'
    if rank1_round != 2:
        with pytest.raises((FileNotFoundError, AssertionError)):
            exec(source, env)
        assert not target.exists()
    else:
        exec(source, env)
        assert (target / 'models/r002.pt').read_bytes() == b'snapshot'
        assert (target / 'state.pt').stat().st_mode & 0o200  # read-only parent does not make imported state read-only
