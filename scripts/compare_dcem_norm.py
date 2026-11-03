"""
Offline DCEM value-normalization comparison on frozen TD-MPC agents.

For each TD-MPC run <source_exp>/ar<AR>_s<seed> and each std mode (none | graph | detach):
a fresh MLPDecoder (same init for every mode) is trained through DCEM for n_rounds x
updates_per_round updates from the run's replay buffer, with one evaluation episode
after every round (round 0 = untrained decoder). One TD-MPC planner episode per agent
is logged as a reference.

    python scripts/compare_dcem_norm.py --ars 1 2 3 4 --seeds 1 2

Output: experiments/o2_base/<task>/<exp>/ar<AR>_std-<mode>_s<seed>/
    config.yaml, meta.json, updates.csv, iters.csv, eval.csv, decoder_final.pt
and <exp>/reference_tdmpc.csv.
"""

import os
os.environ['MUJOCO_GL'] = 'egl'
import warnings
warnings.filterwarnings('ignore')

import argparse
import csv
import datetime
import pickle
import sys
import time
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'tdmpc' / 'src'))

import torch
from omegaconf import OmegaConf

from lml import LML
import algorithm.helper as h
from algorithm.helper import ReplayBuffer
from env import make_env
from o2_base.agent import TDMPC_Dec
from o2_base.decoder import MLPDecoder
from o2_base.dcem import estimate_value_with_grad
from experiments_utils import EXPERIMENTS_ROOT, make_run_dir

TASK = 'cheetah-run'
ITER_KEYS = ['value_std_med', 'value_std_p10', 'value_std_p90', 'eff_sharpness_med',
             'eff_elites_med', 'eff_elites_p10', 'eff_elites_p90', 'u_std']


# ── DCEM with value normalization (semantics from the notebook) ──────────────

def dcem_norm(agent, obs, std_mode, init_std=1.0, stats=None):
    cfg = agent.cfg
    B, N, K, d = obs.shape[0], cfg.latent_num_samples, cfg.latent_num_elites, cfg.latent_action_dim
    z = agent.model.h(obs).detach()
    z_rep = z.repeat_interleave(N, dim=0)
    u_mean = torch.zeros(B, d, device=agent.device)
    u_std = init_std * torch.ones(B, d, device=agent.device)
    for i in range(cfg.dcem_iterations):
        u = u_mean.unsqueeze(1) + u_std.unsqueeze(1) * torch.randn(B, N, d, device=agent.device)
        actions = agent.decoder(u.view(B * N, d), z_rep)                        # [H, B*N, A]
        value = estimate_value_with_grad(agent, z_rep, actions, cfg.horizon).view(B, N)
        if std_mode == 'none':
            x = value
        else:
            s = value.std(dim=1, keepdim=True)
            if std_mode == 'detach':
                s = s.detach()
            x = (value - value.mean(dim=1, keepdim=True)) / (s + 1e-8)
        w = LML(N=K, verbose=0)(cfg.lml_temperature * x) / K                    # [B, N]
        if stats is not None:                                                    # diagnostics only, no grad
            with torch.no_grad():
                stats.append({'iter': i,
                              'value_std': value.std(1),
                              'eff_sharpness': (cfg.lml_temperature * x).std(1),
                              'eff_elites': 1 / w.pow(2).sum(1)})
        w = w.unsqueeze(-1)
        u_mean = (w * u).sum(dim=1)
        u_std = ((w * (u - u_mean.unsqueeze(1)) ** 2).sum(dim=1) + 1e-6).sqrt()
        if stats is not None:
            stats[-1]['u_std'] = u_std.detach().mean()
    return u_mean, u_std


def decoder_update(agent, buffer, std_mode, batch, clip, stats=None):
    obs = buffer.sample()[0][:batch]
    h.set_requires_grad(agent.model, False)
    u_mean, _ = dcem_norm(agent, obs, std_mode, stats=stats)
    z = agent.model.h(obs).detach()
    value = estimate_value_with_grad(agent, z, agent.decoder(u_mean, z), agent.cfg.horizon)
    loss = -value.mean()
    agent.dec_optim.zero_grad(set_to_none=True)
    loss.backward()
    g = torch.nn.utils.clip_grad_norm_(agent.decoder.parameters(), clip).item()
    agent.dec_optim.step()
    h.set_requires_grad(agent.model, True)
    return value.mean().item(), loss.item(), g


@torch.no_grad()
def run_episode(agent, env, std_mode, stats=None):
    obs, done, ret = env.reset(), False, 0.0
    while not done:
        o = torch.tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
        u_mean, _ = dcem_norm(agent, o, std_mode, stats=stats)
        a = agent.decoder(u_mean, agent.model.h(o))[0, 0]
        obs, r, done, _ = env.step(a.cpu().numpy())
        ret += r
    return ret


@torch.no_grad()
def run_tdmpc_episode(agent, env):
    obs, done, ret, t = env.reset(), False, 0.0, 0
    while not done:
        a = agent.plan(obs, eval_mode=True, step=10**9, t0=(t == 0))
        obs, r, done, _ = env.step(a.cpu().numpy())
        ret += r
        t += 1
    return ret


# ── Stats aggregation ────────────────────────────────────────────────────────

def summarize_batch(stats):
    """stats from one dcem_norm call on a batch -> one row per iteration (quantiles over B)."""
    q = lambda x, p: torch.quantile(x.float(), p).item()
    return [{'iter': s['iter'],
             'value_std_med': q(s['value_std'], .5), 'value_std_p10': q(s['value_std'], .1),
             'value_std_p90': q(s['value_std'], .9),
             'eff_sharpness_med': q(s['eff_sharpness'], .5),
             'eff_elites_med': q(s['eff_elites'], .5), 'eff_elites_p10': q(s['eff_elites'], .1),
             'eff_elites_p90': q(s['eff_elites'], .9),
             'u_std': s['u_std'].item()} for s in stats]


def summarize_episode(stats, n_iters):
    """stats from all B=1 dcem_norm calls of an episode -> per-iteration mean over steps."""
    rows = []
    for i in range(n_iters):
        rs = summarize_batch([s for s in stats if s['iter'] == i])
        rows.append({'iter': i, **{k: sum(r[k] for r in rs) / len(rs) for k in ITER_KEYS}})
    return rows


class CSV:
    def __init__(self, path, fields):
        self.f = open(path, 'w', newline='')
        self.w = csv.DictWriter(self.f, fieldnames=fields)
        self.w.writeheader()

    def write(self, row):
        self.w.writerow(row)
        self.f.flush()

    def close(self):
        self.f.close()


# ── Loading ──────────────────────────────────────────────────────────────────

class _Stub:
    def __init__(self, *a, **k): pass
    def __setstate__(self, s): pass


class _LegacyUnpickler(pickle.Unpickler):
    """Kaggle buffers pickle buffer.cfg with a newer omegaconf (omegaconf.tupleconfig); stub what can't be imported."""
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            if module.startswith('omegaconf'):
                return _Stub
            raise


_legacy_pickle = types.ModuleType('legacy_pickle')
_legacy_pickle.__dict__.update(pickle.__dict__)
_legacy_pickle.Unpickler = _LegacyUnpickler


def load_run(run_dir, args):
    cfg = OmegaConf.load(run_dir / 'config.yaml')
    if not isinstance(cfg.episode_length, int):
        cfg.episode_length = 1000 // cfg.action_repeat
    make_env(cfg)                                       # sets obs_shape / action_dim (missing in legacy configs)
    cfg.latent_action_dim = cfg.horizon * cfg.action_dim
    cfg.dec_hidden_dim, cfg.dec_num_layers = 256, 2
    cfg.latent_num_samples, cfg.latent_num_elites = args.n_samples, args.n_elites
    cfg.dcem_iterations = args.dcem_iterations
    cfg.lml_temperature = args.lml_temperature
    agent = TDMPC_Dec(cfg)
    agent.load(run_dir / 'model_final.pt')
    buffer = ReplayBuffer(cfg)
    state = torch.load(run_dir / 'buffer.pth', map_location=cfg.device, weights_only=False,
                       pickle_module=_legacy_pickle)
    state.pop('cfg', None)                              # keep our cfg (Kaggle's may not unpickle)
    buffer.__dict__.update(state)
    return cfg, agent, buffer


def reset_decoder(agent, seed):
    cfg = agent.cfg
    torch.manual_seed(seed)
    agent.decoder = MLPDecoder(cfg.latent_action_dim, cfg.latent_dim, cfg.horizon, cfg.action_dim,
                               hidden=cfg.dec_hidden_dim, num_layers=cfg.dec_num_layers).to(agent.device)
    agent.dec_optim = torch.optim.Adam(agent.decoder.parameters(), lr=cfg.lr)


# ── Main loop ────────────────────────────────────────────────────────────────

def run_mode(agent, cfg, buffer, mode, args, out_dir):
    reset_decoder(agent, args.dec_seed)                 # also seeds DCEM noise and buffer sampling
    env = make_env(cfg)                                 # fresh env -> same initial states for every mode
    n_it = cfg.dcem_iterations
    upd_csv = CSV(out_dir / 'updates.csv', ['round', 'update', 'value_mean', 'loss', 'grad_norm', 'time'])
    it_csv = CSV(out_dir / 'iters.csv', ['round', 'update', 'iter'] + ITER_KEYS)
    ev_csv = CSV(out_dir / 'eval.csv', ['round', 'n_updates', 'episode_return', 'eval_eff_elites_it0',
                                        'eval_eff_elites_last', 'eval_value_std_it0', 'eval_value_std_last'])

    def evaluate(rnd, n_updates):
        stats = []
        ret = run_episode(agent, env, mode, stats=stats)
        rows = summarize_episode(stats, n_it)
        for r in rows:
            it_csv.write({'round': rnd, 'update': -1, **r})
        ev_csv.write({'round': rnd, 'n_updates': n_updates, 'episode_return': ret,
                      'eval_eff_elites_it0': rows[0]['eff_elites_med'],
                      'eval_eff_elites_last': rows[-1]['eff_elites_med'],
                      'eval_value_std_it0': rows[0]['value_std_med'],
                      'eval_value_std_last': rows[-1]['value_std_med']})
        print(f'    round {rnd:2d}  updates {n_updates:4d}  return {ret:7.1f}  '
              f'eff_elites it0/last {rows[0]["eff_elites_med"]:.1f}/{rows[-1]["eff_elites_med"]:.1f}')

    t0 = time.time()
    evaluate(0, 0)
    n_updates = 0
    for rnd in range(1, args.n_rounds + 1):
        log_all = rnd in (1, args.n_rounds)
        for k in range(args.updates_per_round):
            n_updates += 1
            stats = [] if (log_all or k % 10 == 0) else None
            v, loss, g = decoder_update(agent, buffer, mode, args.batch, args.dec_grad_clip, stats=stats)
            upd_csv.write({'round': rnd, 'update': n_updates, 'value_mean': v, 'loss': loss,
                           'grad_norm': g, 'time': time.time() - t0})
            if stats is not None:
                for r in summarize_batch(stats):
                    it_csv.write({'round': rnd, 'update': n_updates, **r})
        evaluate(rnd, n_updates)

    torch.save(agent.decoder.state_dict(), out_dir / 'decoder_final.pt')
    for c in (upd_csv, it_csv, ev_csv):
        c.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source_exp', default='2026-11-03_50ep-ar-sweep')
    p.add_argument('--ars', type=int, nargs='+', default=[1, 2, 3, 4])
    p.add_argument('--seeds', type=int, nargs='+', default=[1, 2])
    p.add_argument('--modes', nargs='+', default=['none', 'graph', 'detach'], choices=['none', 'graph', 'detach'])
    p.add_argument('--n_rounds', type=int, default=10)
    p.add_argument('--updates_per_round', type=int, default=50)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--n_samples', type=int, default=32)
    p.add_argument('--n_elites', type=int, default=8)
    p.add_argument('--dcem_iterations', type=int, default=5)
    p.add_argument('--lml_temperature', type=float, default=1.0)
    p.add_argument('--dec_grad_clip', type=float, default=10.0)
    p.add_argument('--dec_seed', type=int, default=0)
    p.add_argument('--exp', default=f'{datetime.date.today().isoformat()}_offline-dcem-norm')
    args = p.parse_args()

    src_root = EXPERIMENTS_ROOT / 'tdmpc' / TASK / args.source_exp
    exp_dir = EXPERIMENTS_ROOT / 'o2_base' / TASK / args.exp
    exp_dir.mkdir(parents=True, exist_ok=True)
    ref_path = exp_dir / 'reference_tdmpc.csv'
    ref_csv = CSV(ref_path, ['ar', 'seed', 'tdmpc_return']) if not ref_path.exists() else None
    if ref_csv is None:
        sys.exit(f'{ref_path} already exists, pick a new --exp')

    for ar in args.ars:
        for seed in args.seeds:
            src = src_root / f'ar{ar}_s{seed}'
            print(f'\n=== {src} ===')
            cfg, agent, buffer = load_run(src, args)

            ref = run_tdmpc_episode(agent, make_env(cfg))
            ref_csv.write({'ar': ar, 'seed': seed, 'tdmpc_return': ref})
            print(f'  TD-MPC reference return: {ref:.1f}')

            for mode in args.modes:
                run_cfg = OmegaConf.create({'source_run': str(src), 'mode': mode, **vars(args),
                                            'agent_cfg': OmegaConf.to_container(cfg)})
                out_dir = make_run_dir('o2_base', TASK, args.exp, f'ar{ar}_std-{mode}', seed, run_cfg)
                print(f'  mode {mode} -> {out_dir}')
                run_mode(agent, cfg, buffer, mode, args, out_dir)
    ref_csv.close()


if __name__ == '__main__':
    main()
