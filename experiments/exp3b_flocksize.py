"""Experiment 3b: flock-size sweep, rerun properly for the paper (Section 3).

Three conditions, same seeds in each (paired design):

  single_walled : the original Fig. 4A setup. Every fanboid is pulled toward
                  the captain himself, inside the original walled arena.
  single_open   : same rule, walls switched off (TF = 0). Separates "error
                  grows because the arena is too small for N boids at this PR"
                  from "error grows because of the shared target".
  slots_open    : each fanboid gets its own vee slot behind the captain
                  (spacing > PR, so the spacing condition is satisfied),
                  walls off. Tests whether the delta = 0 jam is what makes
                  error grow with N.

Metrics are PER FANBOID so they don't grow just because there are more
fanboids. The old chi^2 (a SUM over fanboids) is also saved so Fig. 4A can be
redrawn and the linear-vs-quadratic question checked directly.

Usage (from the captainFalcon folder):
    python experiments/exp3b_flocksize.py                 # full run, 50 trials
    python experiments/exp3b_flocksize.py --quick         # 3 trials, smoke test
    python experiments/exp3b_flocksize.py --summarize     # re-plot from the CSV
"""
import os
import sys
import csv
import time
import argparse
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from grainframe import SimOptions, load_dataset, make_arena
from grainframe.data_io import apply_dataset
from grainframe.simulate import runSimulation
from grainframe.metrics import fanChi2
from grainframe.utils import wrapToPi
from grainframe.priority import formationSlots

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, 'results')
CONDITIONS = ('single_walled', 'single_open', 'slots_open')
FIELDS = ['condition', 'nFan', 'trial', 'seed', 'reachedGoal',
          'chi2pos_sum', 'chi2ang_sum',
          'errFinal', 'errSettled', 'spreadFinal', 'headErrFinalDeg',
          'fanCapHits', 'fanFanPairs', 'anyCollision', 'spawnPairs', 'spawnCapHits',
          'minSepPair', 'minSepCap', 'wallFrac']

_G = {}   # per-worker globals


def _init(args):
    opts = SimOptions()
    data = load_dataset(args['dataset'])
    apply_dataset(opts, data)
    opts.fanLeaderVelocityFactor = args['velFactor']
    for kv in args.get('set') or []:
        k, v = kv.split('=', 1)
        cur = getattr(opts, k)                       # AttributeError on typos
        if isinstance(cur, bool):
            setattr(opts, k, v.lower() in ('1', 'true', 'yes'))
        elif isinstance(cur, (int, float)):
            setattr(opts, k, float(v))            # never int(): RboidFactor=0.5 must stay 0.5
        else:
            setattr(opts, k, v)
    if args.get('PR') is not None:
        opts.PR = args['PR']
    if args.get('VR') is not None:
        opts.VR = args['VR']
    arena = make_arena(data, opts)
    _G.update(opts=opts, data=data, arena=arena, args=args)


def _targets(run, slots):
    """(T, n) target positions for every fanboid at every logged step."""
    x = run['x'][:, None]
    y = run['y'][:, None]
    if slots is None:
        return np.broadcast_to(x, run['fanX'].shape), np.broadcast_to(y, run['fanY'].shape)
    th = run['theta'][:, None]
    c, s = np.cos(th), np.sin(th)
    return x + c * slots[0] - s * slots[1], y + s * slots[0] + c * slots[1]


def _collisions(run, radius, start=0):
    """Collisions from logged step `start` on. Fanboids spawn in random clumps
    (initClumped, sigma = 2), so some overlap at t = 0; a grace window keeps
    those spawn overlaps out of the collision counts."""
    X, Y = run['fanX'][start:], run['fanY'][start:]
    run = {'x': run['x'][start:], 'y': run['y'][start:]}
    T, n = X.shape
    dCap = np.hypot(X - run['x'][:, None], Y - run['y'][:, None])
    fanCapHits = int(np.sum(np.any(dCap < radius, axis=0)))
    minSepCap = float(dCap.min())
    fanFanPairs, minSepPair = 0, np.inf
    if n > 1:
        iu, ju = np.triu_indices(n, 1)
        hit = np.zeros(len(iu), dtype=bool)
        for t0 in range(0, T, 500):                      # chunk to keep memory small
            dx = X[t0:t0 + 500, iu] - X[t0:t0 + 500, ju]
            dy = Y[t0:t0 + 500, iu] - Y[t0:t0 + 500, ju]
            d = np.hypot(dx, dy)
            hit |= np.any(d < radius, axis=0)
            minSepPair = min(minSepPair, float(d.min()))
        fanFanPairs = int(hit.sum())
    return fanCapHits, fanFanPairs, minSepPair, minSepCap


def _one(job):
    cond, n, k = job
    opts, data, arena, args = _G['opts'], _G['data'], _G['arena'], _G['args']
    o = SimOptions(**{f: getattr(opts, f) for f in opts.__dataclass_fields__})
    if cond != 'single_walled':
        o.TF = 0.0                                        # walls off
    slots = None
    if cond == 'slots_open':
        spacing = args['slotRatio'] * o.PR
        slots = formationSlots(n, spacing, rows=2, minRadius=spacing,
                               shape='vee', sweep=np.deg2rad(45.0))
    seed = o.seed + 101 * k + 7 * n                      # same scheme as exp3
    r = runSimulation(data, arena, o, nFan=n, seed=seed, slots=slots)

    c2p, c2a = fanChi2(r, o)
    tx, ty = _targets(r, slots)
    err = np.hypot(r['fanX'] - tx, r['fanY'] - ty)       # (T, n)
    T = err.shape[0]
    spread = np.hypot(r['fanX'][-1] - r['x'][-1], r['fanY'][-1] - r['y'][-1])
    fanTh = np.arctan2(r['fanVy'][-1], r['fanVx'][-1])
    dth = np.abs(wrapToPi(fanTh - r['theta'][-1]))
    g = int(round(args['grace'] / o.dt))
    hitsCap, pairs, minPair, minCap = _collisions(r, o.collisionRadius, start=g)
    spawnCap, spawnPairs, _, _ = _collisions(
        {k: r[k][:g + 1] for k in ('x', 'y', 'fanX', 'fanY')}, o.collisionRadius)
    outside = ((r['fanX'] < arena.xSafeMin) | (r['fanX'] > arena.xSafeMax) |
               (r['fanY'] < arena.ySafeMin) | (r['fanY'] > arena.ySafeMax))
    return {
        'condition': cond, 'nFan': n, 'trial': k, 'seed': seed,
        'reachedGoal': int(bool(r['reachedGoal'])),
        'chi2pos_sum': c2p, 'chi2ang_sum': c2a,
        'errFinal': float(err[-1].mean()),
        'errSettled': float(err[T // 2:].mean()),       # second half of the run
        'spreadFinal': float(spread.mean()),
        'headErrFinalDeg': float(np.degrees(dth).mean()),
        'fanCapHits': hitsCap, 'fanFanPairs': pairs,
        'anyCollision': int(hitsCap + pairs > 0),
        'spawnPairs': spawnPairs, 'spawnCapHits': spawnCap,
        'minSepPair': minPair if np.isfinite(minPair) else '',
        'minSepCap': minCap,
        'wallFrac': float(outside.mean()),
    }


# ---------------------------------------------------------------- summary --

def _wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, mid - half), min(1.0, mid + half)


def _ci(x):
    x = np.asarray(x, float)
    m = x.mean()
    h = 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
    return m, h


def _fit(N, y, deg):
    X = np.vander(N, deg + 1, increasing=True)
    beta, res, *_ = np.linalg.lstsq(X, y, rcond=None)
    rss = float(np.sum((y - X @ beta) ** 2))
    n, p = len(y), deg + 1
    aic = n * np.log(rss / n) + 2 * p
    sigma2 = rss / (n - p)
    cov = sigma2 * np.linalg.inv(X.T @ X)
    return beta, np.sqrt(np.diag(cov)), aic


def summarize(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        print('no rows in', path)
        return
    for r in rows:
        for f in ('nFan', 'trial', 'anyCollision', 'fanCapHits', 'fanFanPairs', 'reachedGoal'):
            r[f] = int(r[f])
        for f in ('chi2pos_sum', 'errFinal', 'errSettled', 'spreadFinal',
                  'headErrFinalDeg', 'wallFrac'):
            r[f] = float(r[f])
    conds = [c for c in CONDITIONS if any(r['condition'] == c for r in rows)]
    Ns = sorted({r['nFan'] for r in rows})

    print('Collisions are counted after the grace window (spawn overlaps excluded).')
    print('\nPer-fanboid error to own target (settled = mean over 2nd half of run), 95% CI')
    print('%-14s %4s %5s %18s %18s %10s %8s %22s' % (
        'condition', 'N', 'runs', 'errSettled', 'errFinal', 'headDeg', 'wall%', 'P(any collision)'))
    summ = {}
    for c in conds:
        for n in Ns:
            rr = [r for r in rows if r['condition'] == c and r['nFan'] == n]
            if not rr:
                continue
            es, esh = _ci([r['errSettled'] for r in rr])
            ef, efh = _ci([r['errFinal'] for r in rr])
            hd, _ = _ci([r['headErrFinalDeg'] for r in rr])
            wf = np.mean([r['wallFrac'] for r in rr]) * 100
            p, lo, hi = _wilson(sum(r['anyCollision'] for r in rr), len(rr))
            summ[(c, n)] = (es, esh)
            print('%-14s %4d %5d %9.2f +- %5.2f %9.2f +- %5.2f %10.1f %8.1f %6.2f [%.2f, %.2f]' % (
                c, n, len(rr), es, esh, ef, efh, hd, wf, p, lo, hi))

    if len(Ns) < 4:
        print('\n(fewer than 4 flock sizes, skipping the trend fits)')
        return
    print('\nDoes per-fanboid error grow with N?  OLS slope of errSettled on N (per run)')
    for c in conds:
        rr = [r for r in rows if r['condition'] == c]
        N = np.array([r['nFan'] for r in rr], float)
        y = np.array([r['errSettled'] for r in rr])
        b, se, _ = _fit(N, y, 1)
        print('  %-14s slope %+.3f per fanboid  (95%% CI %+.3f to %+.3f)' % (
            c, b[1], b[1] - 1.96 * se[1], b[1] + 1.96 * se[1]))

    print('\nOld Fig. 4A metric (chi^2 SUM over fanboids): linear vs quadratic in N')
    for c in conds:
        rr = [r for r in rows if r['condition'] == c]
        N = np.array([r['nFan'] for r in rr], float)
        y = np.array([r['chi2pos_sum'] for r in rr])
        b1, _, a1 = _fit(N, y, 1)
        b2, se2, a2 = _fit(N, y, 2)
        print('  %-14s AIC linear %.1f, quadratic %.1f (lower is better); '
              'N^2 coefficient %.2f (95%% CI %.2f to %.2f)' % (
                  c, a1, a2, b2[2], b2[2] - 1.96 * se2[2], b2[2] + 1.96 * se2[2]))
    print('  Note: a SUM over N fanboids of squared distance grows faster than N even')
    print('  if tracking does not degrade (more terms, and a packed flock spreads out).')

    try:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    labels = {'single_walled': 'single target, walled (original)',
              'single_open': 'single target, open field',
              'slots_open': 'vee slots, open field'}
    for c in conds:
        m = [summ[(c, n)][0] for n in Ns if (c, n) in summ]
        h = [summ[(c, n)][1] for n in Ns if (c, n) in summ]
        nn = [n for n in Ns if (c, n) in summ]
        ax[0].errorbar(nn, m, yerr=h, fmt='o-', capsize=3, label=labels[c])
    ax[0].set_xlabel('number of fanboids $N$')
    ax[0].set_ylabel('mean distance to own target (settled)')
    ax[0].legend(fontsize=8)
    for c in conds:
        rr = [r for r in rows if r['condition'] == c]
        N = np.array([r['nFan'] for r in rr], float)
        y = np.array([r['chi2pos_sum'] for r in rr])
        mm = [y[N == n].mean() for n in Ns]
        ax[1].plot(Ns, mm, 'o', label=labels[c])
        if c == 'single_walled':
            b1, _, _ = _fit(N, y, 1)
            b2, _, _ = _fit(N, y, 2)
            g = np.linspace(min(Ns), max(Ns), 100)
            ax[1].plot(g, b1[0] + b1[1] * g, 'k--', lw=1, label='linear fit (original)')
            ax[1].plot(g, b2[0] + b2[1] * g + b2[2] * g * g, 'k:', lw=1, label='quadratic fit (original)')
    ax[1].set_xlabel('number of fanboids $N$')
    ax[1].set_ylabel(r'$\chi^2_{pos}$ summed over fanboids (Fig. 4A metric)')
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    base = os.path.splitext(path)[0]
    fig.savefig(base + '.png', dpi=200)
    fig.savefig(base + '.eps')
    print('\nfigure saved to %s.png / .eps' % base)


# ------------------------------------------------------------------- main --

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--trials', type=int, default=50)
    ap.add_argument('--dataset', default='1')
    ap.add_argument('--N', type=int, nargs='+', default=[1, 2, 3, 5, 7, 12, 20])
    ap.add_argument('--conditions', nargs='+', default=list(CONDITIONS), choices=CONDITIONS)
    ap.add_argument('--slot-ratio', type=float, default=1.25,
                    help='slot spacing as a multiple of PR (must be > 1)')
    ap.add_argument('--vel-factor', type=float, default=0.0,
                    help='fanLeaderVelocityFactor. Set to whatever made Fig. 4A.')
    ap.add_argument('--PR', type=float, default=None,
                    help='protected range (default: config value, 10)')
    ap.add_argument('--VR', type=float, default=None,
                    help='visual range (default: config value, 8)')
    ap.add_argument('--set', nargs='+', default=[], metavar='KEY=VALUE',
                    help='override any SimOptions field, e.g. --set SF=5 RboidFactor=0.5')
    ap.add_argument('--section3', action='store_true',
                    help='the settings Adam\'s Section 3 figures were made with '
                         '(git 9ae2de3, 16 Jul): PR=2.5 SF=5 RboidFactor=0.5')
    ap.add_argument('--grace', type=float, default=2.0,
                    help='seconds at the start ignored for collisions (spawn overlaps)')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--out', default=os.path.join(RESULTS, 'exp3b_flocksize.csv'))
    ap.add_argument('--quick', action='store_true', help='3 trials, for a smoke test')
    ap.add_argument('--summarize', action='store_true', help='only summarize an existing CSV')
    a = ap.parse_args()

    if a.section3:
        a.set = ['PR=2.5', 'SF=5', 'RboidFactor=0.5'] + a.set
    if a.summarize:
        summarize(a.out)
        return
    if a.quick:
        a.trials = 3
        a.out = a.out.replace('.csv', '_quick.csv')
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    jobs = [(c, n, k) for c in a.conditions for n in a.N for k in range(a.trials)]
    args = {'dataset': a.dataset, 'slotRatio': a.slot_ratio, 'velFactor': a.vel_factor,
            'PR': a.PR, 'VR': a.VR, 'grace': a.grace, 'set': a.set}
    _init(args)
    o = _G['opts']
    print('options: PR=%g SF=%g VR=%g fanboid turning radius=%g' % (o.PR, o.SF, o.VR, o.Rboid))
    print('%d runs (%d conditions x %d sizes x %d trials) on %d workers -> %s' % (
        len(jobs), len(a.conditions), len(a.N), a.trials, a.workers, a.out))
    t0 = time.time()
    with open(a.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        if a.workers > 1:
            with Pool(a.workers, initializer=_init, initargs=(args,)) as pool:
                for i, row in enumerate(pool.imap_unordered(_one, jobs, chunksize=2), 1):
                    w.writerow(row); fh.flush()
                    if i % 25 == 0 or i == len(jobs):
                        print('  %4d / %d  (%.0f s)' % (i, len(jobs), time.time() - t0))
        else:
            _init(args)
            for i, job in enumerate(jobs, 1):
                w.writerow(_one(job)); fh.flush()
                if i % 25 == 0 or i == len(jobs):
                    print('  %4d / %d  (%.0f s)' % (i, len(jobs), time.time() - t0))
    summarize(a.out)


if __name__ == '__main__':
    main()
