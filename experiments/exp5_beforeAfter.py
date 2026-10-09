"""Before/after figure: collision incidence under violated vs satisfied spacing.

Reads two study CSVs produced by the SAME experimental design and differing
only in configuration, and overlays them. This is the cleanest statement of
the section's central claim, because everything except the spacing condition
is held fixed: same grid, same seeds, same number of runs.

    python3 experiments/exp5_before_after.py

Expects, in results/:
    exp5_study_oldconfig.csv   spacing condition VIOLATED (delta < P)
    exp5_study.csv             spacing condition SATISFIED (delta > P)

Writes results/exp5_before_after.{eps,png}.
"""

import os
import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp5_tune import wilson

OUTDIR = os.path.join(os.path.dirname(HERE), 'results')
BEFORE = os.path.join(OUTDIR, 'exp5_study_oldconfig.csv')
AFTER = os.path.join(OUTDIR, 'exp5_study.csv')

plt.rcParams.update({'font.size': 9, 'axes.grid': True,
                     'grid.color': '#cccccc', 'grid.linewidth': 0.5,
                     'legend.framealpha': 1.0})


def load(path, guidance='vfield'):
    if not os.path.exists(path):
        sys.exit('missing %s' % path)
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get('status', 'ok') != 'ok':
                continue
            if r['guidance'] != guidance:
                continue
            rows.append((r['geometry'], int(float(r['nFan'])),
                         float(r['totalCollisions'] or 0)))
    hashes = set()
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get('configHash'):
                hashes.add(r['configHash'])
    return rows, hashes


def series(rows, geom):
    ns = sorted(set(n for g, n, _ in rows if g == geom))
    out = []
    for n in ns:
        cell = [c for g, m, c in rows if g == geom and m == n]
        k = sum(1 for c in cell if c > 0)
        p, lo, hi = wilson(k, len(cell))
        out.append((n, p, p - lo, hi - p, len(cell)))
    return out


def main():
    before, hb = load(BEFORE)
    after, ha = load(AFTER)
    print('before: %d runs, config %s' % (len(before), ', '.join(sorted(hb))))
    print('after:  %d runs, config %s' % (len(after), ', '.join(sorted(ha))))
    if hb & ha:
        sys.exit('the two files share a config hash; they are not a '
                 'before/after pair')

    geoms = ['headon', 'cross']
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)

    for ax, geom in zip(axes, geoms):
        for rows, label, colour, marker, dx in (
                (before, r'$\delta < P$  (self-repelling)', '#B22222', 's', -0.10),
                (after, r'$\delta > P$  (satisfied)', '#1F6FB2', 'o', +0.10)):
            s = series(rows, geom)
            if not s:
                continue
            ns = np.array([r[0] for r in s], dtype=float)
            ps = [r[1] for r in s]
            lo = [r[2] for r in s]
            hi = [r[3] for r in s]
            # dodge in log space so the offset looks even across the axis
            ax.errorbar(ns * (1.0 + dx * 0.10), ps, yerr=[lo, hi],
                        marker=marker, ms=4.5, lw=1.3, capsize=2.5,
                        color=colour, label=label, clip_on=False, zorder=3)
        ax.set_xscale('log')
        ax.set_xticks([1, 2, 3, 4, 6, 8, 12])
        ax.get_xaxis().set_major_formatter(
            matplotlib.ticker.ScalarFormatter())
        ax.set_ylim(-0.06, 1.06)
        ax.set_xlabel('wingmen per squadron')
        ax.set_title(geom, fontsize=9)
    axes[0].set_ylabel('P(at least one collision)')
    axes[0].legend(fontsize=7.5, loc='upper left')

    fig.tight_layout()
    for ext in ('eps', 'png'):
        p = os.path.join(OUTDIR, 'exp5_before_after.%s' % ext)
        fig.savefig(p, dpi=200, format=ext)
        print('  wrote exp5_before_after.%s' % ext)
    plt.close(fig)

    print('\n  values (vector field guidance):')
    for geom in geoms:
        print('    %s' % geom)
        b = {r[0]: r for r in series(before, geom)}
        a = {r[0]: r for r in series(after, geom)}
        for n in sorted(set(b) | set(a)):
            bs = '%.2f' % b[n][1] if n in b else '  - '
            as_ = '%.2f' % a[n][1] if n in a else '  - '
            print('      n=%-3d  before %s   after %s' % (n, bs, as_))


if __name__ == '__main__':
    main()