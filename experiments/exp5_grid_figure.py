"""Plot the PR x spacing-ratio grid as a heatmap.

This is the figure for the section's principal finding, and nothing in
exp5_study.py or exp5_tune.py produces it -- `gridreport` prints a text table
only. Run after `exp5_tune.py grid`.

    python experiments/exp5_grid_figure.py

Writes results/exp5_spacing_grid.{eps,png}.
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
GRIDCSV = os.path.join(OUTDIR, 'exp5_grid2_flatNS.csv')

# autolayout OFF: it cannot cope with a colorbar spanning several axes, and
# silently overlaps the last panel when it tries.
plt.rcParams.update({'font.size': 9, 'legend.framealpha': 1.0})


def main():
    if not os.path.exists(GRIDCSV):
        sys.exit('no %s -- run `exp5_tune.py grid` first' % GRIDCSV)

    rows = []
    with open(GRIDCSV) as f:
        for r in csv.DictReader(f):
            if r.get('status', 'ok') != 'ok':
                continue
            r['nFan'] = int(float(r['nFan']))
            r['tc'] = float(r['totalCollisions']) if r['totalCollisions'] else 0.0
            r['pr'] = float(r['label'].split('_')[0][2:])
            r['ratio'] = float(r['label'].split('_x')[1])
            rows.append(r)

    sizes = sorted(set(r['nFan'] for r in rows))
    prs = sorted(set(r['pr'] for r in rows))
    rts = sorted(set(r['ratio'] for r in rows))
    print('read %d runs: %d sizes, %d PR values, %d ratios'
          % (len(rows), len(sizes), len(prs), len(rts)))

    fig, axes = plt.subplots(1, len(sizes), figsize=(4.6 * len(sizes), 3.6),
                             squeeze=False,
                             gridspec_kw=dict(left=0.07, right=0.86,
                                              bottom=0.15, top=0.90,
                                              wspace=0.22))

    # Shared colour scale across panels: the comparison between squadron sizes
    # is part of the point, and per-panel autoscaling would hide it.
    grids = []
    for n in sizes:
        g = np.full((len(prs), len(rts)), np.nan)
        for i, pr in enumerate(prs):
            for j, rt in enumerate(rts):
                cell = [r for r in rows
                        if r['nFan'] == n and r['pr'] == pr and r['ratio'] == rt]
                if cell:
                    k = sum(1 for r in cell if r['tc'] > 0)
                    g[i, j] = k / len(cell)
        grids.append(g)
    vmax = float(np.nanmax(grids))

    for ax, n, g in zip(axes[0], sizes, grids):
        im = ax.imshow(g, origin='lower', aspect='auto', cmap='RdYlGn_r',
                       vmin=0.0, vmax=vmax)
        ax.set_xticks(range(len(rts)))
        ax.set_xticklabels(['%g' % r for r in rts])
        ax.set_yticks(range(len(prs)))
        ax.set_yticklabels(['%g' % p for p in prs])
        ax.set_xlabel(r'slot spacing / protected range,  $\delta / P$')
        ax.set_title('%d wingmen per squadron' % n, fontsize=9)

        # annotate every cell: the numbers are the result, the colour is
        # only there to make the trend visible at a glance
        for i in range(len(prs)):
            for j in range(len(rts)):
                if np.isnan(g[i, j]):
                    continue
                val = g[i, j]
                ax.text(j, i, '%.2f' % val, ha='center', va='center',
                        fontsize=7.5,
                        color='white' if val > 0.6 * vmax else 'black')

        # Mark the self-repelling boundary. At ratio 1.0 adjacent slots sit
        # exactly AT the protected range, so that column and anything left of
        # it is broken by construction. The line goes just LEFT of the 1.0
        # column so that the column itself falls on the broken side.
        if 1.0 in rts:
            ax.axvline(rts.index(1.0) - 0.5, color='k', lw=1.6, ls='--')
            ax.text(rts.index(1.0) - 0.40, len(prs) - 0.65,
                    'self-repelling', fontsize=6.5, rotation=90,
                    va='top', ha='left')

    axes[0][0].set_ylabel(r'protected range  $P$')
    cax = fig.add_axes([0.885, 0.15, 0.018, 0.75])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('P(at least one collision)')

    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ('eps', 'png'):
        path = os.path.join(OUTDIR, 'exp5_spacing_grid.%s' % ext)
        fig.savefig(path, dpi=200, format=ext)
        print('  wrote exp5_spacing_grid.%s' % ext)
    plt.close(fig)

    # the marginal effect of each axis, for the text
    print('\n  marginal means (averaged over the other axis):')
    for n, g in zip(sizes, grids):
        print('    nFan=%d  by P:     %s' % (n, '  '.join(
            '%g:%.2f' % (p, v) for p, v in zip(prs, np.nanmean(g, axis=1)))))
        print('    nFan=%d  by ratio: %s' % (n, '  '.join(
            '%g:%.2f' % (r, v) for r, v in zip(rts, np.nanmean(g, axis=0)))))


if __name__ == '__main__':
    main()