"""Leader path fidelity: waypoint law against vector field guidance.

Replaces the box plot in Sec. 5.1. That plot was not defensible: leader
cross-track RMS has zero variance across seeds, because the seeds perturb
wingman placement only and the leaders do not react to wingmen. Every cell
held one value repeated 280 times, so there was no interquartile range,
no whiskers and no outliers to draw.

What the reader actually wants is the trajectory itself, which is what the
surrounding text describes: the waypoint law overshoots each waypoint and
weaves, the vector field law rotates onto the line and stays there. That is
one deterministic run per condition, which is exactly what the data supports.

Writes:
    results/figs/exp5_guidance_tracks.{eps,pdf,png}
    results/exp5_guidance_numbers.tex     (the four RMS values, as a table)

Usage:
    python experiments/exp5_guidance_figure.py
    python experiments/exp5_guidance_figure.py --nfan 4 --seed 0
"""
import argparse
import contextlib
import importlib
import importlib.util
import io
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from grainframe.guidance import crossTrackError          # noqa: E402

OUTDIR = os.path.join(ROOT, 'results')
FIGDIR = os.path.join(OUTDIR, 'figs')

WAYPOINT_COLOR = '#d55e00'
VFIELD_COLOR = '#0072b2'
PATH_COLOR = '0.55'


EXP = os.path.join(HERE, 'exp5_priority.py')     # same file exp5_study.py drives


def loadExperiment():
    """Fresh module per run, exactly as exp5_study.loadExperiment does."""
    spec = importlib.util.spec_from_file_location('exp5run', EXP)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def runOne(geometry, guidance, nFan, seed):
    """One simulation, returning the pieces needed to draw a track."""
    m = loadExperiment()
    m.GEOMETRY = geometry
    m.GUIDANCE = guidance
    m.N_FAN = nFan
    m.SEED = seed
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = m.run(plot=False)
    return r


def leaderRMS(r):
    """Cross-track RMS for both leaders, against their own reference paths."""
    out = []
    for xk, yk, dk in (('xR', 'yR', 'dataR'), ('xG', 'yG', 'dataG')):
        e = crossTrackError(np.asarray(r[xk]), np.asarray(r[yk]),
                            r[dk].refPath)
        out.append(float(np.sqrt(np.mean(np.square(e)))))
    return out


def draw(ax, runs, geometry):
    """Both guidance laws for one geometry, on shared axes."""
    ref = runs['vfield']['dataR'].refPath
    ax.plot(ref[:, 0], ref[:, 1], color=PATH_COLOR, lw=1.0, ls='--',
            zorder=1, label='reference path')
    ref2 = runs['vfield']['dataG'].refPath
    ax.plot(ref2[:, 0], ref2[:, 1], color=PATH_COLOR, lw=1.0, ls='--',
            zorder=1)

    for guid, colour, label in (('waypoint', WAYPOINT_COLOR, 'waypoint'),
                                ('vfield', VFIELD_COLOR, 'vector field')):
        r = runs[guid]
        ax.plot(r['xR'], r['yR'], color=colour, lw=1.3, zorder=3,
                label=label)
        ax.plot(r['xG'], r['yG'], color=colour, lw=1.3, zorder=3, alpha=0.75)

    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel('x')
    ax.grid(alpha=0.25, lw=0.5)


def writeTable(stats, path):
    """The four RMS values as a table, since they carry no spread."""
    lines = [
        r'\begin{tabular}{lcc}',
        r'\hline',
        r'Geometry & Waypoint & Vector field \\',
        r'\hline',
    ]
    for geom, label in (('headon', 'Head-on'), ('cross', 'Crossing')):
        wp = stats[(geom, 'waypoint')]
        vf = stats[(geom, 'vfield')]
        lines.append(r'%s & $%.2f$ & $%.2f$ \\' % (label, wp, vf))
    lines += [r'\hline', r'\end{tabular}']
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nfan', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(FIGDIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    stats = {}

    for ax, geometry in zip(axes, ('headon', 'cross')):
        runs = {}
        for guid in ('waypoint', 'vfield'):
            print('  running %-7s %-8s ...' % (geometry, guid), flush=True)
            r = runOne(geometry, guid, args.nfan, args.seed)
            runs[guid] = r
            rms = leaderRMS(r)
            stats[(geometry, guid)] = float(np.mean(rms))
            print('      leader cross-track RMS: %.3f, %.3f' % tuple(rms))
        draw(ax, runs, geometry)

    axes[0].set_ylabel('y')
    axes[1].legend(fontsize=7.5, frameon=False, loc='best')
    fig.tight_layout()

    for ext in ('eps', 'pdf', 'png'):
        p = os.path.join(FIGDIR, 'exp5_guidance_tracks.' + ext)
        fig.savefig(p, dpi=200, bbox_inches='tight')
        print('  wrote', p)
    plt.close(fig)

    tab = os.path.join(OUTDIR, 'exp5_guidance_numbers.tex')
    writeTable(stats, tab)
    print('  wrote', tab)

    print('\n  RMS summary (pooled over both leaders):')
    for geom in ('headon', 'cross'):
        wp, vf = stats[(geom, 'waypoint')], stats[(geom, 'vfield')]
        print('    %-7s  %6.2f -> %6.2f   (factor %.1f)'
              % (geom, wp, vf, wp / vf if vf else float('nan')))


if __name__ == '__main__':
    main()