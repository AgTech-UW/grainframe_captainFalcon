"""Figures for the experiment-5 paper section.

Stdlib + numpy + matplotlib only -- no pandas, no scipy required. (If scipy
happens to be installed the logistic fit uses it; otherwise a small built-in
optimiser is used instead and the result is the same to plotting precision.)

Reads whatever is present in results/ and writes figures to results/figs/.
Nothing is fabricated: if the data for a figure is missing, the figure is
skipped and the reason is printed.

    python3 experiments/exp5_paperFigures.py
    python3 experiments/exp5_paperFigures.py --results results --out results/figs
    python3 experiments/exp5_paperFigures.py --spacing-csv results/exp5_spacing.csv

Inputs it looks for
-------------------
  exp5_study.csv     long-form runs: geometry, guidance, nFan, seed, per-pair
                     collision counts, RMS, firstCollisionRelCPA
  exp5_capacity.csv  held-out capacity runs (mode/param/value/label/split ...)
  exp5_tune.csv      sensitivity / grid rows; used for the spacing-ratio figure
                     if it carries a FORM_SPACING (or PR) sweep

The spacing-ratio figure needs a sweep that varies slot spacing and/or the
wingman protected range. Point --spacing-csv at that file when the run lands.
"""
import argparse
import csv
import math
import os
import sys
from collections import OrderedDict, defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

PAIRS = ['leaderLeader', 'fanOwnCap', 'fanForeignCap', 'fanFanSame',
         'fanFanCross']
PAIRLABEL = {
    'leaderLeader':  'leader--leader',
    'fanOwnCap':     'wingman--own leader',
    'fanForeignCap': 'wingman--opposing leader',
    'fanFanSame':    'wingman--squadmate',
    'fanFanCross':   'wingman--opposing wingman',
}
# colourblind-safe, ordered to match PAIRS
PAIRCOLOR = ['#4c4c4c', '#d55e00', '#e69f00', '#0072b2', '#009e73']

plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.5,
    'axes.axisbelow': True,
    'figure.dpi': 150,
})


# ------------------------------------------------------------------ csv layer
#
# A "table" here is just a list of dicts, one per row, values left as strings.
# num()/where()/groupsum() are the three operations the figures actually need.

def readTable(path):
    """Read a CSV into a list of dicts. Returns None if absent or empty."""
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    bad = [r for r in rows if r.get('status', 'ok') not in ('ok', '')]
    if bad:
        print('  note: dropping %d non-ok rows from %s'
              % (len(bad), os.path.basename(path)))
    rows = [r for r in rows if r.get('status', 'ok') in ('ok', '')]
    return rows or None


def num(row, key, default=float('nan')):
    """Numeric value of a field, NaN for blank/missing/unparseable."""
    v = row.get(key, '')
    if v is None or v == '':
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def where(rows, **eq):
    """Rows whose string fields match all the given key=value pairs."""
    return [r for r in rows
            if all(r.get(k) == v for k, v in eq.items())]


def byKey(rows, key, cast=float):
    """Group rows by a numeric field. Returns OrderedDict sorted by key."""
    g = defaultdict(list)
    for r in rows:
        v = num(r, key)
        if not math.isnan(v):
            g[cast(v)].append(r)
    return OrderedDict(sorted(g.items()))


def colSum(rows, key):
    return sum(0.0 if math.isnan(v) else v
               for v in (num(r, key) for r in rows))


def hasCol(rows, key):
    return bool(rows) and key in rows[0]


def anyCollision(rows):
    """(k, n): runs with at least one collision, and total runs."""
    n = len(rows)
    k = sum(1 for r in rows if num(r, 'totalCollisions', 0.0) > 0)
    return k, n


# ---------------------------------------------------------------- statistics

def wilson(k, n, z=1.959963985):
    """Wilson score interval for a binomial proportion. Returns (lo, hi)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def _nelderMead(fn, x0, step=0.5, tol=1e-8, maxIter=2000):
    """Minimal 2-D Nelder-Mead, so scipy stays optional."""
    x0 = np.asarray(x0, float)
    simplex = [x0]
    for i in range(len(x0)):
        p = x0.copy()
        p[i] += step * (abs(p[i]) + 1.0)
        simplex.append(p)
    simplex = np.array(simplex)
    fv = np.array([fn(p) for p in simplex])
    for _ in range(maxIter):
        order = np.argsort(fv)
        simplex, fv = simplex[order], fv[order]
        if abs(fv[-1] - fv[0]) < tol:
            break
        centroid = simplex[:-1].mean(axis=0)
        xr = centroid + (centroid - simplex[-1])
        fr = fn(xr)
        if fr < fv[0]:
            xe = centroid + 2.0 * (centroid - simplex[-1])
            fe = fn(xe)
            simplex[-1], fv[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < fv[-2]:
            simplex[-1], fv[-1] = xr, fr
        else:
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = fn(xc)
            if fc < fv[-1]:
                simplex[-1], fv[-1] = xc, fc
            else:
                simplex = simplex[0] + 0.5 * (simplex - simplex[0])
                fv = np.array([fn(p) for p in simplex])
    i = int(np.argmin(fv))
    return simplex[i], fv[i]


def logisticFit(x, k, n):
    """Fit P = 1/(1+exp(-b(x-x50))) by binomial MLE. Returns (x50, b) or None."""
    x = np.asarray(x, float)
    k = np.asarray(k, float)
    n = np.asarray(n, float)
    if len(x) < 3 or k.sum() == 0 or (k == n).all():
        return None

    def nll(th):
        x50, b = th
        z = np.clip(b * (x - x50), -40, 40)
        p = np.clip(1.0 / (1.0 + np.exp(-z)), 1e-9, 1 - 1e-9)
        return -float(np.sum(k * np.log(p) + (n - k) * np.log(1 - p)))

    best, bestF = None, np.inf
    for x0 in (float(np.median(x)), float(x.mean())):
        for b0 in (0.3, 1.0):
            try:
                from scipy.optimize import minimize
                r = minimize(nll, [x0, b0], method='Nelder-Mead')
                th, f = r.x, float(r.fun)
            except ImportError:
                th, f = _nelderMead(nll, [x0, b0])
            if np.isfinite(f) and f < bestF:
                best, bestF = th, f
    if best is None:
        return None
    return float(best[0]), float(best[1])


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ('pdf', 'png', 'eps'):
        fig.savefig(os.path.join(out, '%s.%s' % (name, ext)),
                    bbox_inches='tight')
    plt.close(fig)
    print('  wrote %s.{pdf,png,eps}' % name)


# ------------------------------------------------------------------ figure 1

def figProbability(study, capacity, out):
    """P(at least one collision) against squadron size, Wilson intervals."""
    if study is None:
        print('  skip probability: no exp5_study.csv')
        return

    def series(rows, ax, color, marker, label, ls='-'):
        g = byKey(rows, 'nFan', int)
        if not g:
            return
        sizes = list(g.keys())
        kn = [anyCollision(g[s]) for s in sizes]
        k = [a for a, _ in kn]
        n = [b for _, b in kn]
        p = np.array([a / b for a, b in kn])
        ci = np.array([wilson(a, b) for a, b in kn])
        ax.errorbar(sizes, p, yerr=[p - ci[:, 0], ci[:, 1] - p],
                    fmt=marker + ls, color=color, ms=4, lw=1.2,
                    capsize=2.5, elinewidth=0.9,
                    label=label % n[0])
        fit = logisticFit(sizes, k, n)
        if fit:
            x50, b = fit
            xs = np.linspace(min(sizes), max(sizes), 200)
            ax.plot(xs, 1 / (1 + np.exp(-b * (xs - x50))), '-',
                    color=color, lw=0.8, alpha=0.45)
            if min(sizes) <= x50 <= max(sizes):
                ax.axvline(x50, color=color, ls=':', lw=0.8, alpha=0.6)
                ax.annotate(r'$\hat{N}_{50}=%.1f$' % x50, (x50, 0.30),
                            color=color, fontsize=7, ha='center',
                            bbox=dict(fc='white', ec='none', alpha=0.75,
                                      pad=1))

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    for ax, geom in zip(axes, ['headon', 'cross']):
        sub = where(study, geometry=geom)
        if not sub:
            continue
        for guid, col, mk in (('vfield', '#0072b2', 'o'),
                              ('waypoint', '#d55e00', 's')):
            series(where(sub, guidance=guid), ax, col, mk,
                   guid + r' ($n=%d$/size)')
        # corrected-config overlay, if the held-out capacity run has rows
        if capacity is not None and geom == 'headon':
            c = capacity
            if hasCol(c, 'geometry'):
                c = where(c, geometry=geom)
            if c:
                series(c, ax, '#009e73', '^',
                       r'held-out, corrected ($n=%d$)', ls='--')
        ax.set_title({'headon': 'head-on', 'cross': 'crossing'}[geom])
        ax.set_xlabel('wingmen per squadron')
        ax.set_ylim(-0.04, 1.04)
        ax.legend(loc='upper left', framealpha=0.9)
    axes[0].set_ylabel(r'$\hat{P}$(at least one collision)')
    save(fig, out, 'exp5_collision_probability')


# ------------------------------------------------------------------ figure 2

def figComposition(study, out):
    """Collision composition by pair type against squadron size."""
    if study is None:
        print('  skip composition: no exp5_study.csv')
        return
    have = [c for c in PAIRS if hasCol(study, c)]
    if not have:
        print('  skip composition: no per-pair columns')
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    for ax, geom in zip(axes, ['headon', 'cross']):
        sub = where(study, geometry=geom, guidance='vfield')
        if not sub:
            continue
        g = byKey(sub, 'nFan', int)
        sizes = list(g.keys())
        xs = np.arange(len(sizes))
        counts = {c: np.array([colSum(g[s], c) for s in sizes]) for c in have}
        tot = np.sum([counts[c] for c in have], axis=0)
        safe = np.where(tot == 0, np.nan, tot)
        bottom = np.zeros(len(sizes))
        for c in have:
            frac = np.nan_to_num(counts[c] / safe)
            ax.bar(xs, frac, bottom=bottom, width=0.68,
                   color=PAIRCOLOR[PAIRS.index(c)], label=PAIRLABEL[c],
                   edgecolor='white', linewidth=0.4)
            bottom += frac
        for j, t in enumerate(tot):
            ax.text(j, 1.02, '%d' % int(t), ha='center', fontsize=6.5,
                    color='0.35')
        ax.set_xticks(xs)
        ax.set_xticklabels([str(s) for s in sizes])
        ax.set_xlabel('wingmen per squadron')
        ax.set_ylim(0, 1.12)
        ax.set_title({'headon': 'head-on', 'cross': 'crossing'}[geom] +
                     ' (vector field)')
        ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('share of all collisions')
    axes[1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                   frameon=False)
    save(fig, out, 'exp5_type_composition')


# ------------------------------------------------------------------ figure 3

def figTiming(study, out):
    """Timing of first collision relative to leader closest approach."""
    if study is None or not hasCol(study, 'firstCollisionRelCPA'):
        print('  skip timing: no firstCollisionRelCPA column')
        return
    v = np.array([num(r, 'firstCollisionRelCPA') for r in study])
    v = v[~np.isnan(v)]
    if len(v) < 5:
        print('  skip timing: only %d first-collision events' % len(v))
        return
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    lim = float(np.percentile(np.abs(v), 99)) * 1.05
    bins = np.linspace(-lim, lim, 31)
    ax.hist(v[v < 0], bins=bins, color='#0072b2', alpha=0.85,
            label='before CPA (n=%d)' % int((v < 0).sum()))
    ax.hist(v[v >= 0], bins=bins, color='#d55e00', alpha=0.85,
            label='at or after CPA (n=%d)' % int((v >= 0).sum()))
    ax.axvline(0, color='k', lw=1.0)
    ax.text(0, ax.get_ylim()[1] * 0.62, ' leader CPA', fontsize=7,
            va='top', ha='left', color='0.3')
    ax.set_xlabel('time of first collision relative to leader CPA')
    ax.set_ylabel('runs')
    ax.legend(frameon=False, loc='upper right')
    print('    -> %.0f%% of first collisions occur at or after CPA'
          % (100.0 * float((v >= 0).mean())))
    save(fig, out, 'exp5_collision_timing')


# ------------------------------------------------------------------ figure 4

def figGuidance(study, out):
    """Cross-track RMS, waypoint against vector field guidance."""
    if study is None or not hasCol(study, 'redRMS'):
        print('  skip guidance: no RMS columns')
        return
    rmscols = [c for c in ('redRMS', 'goldRMS') if hasCol(study, c)]
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    labels, data, colors = [], [], []
    for geom in ('headon', 'cross'):
        for guid, col in (('waypoint', '#d55e00'), ('vfield', '#0072b2')):
            sub = where(study, geometry=geom, guidance=guid)
            if not sub:
                continue
            vals = np.array([num(r, c) for r in sub for c in rmscols])
            vals = vals[~np.isnan(vals)]
            if not len(vals):
                continue
            labels.append('%s\n%s' % (geom, guid))
            data.append(vals)
            colors.append(col)
    if not data:
        print('  skip guidance: no rows')
        return
    try:
        bp = ax.boxplot(data, tick_labels=labels, showfliers=False,
                        widths=0.55, patch_artist=True,
                        medianprops=dict(color='k', lw=1.1))
    except TypeError:      # matplotlib < 3.9
        bp = ax.boxplot(data, labels=labels, showfliers=False, widths=0.55,
                        patch_artist=True,
                        medianprops=dict(color='k', lw=1.1))
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
        patch.set_edgecolor(c)
    ax.set_ylabel('leader cross-track RMS')
    ax.set_yscale('log')
    ax.grid(axis='x', visible=False)
    save(fig, out, 'exp5_guidance_comparison')


# ------------------------------------------------------------------ figure 5

def figSpacing(path, out):
    """Collisions against the spacing--protection ratio delta/P.

    Needs a sweep CSV carrying a slot-spacing and/or protected-range column.
    Recognised column names are listed in SPACING_COLS / PR_COLS below; if the
    sweep writes something else, add it there.
    """
    SPACING_COLS = ['formSpacing', 'FORM_SPACING', 'slotSpacing', 'delta',
                    'spacing']
    PR_COLS = ['fanPR', 'FAN_OWN_PR', 'fanOwnPR', 'PR', 'protectedRange']
    rows = readTable(path)
    if rows is None:
        print('  skip spacing ratio: no sweep CSV yet (--spacing-csv)')
        return

    def pull(names):
        """Value per row for the first recognised name, wide or long form."""
        for c in names:
            if hasCol(rows, c):
                v = [num(r, c) for r in rows]
                if any(not math.isnan(x) for x in v):
                    return v
        if hasCol(rows, 'param') and hasCol(rows, 'value'):
            for c in names:
                v = [num(r, 'value') if r.get('param') == c else float('nan')
                     for r in rows]
                if any(not math.isnan(x) for x in v):
                    return v
        return None

    delta = pull(SPACING_COLS)
    pr = pull(PR_COLS)
    if delta is None or pr is None:
        print('  skip spacing ratio: %s has no recognised spacing/PR columns\n'
              '    (looked for %s and %s)'
              % (os.path.basename(path), SPACING_COLS, PR_COLS))
        return

    byRatio = defaultdict(list)
    for r, d, p in zip(rows, delta, pr):
        if math.isnan(d) or math.isnan(p) or p == 0:
            continue
        c = num(r, 'totalCollisions')
        if not math.isnan(c):
            byRatio[round(d / p, 6)].append(c)
    if not byRatio:
        print('  skip spacing ratio: no rows with both spacing and PR')
        return

    ratios = sorted(byRatio)
    mean = np.array([np.mean(byRatio[x]) for x in ratios])
    sem = np.array([(np.std(byRatio[x], ddof=1) / math.sqrt(len(byRatio[x])))
                    if len(byRatio[x]) > 1 else 0.0 for x in ratios])
    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    if min(ratios) < 1.0:
        ax.axvspan(min(ratios), 1.0, color='#d55e00', alpha=0.10)
    ax.axvline(1.0, color='#d55e00', ls='--', lw=1.1)
    ax.errorbar(ratios, mean, yerr=sem, fmt='o-', color='#0072b2', ms=4,
                lw=1.3, capsize=2.5, elinewidth=0.9)
    ax.annotate(r'$\delta = P$', (1.0, ax.get_ylim()[1]), fontsize=8,
                color='#d55e00', ha='right', va='top', rotation=90)
    ax.set_xlabel(r'spacing--protection ratio $\delta / P$')
    ax.set_ylabel('mean collisions per run')
    save(fig, out, 'exp5_spacing_ratio')


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results')
    ap.add_argument('--out', default=os.path.join('results', 'figs'))
    ap.add_argument('--spacing-csv', default=None,
                    help='sweep CSV varying slot spacing / protected range')
    args = ap.parse_args()

    R = args.results
    study = readTable(os.path.join(R, 'exp5_study.csv'))
    capacity = readTable(os.path.join(R, 'exp5_capacity.csv'))
    if study is None and capacity is None:
        sys.exit('no usable CSVs under %s' % R)

    spacing = args.spacing_csv
    if spacing is None:
        for cand in ('exp5_spacing.csv', 'exp5_tune.csv', 'exp5_grid.csv'):
            p = os.path.join(R, cand)
            if os.path.exists(p) and os.path.getsize(p) > 0:
                spacing = p
                break

    print('figures ->', args.out)
    figProbability(study, capacity, args.out)
    figComposition(study, args.out)
    figTiming(study, args.out)
    figGuidance(study, args.out)
    figSpacing(spacing, args.out)


if __name__ == '__main__':
    main()