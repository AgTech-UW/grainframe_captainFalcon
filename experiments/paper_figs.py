"""Paper figures, redrawn WITHOUT titles (titles belong in the LaTeX captions).

Run from the captainFalcon folder:

    python experiments/paper_figs.py

Writes .pdf and .eps for each figure into results/final_figures/:

    fig_leader_distractors   old george_1A  leader with and without distractors
    fig_fanboid_tracks       old george_2A  leader and 10 fanboids
    fig_fanboid_errors       old george_3A  final pose error of each fanboid
    fig_flocksize            replaces george_4A, from the exp3b CSVs
    exp5_formations          vee, echelon and column slots, P and delta marked
    exp5_hierarchy           the total order and the two yield actions kept

The simulation figures use the same datasets, options and seeds as
exp1_ab_distractors.py and exp2_fanboids.py, so they are the same runs as
before, only drawn differently. Wording follows the paper: "leader", not
"Captain Falcon". No figure uses transparency, so the .eps files look the
same as the .pdf files.
"""
import os
import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

from grainframe import SimOptions, load_dataset, make_arena
from grainframe.data_io import apply_dataset
from grainframe.simulate import runSimulation
from grainframe.metrics import fanFinalErrors
from grainframe.priority import formationSlots

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, 'results', 'final_figures')
EXP3B = os.path.join(HERE, 'results', 'exp3b')   # from exp3b_flocksize.py --section3

# Okabe-Ito, as in the Section 4 figures
BLUE, ORANGE, GREEN = '#0072b2', '#d55e00', '#009e73'
RED, GOLD, GREY = '#c0392b', '#c9922b', '#4c4c4c'
LIGHTBLUE, PALEBLUE, PALERED = '#5a9bd4', '#dbe9f5', '#f6e0dd'

plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 9, 'legend.fontsize': 8,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.grid': True, 'grid.color': '#e4e4e4', 'grid.alpha': 1.0, 'grid.linewidth': 0.5,
    'axes.axisbelow': True, 'figure.dpi': 150, 'legend.framealpha': 1.0,
})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ('pdf', 'eps', 'png'):
        fig.savefig(os.path.join(OUT, '%s.%s' % (name, ext)), bbox_inches='tight',
                    dpi=200 if ext == 'png' else None)
    plt.close(fig)
    print('  wrote results/final_figures/%s.{pdf,eps,png}' % name)


def panel(ax, letter):
    ax.text(0.0, 1.03, '(%s)' % letter, transform=ax.transAxes,
            fontsize=9, fontweight='bold', va='bottom', ha='left')


def pose(ax, q, color, L=5.0):
    ax.quiver(q[0], q[1], L * np.cos(q[2]), L * np.sin(q[2]), color=color,
              angles='xy', scale_units='xy', scale=1, width=0.006, zorder=6)


# The settings the original Section 3 figures were made with (git 9ae2de3,
# 16 Jul 2026). The defaults in config.py were changed later for Section 4.
SECTION3 = dict(PR=2.5, SF=5.0, RboidFactor=0.5, fanLeaderVelocityFactor=0.0)


def setup():
    opts = SimOptions(**SECTION3)
    data = load_dataset('1')
    apply_dataset(opts, data)
    return opts, data, make_arena(data, opts)


# ------------------------------------------------------------ Section 3 sims

def figLeaderDistractors():
    opts, data, arena = setup()
    runA = runSimulation(data, arena, opts, nDistract=0)          # as exp1
    runB = runSimulation(data, arena, opts, nDistract=opts.N)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for b in range(runB['boidX'].shape[1]):
        ax.plot(runB['boidX'][:, b], runB['boidY'][:, b], '-', color='0.78',
                lw=0.6, zorder=1, label='distractors' if b == 0 else None)
    ax.plot(runB['boidX'][-1], runB['boidY'][-1], '.', color='0.5', ms=5, zorder=1)
    ax.plot(data.refPath[:, 0], data.refPath[:, 1], '-', color=BLUE, lw=2,
            label='reference Dubins path', zorder=2)
    ax.plot(runA['x'], runA['y'], 'k--', lw=1.3, label='leader, no distractors', zorder=3)
    ax.plot(runB['x'], runB['y'], '-', color=GREEN, lw=1.5,
            label='leader, with distractors', zorder=3)
    ax.plot(data.wayPts[:, 0], data.wayPts[:, 1], 's', color=RED, mfc='none',
            ms=7, label='waypoints', zorder=4)
    pose(ax, data.qi, GREEN)
    pose(ax, data.qf, RED)
    ax.set_aspect('equal')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.legend(loc='lower left', fontsize=7.5)
    save(fig, 'fig_leader_distractors')


def figDubinsBoid():
    """Vector redraw of the old MATLAB dubins_boid.png: one boid (the leader,
    no other boids present) led along the Dubins path by its waypoints."""
    opts, data, arena = setup()
    run = runSimulation(data, arena, opts, nDistract=0)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot(data.refPath[:, 0], data.refPath[:, 1], '-', color=BLUE, lw=2.2,
            label='Dubins path', zorder=2)
    ax.plot(run['x'], run['y'], '--', color=GREEN, lw=1.6, label='boid', zorder=3)
    ax.plot(data.wayPts[:, 0], data.wayPts[:, 1], 's', color=RED, mfc='none',
            ms=7, label='waypoints', zorder=4)
    pose(ax, data.qi, GREEN)
    pose(ax, data.qf, RED)
    ax.set_aspect('equal')
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.legend(loc='lower left', fontsize=7.5)
    save(fig, 'dubins_boid')


def fanRun():
    opts, data, arena = setup()
    return opts, data, runSimulation(data, arena, opts, nFan=opts.nFanShowcase)  # as exp2


def figFanboidTracks(opts, data, runC):
    n = runC['fanX'].shape[1]
    cols = plt.cm.viridis(np.linspace(0.15, 0.9, n))
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(data.refPath[:, 0], data.refPath[:, 1], '-', color=BLUE, lw=2,
            label='reference Dubins path', zorder=2)
    ax.plot(runC['x'], runC['y'], 'k--', lw=1.3, label='leader', zorder=3)
    for b in range(n):
        ax.plot(runC['fanX'][:, b], runC['fanY'][:, b], '-', color=cols[b], lw=0.8,
                zorder=1, label='fanboids' if b == 0 else None)
    ax.plot(runC['fanX0'], runC['fanY0'], 'o', color='0.4', mfc='none', ms=5,
            label='initial positions', zorder=4)
    psi = np.arctan2(runC['fanVy'][-1], runC['fanVx'][-1])
    ax.quiver(runC['fanX'][-1], runC['fanY'][-1], 2 * np.cos(psi), 2 * np.sin(psi),
              color=cols, angles='xy', scale_units='xy', scale=1, width=0.004, zorder=5)
    pose(ax, data.qf, RED)
    ax.set_aspect('equal')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.legend(loc='lower left', fontsize=7.5)
    save(fig, 'fig_fanboid_tracks')
    return cols


def figFanboidErrors(opts, runC, cols):
    # Three panels instead of a bar chart with a second y-axis: a shared
    # x-axis with two unrelated y-scales invents a correlation.
    dx, dy, d, dth = fanFinalErrors(runC)
    n = len(d)
    idx = np.arange(1, n + 1)
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.6),
                           gridspec_kw={'width_ratios': [1.1, 1, 1]})
    ax[0].axhline(0, color='0.8', lw=0.8)
    ax[0].axvline(0, color='0.8', lw=0.8)
    ax[0].add_patch(Circle((0, 0), opts.sigmaPos, fill=False, ls='--', color=RED,
                           label='capture tolerance'))
    ax[0].scatter(dx, dy, c=cols, s=30, zorder=3)
    ax[0].set_xlabel(r'$x$ error')
    ax[0].set_ylabel(r'$y$ error')
    ax[0].set_aspect('equal', adjustable='datalim')
    ax[0].legend(loc='lower right', fontsize=7)
    ax[1].bar(idx, d, color=cols)
    ax[1].set_xlabel('fanboid')
    ax[1].set_ylabel('distance to leader')
    ax[2].plot(idx, np.degrees(np.abs(dth)), '^', color=GREY, ms=6)
    ax[2].set_xlabel('fanboid')
    ax[2].set_ylabel(r'heading error ($^{\circ}$)')
    ax[2].set_ylim(bottom=0)
    for a in ax[1:]:
        a.set_xticks(idx)
    for a, l in zip(ax, 'abc'):
        panel(a, l)
    fig.tight_layout()
    save(fig, 'fig_fanboid_errors')


# ------------------------------------------------------- Section 3 flock size

def readCSV(name, cond='single_walled'):
    rows = [r for r in csv.DictReader(open(os.path.join(EXP3B, name)))
            if r['condition'] == cond]
    for r in rows:
        r['nFan'] = int(r['nFan'])
        for k in ('errSettled', 'fanCapHits', 'fanFanPairs'):
            r[k] = float(r[k])
    return rows


def meanCI(x):
    x = np.asarray(x, float)
    return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, p - max(0.0, mid - h), min(1.0, mid + h) - p


def figFlocksize():
    series = [('exp3b_section3_P2.5.csv', r'$P = 2.5$ (one fanboid turning radius, as above)', BLUE, 'o', '-'),
              ('exp3b_section3_P7.5.csv', r'$P = 7.5$ (three turning radii)', ORANGE, 's', '--')]
    fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.6))
    for fname, lab, col, mk, ls in series:
        rows = readCSV(fname)
        Ns = sorted({r['nFan'] for r in rows})
        err, cap, pairs = [], [], []
        for n in Ns:
            rr = [r for r in rows if r['nFan'] == n]
            err.append(meanCI([r['errSettled'] for r in rr]))
            cap.append(wilson(sum(r['fanCapHits'] > 0 for r in rr), len(rr)))
            pairs.append(meanCI([r['fanFanPairs'] for r in rr]))
        kw = dict(color=col, marker=mk, ls=ls, lw=1.5, ms=4.5, capsize=2.5, elinewidth=0.9)
        ax[0].errorbar(Ns, [m for m, _ in err], yerr=[h for _, h in err], label=lab, **kw)
        ax[1].errorbar(Ns, [p for p, _, _ in cap],
                       yerr=[[lo for _, lo, _ in cap], [hi for _, _, hi in cap]], **kw)
        ax[2].errorbar(Ns, [m for m, _ in pairs], yerr=[h for _, h in pairs], **kw)
    ax[0].set_ylabel('distance to leader')
    ax[0].set_ylim(bottom=0)
    ax[1].set_ylabel('share of runs, fanboid hits leader')
    ax[1].set_ylim(0, 1)
    ax[2].set_ylabel('fanboid pairs colliding per run')
    for a, l in zip(ax, 'abc'):
        a.set_xlabel('number of fanboids $N$')
        a.set_xticks([1, 5, 10, 15, 20])
        panel(a, l)
    h, lab = ax[0].get_legend_handles_labels()
    fig.legend(h, lab, loc='upper center', ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout()
    save(fig, 'fig_flocksize')


# ------------------------------------------------------------- Section 4 art

def figFormations(P=20.0, ratio=1.25, n=6):
    """Slots from grainframe.priority.formationSlots, the code the runs used,
    at the operating point (P = 20, delta = 25). All three panels share one
    scale, so the protected range is drawn the same size in each."""
    delta = ratio * P
    specs = [('vee', 2), ('echelon', 1), ('column', 2)]
    geo, ext = [], []
    for shape, _ in specs:
        a, b = formationSlots(n, delta, rows=2, minRadius=P, shape=shape,
                              sweep=np.deg2rad(45.0))
        xs = list(a) + [0, a[0] - P, a[0] + P]
        ys = list(b) + [0, b[0] - P, b[0] + P]
        geo.append((a, b))
        ext.append((min(xs) - 8, max(xs) + 14, min(ys) - 10, max(ys) + 22))
    y0 = min(e[2] for e in ext)
    y1 = max(e[3] for e in ext)
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.0),
                             gridspec_kw={'width_ratios': [e[1] - e[0] for e in ext]})
    for ax, (shape, other), (along, lat), e, l in zip(axes, specs, geo, ext, 'abc'):
        ax.add_patch(Circle((along[0], lat[0]), P, fc=PALEBLUE, ec=BLUE, lw=0.7,
                            ls=':', zorder=2))
        ax.annotate('$P$', (along[0], lat[0] + P + 1.5), fontsize=8, color=BLUE,
                    ha='center', va='bottom')
        ax.plot([0], [0], marker=(3, 0, -90), ms=9, color=RED, ls='none', zorder=6)
        ax.annotate('leader', (0, -4), fontsize=6.5, color=RED, ha='center', va='top')
        for k, (a, b) in enumerate(zip(along, lat)):
            ax.plot([a], [b], marker=(3, 0, -90), ms=6.5, color=BLUE, ls='none', zorder=5)
            ax.annotate('%d' % (k + 1), (a + 3.5, b + 3), fontsize=6, color=GREY,
                        ha='left', va='bottom', zorder=6)
        a0, b0, a1, b1 = along[0], lat[0], along[other], lat[other]
        ax.annotate('', xy=(a1, b1), xytext=(a0, b0), zorder=7,
                    arrowprops=dict(arrowstyle='<->', lw=1.1, color=ORANGE,
                                    shrinkA=4, shrinkB=4))
        nx, ny = (b1 - b0), -(a1 - a0)          # label on the outside of the arm
        if shape == 'column':
            nx, ny = 0.0, -1.0
        sc = 5.0 / np.hypot(nx, ny)
        ax.annotate(r'$\delta$', ((a0 + a1) / 2 + sc * nx, (b0 + b1) / 2 + sc * ny),
                    fontsize=9, color=ORANGE, ha='center', va='center', zorder=7)
        ax.text(0.04, 0.97, '(%s) %s' % (l, shape), transform=ax.transAxes,
                fontsize=9, va='top', ha='left')
        ax.set_xlim(e[0], e[1])
        ax.set_ylim(y0, y1)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.tight_layout()
    save(fig, 'exp5_formations')


def figHierarchy():
    """exp5_conceptFig.figHierarchy without panel titles. It never had the
    head-on rotation panel, which the ablation removed from the system."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0),
                             gridspec_kw={'width_ratios': [1.15, 1]})
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    def box(x, y, w, h, label, fc):
        ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=fc, lw=1.0, zorder=3))
        ax.annotate(label, (x + w / 2, y + h / 2), fontsize=7.5, color='white',
                    ha='center', va='center', zorder=4)

    ax.annotate('tier 1: leaders', (0.2, 9.2), fontsize=8, color=GREY)
    box(0.4, 8.0, 3.9, 0.95, 'red leader', RED)
    box(4.9, 8.0, 3.9, 0.95, 'gold leader', GOLD)
    ax.annotate('tier 2: wingmen', (0.2, 6.9), fontsize=8, color=GREY)
    for k, lab in enumerate(['red wingman 1', 'red wingman 2', r'$\cdots$']):
        box(0.4, 5.1 - k, 3.9, 0.85, lab, BLUE)
    for k, lab in enumerate(['gold wingman 1', 'gold wingman 2', r'$\cdots$']):
        box(4.9, 5.1 - k, 3.9, 0.85, lab, LIGHTBLUE)
    ax.annotate('', xy=(9.45, 3.1), xytext=(9.45, 8.95),
                arrowprops=dict(arrowstyle='-|>', lw=1.2, color=GREY))
    ax.annotate('decreasing precedence', (9.75, 6.0), fontsize=7.5, color=GREY,
                rotation=90, va='center', ha='center')
    ax.annotate(r'key $=(\mathrm{tier},\ \mathrm{squadron},\ \mathrm{index})$',
                (4.6, 1.9), fontsize=8, ha='center')
    ax.annotate('total, so every pair is decided', (4.6, 1.2), fontsize=7.5,
                ha='center', color=GREY)
    ax.text(0.0, 1.0, '(a)', transform=ax.transAxes, fontsize=9,
            fontweight='bold', va='top')

    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.text(0.0, 1.0, '(b)', transform=ax.transAxes, fontsize=9,
            fontweight='bold', va='top')
    ax.annotate('wingman yields to wingman', (1.0, 9.3), fontsize=8, color=GREY)
    ax.plot([1.2], [7.9], marker=(3, 0, -90), ms=11, color=BLUE, ls='none')
    ax.plot([5.2], [7.9], marker=(3, 0, 90), ms=11, color=LIGHTBLUE, ls='none')
    ax.add_patch(FancyArrowPatch((2.9, 7.9), (2.0, 7.9), arrowstyle='-|>',
                                 mutation_scale=11, lw=1.4, color=ORANGE))
    ax.annotate('brake', (2.45, 8.35), fontsize=7.5, color=ORANGE, ha='center')
    ax.annotate('both can manoeuvre, and the order lets\n'
                'only one of the pair brake, so neither\n'
                "is left stationary in the other's path",
                (0.3, 6.05), fontsize=7, color=GREY)
    ax.annotate('wingman yields to leader', (0.3, 4.8), fontsize=8, color=GREY)
    ax.add_patch(Rectangle((0.9, 2.05), 5.4, 1.6, fc=PALERED, ec=RED, lw=0.7,
                           ls=':', zorder=1))
    ax.annotate('leader corridor', (3.6, 2.25), fontsize=6.5, color=RED, ha='center')
    ax.plot([6.0], [2.85], marker=(3, 0, 90), ms=13, color=RED, ls='none', zorder=4)
    ax.plot([2.6], [2.85], marker=(3, 0, -90), ms=10, color=BLUE, ls='none', zorder=4)
    ax.add_patch(FancyArrowPatch((2.6, 3.35), (2.6, 4.15), arrowstyle='-|>',
                                 mutation_scale=11, lw=1.4, color=GREEN))
    ax.annotate('clear laterally, at full speed', (2.95, 4.05), fontsize=7.5,
                color=GREEN, va='center')
    ax.annotate('a leader holds speed and does not yield to a\n'
                'wingman, so braking here guarantees the\n'
                'collision rather than avoiding it',
                (0.3, 0.75), fontsize=7, color=GREY)
    save(fig, 'exp5_hierarchy')


def main():
    print('paper figures ->', OUT)
    figDubinsBoid()
    figLeaderDistractors()
    opts, data, runC = fanRun()
    cols = figFanboidTracks(opts, data, runC)
    figFanboidErrors(opts, runC, cols)
    if os.path.exists(os.path.join(EXP3B, 'exp3b_section3_P7.5.csv')):
        figFlocksize()
    else:
        print('  skipped fig_flocksize: run the two exp3b --section3 commands first')
    figFormations()
    figHierarchy()


if __name__ == '__main__':
    main()
