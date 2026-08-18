"""Concept figures for the experiment-5 paper section.

These illustrate the METHOD rather than the results, so they depend on no CSV
and can be regenerated at any time. Stdlib + numpy + matplotlib only.

    python3 experiments/exp5_conceptFigures.py
    python3 experiments/exp5_conceptFigures.py --out results/figs

Produces, into --out:
    exp5_formations          slot layouts, with protected ranges drawn
    exp5_vector_field        the Nelson guidance field and a recovering vehicle
    exp5_potentials          Khatib against the flat Reynolds separation rule
    exp5_hierarchy           the total order and the yield actions it implies

The formation geometry mirrors formationSlots() in exp5_safe2.py. If that
function changes, change SLOTS below to match.
"""
import argparse
import math
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

RED, GOLD = '#c0392b', '#c9922b'
BLUE, GREY = '#0072b2', '#4c4c4c'
GREEN, ORANGE = '#009e73', '#d55e00'

plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.grid': True,
    'grid.alpha': 0.22,
    'grid.linewidth': 0.5,
    'axes.axisbelow': True,
    'figure.dpi': 150,
})


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ('pdf', 'png', 'eps'):
        fig.savefig(os.path.join(out, '%s.%s' % (name, ext)),
                    bbox_inches='tight')
    plt.close(fig)
    print('  wrote %s.{pdf,png,eps}' % name)


# ------------------------------------------------------------- formation slots

def formationSlots(n, kind, spacing=3.0, standoff=3.0, sweep=0.7, cols=3,
                   side=+1):
    """Slot offsets (along, lateral) in the leader's body frame.

    along < 0 is behind the leader, lateral > 0 is to the leader's left.
    Mirrors formationSlots() in exp5_safe2.py.
    """
    if kind == 'ring':
        rMin = n * spacing / (2.0 * math.pi)
        rad = max(standoff, rMin)
        ang = 2.0 * math.pi * np.arange(n) / n
        return [(rad * math.cos(a), rad * math.sin(a)) for a in ang]

    slots = []
    for j in range(n):
        if kind == 'vee':
            rank = j // 2 + 1
            s = 1.0 if j % 2 == 0 else -1.0
            d = standoff + (rank - 1) * spacing
            slots.append((-d, s * d * sweep))
        elif kind == 'echelon':
            d = standoff + j * spacing
            slots.append((-d, side * d * sweep))
        elif kind == 'column':
            slots.append((-(standoff + j * spacing), 0.0))
        elif kind == 'line':
            rank = j // 2 + 1
            s = 1.0 if j % 2 == 0 else -1.0
            slots.append((-standoff, s * rank * spacing))
        elif kind == 'grid':
            row, col = divmod(j, cols)
            lat = (col - (cols - 1) / 2.0) * spacing
            slots.append((-(standoff + row * spacing), lat))
        else:
            raise ValueError('unknown formation %r' % kind)
    return slots


def figFormations(out, n=6, spacing=3.0, protect=2.0):
    """Slot layouts, with the protected range shown on one representative slot.

    Only one protected circle is drawn per panel. Drawing one around every
    wingman invites the misreading that two circles touching is a violation:
    the separation rule tests the centre-to-centre distance against P, so the
    condition that matters is delta > P, not delta > 2P.
    """
    kinds = ['vee', 'column', 'line', 'echelon', 'grid', 'ring']
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.2))
    for ax, kind in zip(axes.ravel(), kinds):
        slots = formationSlots(n, kind, spacing=spacing)
        ax.plot([0], [0], marker=(3, 0, -90), ms=12, color=RED, ls='none',
                zorder=6)
        ax.annotate('leader', (0.0, 0.95), fontsize=6.5, color=RED,
                    ha='center', va='bottom')
        for i, (a, l) in enumerate(slots):
            ax.plot([a], [l], marker=(3, 0, -90), ms=8, color=BLUE,
                    ls='none', zorder=5)
            ax.annotate('%d' % (i + 1), (a, l - 0.95), fontsize=6,
                        color=GREY, ha='center', va='top')
        # protected range on slot 1 only
        a0, l0 = slots[0]
        ax.add_patch(Circle((a0, l0), protect, fc=BLUE, ec=BLUE, lw=0.6,
                            ls=':', alpha=0.13, zorder=2))
        ax.annotate('$P$', (a0, l0 + protect + 0.35), fontsize=6.5,
                    color=BLUE, ha='center')
        # delta between the first two slots
        if len(slots) > 1:
            a1, l1 = slots[1]
            ax.annotate('', xy=(a1, l1), xytext=(a0, l0),
                        arrowprops=dict(arrowstyle='<->', lw=0.9,
                                        color=ORANGE, shrinkA=4, shrinkB=4))
            ax.annotate(r'$\delta$', ((a0 + a1) / 2, (l0 + l1) / 2 - 0.75),
                        fontsize=7, color=ORANGE, ha='center')
        ax.set_title(kind, fontsize=9)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        axs = [a for a, _ in slots] + [0]
        lat = [l for _, l in slots] + [0]
        pad = protect + 1.2
        cx = (min(axs) + max(axs)) / 2.0
        cy = (min(lat) + max(lat)) / 2.0
        half = max(max(axs) - min(axs), max(lat) - min(lat)) / 2.0 + pad
        ax.set_xlim(cx - half * 1.15, cx + half * 1.15)
        ax.set_ylim(cy - half, cy + half)
    fig.suptitle('Leader-referenced formation slots, %d wingmen, '
                 r'$\delta = %.0f$, $P = %.0f$' % (n, spacing, protect),
                 fontsize=9, y=0.99)
    save(fig, out, 'exp5_formations')


# --------------------------------------------------------------- vector field

def desiredCourse(e, chiPath, kCross, chiInf):
    return chiPath - chiInf * (2.0 / math.pi) * np.arctan(kCross * e)


def figVectorField(out, kCross=0.5, chiInfDeg=70.0, R=5.0, speed=1.0,
                   dt=0.15, steps=420):
    """The guidance field, with a Dubins vehicle recovering onto the path."""
    chiInf = math.radians(chiInfDeg)
    chiPath = 0.0                      # path along +x, at y = 0

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1),
                             gridspec_kw={'width_ratios': [1.5, 1]})
    ax = axes[0]

    # the field itself
    xs = np.linspace(0, 60, 22)
    ys = np.linspace(-14, 14, 13)
    X, Y = np.meshgrid(xs, ys)
    chi = desiredCourse(Y, chiPath, kCross, chiInf)
    ax.quiver(X, Y, np.cos(chi), np.sin(chi), color=GREY, alpha=0.42,
              width=0.0022, scale=38, headwidth=3.5)
    ax.axhline(0, color=BLUE, lw=1.6, zorder=3)
    ax.annotate('reference path', (58, 0.9), fontsize=7, color=BLUE,
                ha='right')

    # a turn-limited vehicle released off the path, tracking the field
    for y0, psi0, col, lab in ((-11.0, math.radians(20.0), ORANGE,
                               'start below, heading across'),
                               (9.0, math.radians(-5.0), GREEN,
                               'start above')):
        x, y, psi = 0.0, y0, psi0
        tx, ty = [x], [y]
        for _ in range(steps):
            chiD = desiredCourse(y, chiPath, kCross, chiInf)
            err = math.atan2(math.sin(chiD - psi), math.cos(chiD - psi))
            # bang-bang on the Dubins turn-rate limit
            rate = math.copysign(speed / R, err)
            if abs(err) < rate * dt:
                rate = err / dt
            psi += rate * dt
            x += speed * math.cos(psi) * dt
            y += speed * math.sin(psi) * dt
            tx.append(x)
            ty.append(y)
            if x > 60:
                break
        ax.plot(tx, ty, '-', color=col, lw=1.6, zorder=4, label=lab)
        ax.plot(tx[0], ty[0], 'o', color=col, ms=4.5, zorder=5)

    ax.set_xlim(0, 60)
    ax.set_ylim(-14, 14)
    ax.set_xlabel('along-track distance')
    ax.set_ylabel('cross-track error $e$')
    ax.legend(loc='lower right', framealpha=0.92)
    ax.set_title('Guidance field and recovery')

    # the course law in isolation
    ax = axes[1]
    e = np.linspace(-20, 20, 400)
    ax.plot(e, np.degrees(desiredCourse(e, chiPath, kCross, chiInf)),
            color=BLUE, lw=1.6)
    for s, lab in ((+1, r'$+\chi^{\infty}$'), (-1, r'$-\chi^{\infty}$')):
        ax.axhline(s * chiInfDeg, color=GREY, ls='--', lw=0.8)
        ax.annotate(lab, (19, s * chiInfDeg + s * 5), fontsize=7,
                    color=GREY, ha='right', va='center')
    ax.axhline(0, color='k', lw=0.6)
    ax.axvline(0, color='k', lw=0.6)
    ax.set_xlabel('cross-track error $e$')
    ax.set_ylabel(r'desired course $\chi_d$ (deg)')
    ax.set_ylim(-90, 90)
    ax.set_yticks([-90, -70, -35, 0, 35, 70, 90])
    ax.set_title(r'$k = %.1f$, $\chi^{\infty} = %.0f^{\circ}$'
                 % (kCross, chiInfDeg))
    save(fig, out, 'exp5_vector_field')


# ------------------------------------------------------------------ potentials

def figPotentials(out, d0=12.0, gain=45.0, strength=1.0, clip=0.25):
    """Khatib against the flat Reynolds rule, on one axis, plus the potential."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))

    d = np.linspace(0.05, d0 * 1.25, 800)
    eta = gain * d0 ** 3 / 4.0
    dc = np.clip(d, clip, None)
    F = np.where(d <= d0, eta * (1.0 / dc - 1.0 / d0) / dc ** 2, 0.0)
    flat = np.where(d <= d0, strength * d, 0.0)

    ax = axes[0]
    # Normalise each law by its own value at d0/2. The two laws have entirely
    # different scales, so only the SHAPE is comparable, and the shape is the
    # whole argument: Khatib rises toward contact, the flat rule falls to zero.
    Fn = F / (gain if gain else 1.0)
    flatn = flat / (strength * d0 / 2.0)
    ax.plot(d, Fn, color=BLUE, lw=1.8, label='Khatib, leaders')
    ax.plot(d, flatn, color=ORANGE, lw=1.8, ls='--',
            label='flat separation, wingmen')
    ax.axvline(d0, color=GREY, ls=':', lw=1.0)
    ax.annotate(r'$d_0$', (d0 - 0.25, 3e2), fontsize=8, color=GREY,
                ha='right')
    ax.axvline(d0 / 2, color=GREEN, ls=':', lw=1.0)
    ax.plot([d0 / 2], [1.0], 'o', color=GREEN, ms=5, zorder=5)
    ax.annotate(r'both $=1$ at $d_0/2$', (d0 / 2 + 0.35, 1.0), fontsize=7,
                color=GREEN, va='center')
    ax.annotate('vanishes at contact', (0.5, 0.055), fontsize=7,
                color=ORANGE)
    ax.set_yscale('log')
    ax.set_xlabel('separation $d$')
    ax.set_ylabel(r'force, normalised to its value at $d_0/2$')
    ax.set_ylim(3e-2, 1e3)
    ax.set_xlim(0, d0 * 1.25)
    ax.legend(loc='upper right', framealpha=0.92)
    ax.set_title('Radial profile')

    # potential, U = eta/2 (1/d - 1/d0)^2
    ax = axes[1]
    U = np.where(d <= d0, 0.5 * eta * (1.0 / dc - 1.0 / d0) ** 2, 0.0)
    ax.plot(d, U, color=BLUE, lw=1.8)
    ax.axvline(d0, color=GREY, ls=':', lw=1.0)
    ax.annotate(r'$d_0$', (d0, U.max() * 0.9), fontsize=8, color=GREY,
                ha='right')
    ax.set_yscale('log')
    ax.set_ylim(1e0, 1e7)
    ax.set_xlabel('separation $d$')
    ax.set_ylabel(r'potential $U(d)$')
    ax.set_title(r'$U = \frac{1}{2}\eta(1/d - 1/d_0)^2$')
    save(fig, out, 'exp5_potentials')


# ------------------------------------------------------------------- hierarchy

def figHierarchy(out):
    """The total order over agents, and the yield action each pairing implies."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2),
                             gridspec_kw={'width_ratios': [1.15, 1]})

    # ---- left: the ordering itself
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    def box(x, y, w, h, label, fc, ec, fs=7.5, tc='white'):
        ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=1.0,
                               zorder=3))
        ax.annotate(label, (x + w / 2, y + h / 2), fontsize=fs, color=tc,
                    ha='center', va='center', zorder=4)

    ax.annotate('tier 1: leaders', (0.2, 9.2), fontsize=8, color=GREY)
    box(0.4, 8.0, 3.9, 0.95, 'red leader', RED, RED)
    box(4.9, 8.0, 3.9, 0.95, 'gold leader', GOLD, GOLD)
    ax.annotate('tier 2: wingmen', (0.2, 6.9), fontsize=8, color=GREY)
    box(0.4, 5.1, 3.9, 0.85, 'red wingman 1', BLUE, BLUE)
    box(0.4, 4.1, 3.9, 0.85, 'red wingman 2', BLUE, BLUE)
    box(0.4, 3.1, 3.9, 0.85, r'$\cdots$', BLUE, BLUE)
    box(4.9, 5.1, 3.9, 0.85, 'gold wingman 1', '#5a9bd4', '#5a9bd4')
    box(4.9, 4.1, 3.9, 0.85, 'gold wingman 2', '#5a9bd4', '#5a9bd4')
    box(4.9, 3.1, 3.9, 0.85, r'$\cdots$', '#5a9bd4', '#5a9bd4')

    ax.annotate('', xy=(9.45, 3.1), xytext=(9.45, 8.95),
                arrowprops=dict(arrowstyle='-|>', lw=1.2, color=GREY))
    ax.annotate('decreasing precedence', (9.75, 6.0), fontsize=7.5,
                color=GREY, rotation=90, va='center', ha='center')
    ax.annotate(r'key $=(\mathrm{tier},\ \mathrm{squadron},\ \mathrm{index})$',
                (4.6, 1.9), fontsize=8, ha='center')
    ax.annotate('total, so every pair is decided', (4.6, 1.2), fontsize=7.5,
                ha='center', color=GREY)
    ax.set_title('Global ordering', fontsize=9)

    # ---- right: what yielding means
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Action required of the yielding agent', fontsize=9)

    # peer case: brake
    ax.annotate('wingman yields to wingman', (0.3, 9.3), fontsize=8,
                color=GREY)
    ax.plot([1.2], [7.9], marker=(3, 0, -90), ms=11, color=BLUE, ls='none')
    ax.plot([5.2], [7.9], marker=(3, 0, 90), ms=11, color='#5a9bd4',
            ls='none')
    ax.add_patch(FancyArrowPatch((2.9, 7.9), (2.0, 7.9),
                                 arrowstyle='-|>', mutation_scale=11,
                                 lw=1.4, color=ORANGE))
    ax.annotate('brake', (2.45, 8.35), fontsize=7.5, color=ORANGE,
                ha='center')
    ax.annotate('both can manoeuvre, and the order lets\n'
                'only one of the pair brake, so neither\n'
                'is left stationary in the other\'s path',
                (0.3, 6.05), fontsize=7, color=GREY)

    # leader case: lateral
    ax.annotate('wingman yields to leader', (0.3, 4.5), fontsize=8,
                color=GREY)
    ax.add_patch(Rectangle((0.9, 2.05), 5.4, 1.6, fc=RED, ec='none',
                           alpha=0.10, zorder=1))
    ax.add_patch(Rectangle((0.9, 2.05), 5.4, 1.6, fc='none', ec=RED, lw=0.7,
                           ls=':', zorder=2))
    ax.annotate('leader corridor', (3.6, 2.25), fontsize=6.5, color=RED,
                ha='center')
    ax.plot([6.0], [2.85], marker=(3, 0, 90), ms=13, color=RED, ls='none',
            zorder=4)
    ax.plot([2.6], [2.85], marker=(3, 0, -90), ms=10, color=BLUE, ls='none',
            zorder=4)
    ax.add_patch(FancyArrowPatch((2.6, 3.35), (2.6, 4.15),
                                 arrowstyle='-|>', mutation_scale=11,
                                 lw=1.4, color=GREEN))
    ax.annotate('clear laterally,\nat full speed', (3.0, 3.75), fontsize=7.5,
                color=GREEN, va='center')
    ax.annotate('a leader holds speed and does not yield to a\n'
                'wingman, so braking here guarantees the\n'
                'collision rather than avoiding it',
                (0.3, 0.75), fontsize=7, color=GREY)
    save(fig, out, 'exp5_hierarchy')


# ------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join('results', 'figs'))
    ap.add_argument('--n', type=int, default=6,
                    help='wingmen per squadron in the formation figure')
    args = ap.parse_args()
    print('concept figures ->', args.out)
    figFormations(args.out, n=args.n)
    figVectorField(args.out)
    figPotentials(args.out)
    figHierarchy(args.out)


if __name__ == '__main__':
    main()