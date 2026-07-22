"""Experiment 6: Red Leader and Gold Leader cross paths.

Adam's spec, verbatim-ish:

  1. LEADER INTERACTION: two leaders on straight paths that cross in an X.
     When they get into each other's protected range, SAC rules engage
     between the leaders (on top of their follow-the-path pull).

  2. WINGMEN INTERACTION: same crossing, but each leader brings their own
     squadron of fanboids (Red Squadron and Gold Squadron). Leaders only
     attract THEIR OWN squadron, but SAC rules apply to ALL fanboids across
     squadrons -- everything dodges everything while trying to path-follow.

We run each part two ways:
  (a) PLAIN SAC  -- pure push-apart, exactly as written. Spoiler: on a
      perfectly symmetric X this mirror-locks and they still collide, because
      radial repulsion shoves both leaders along the same mirror line and can
      never break the tie. (Verified: their trajectories stay exact mirror
      images to machine precision.)
  (b) RIGHT-OF-WAY -- same SAC push PLUS a small consistent veer-to-the-right
      while avoiding (capSwirl). A shared handedness breaks the symmetry and
      they pass cleanly, then settle back onto their own lines. Same reason
      boats and planes have rules of the road, not just "stay apart".
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on the path
import numpy as np
from matplotlib import pyplot as plt
from grainframe import SimOptions, make_arena
from grainframe.data_io import apply_dataset
from grainframe.paths import crossingPaths
from grainframe.multifleet import Fleet, runFleets, captainSeparation, pathDeviation

RED, GOLD = 'crimson', 'goldenrod'          # squadron colors


def buildArena(datasets, opts):
    """One arena big enough for EVERY path."""
    d0 = datasets[0]
    merged = type(d0)(                       # a Dataset-shaped stand-in covering all paths
        tag='merged',
        refPath=np.vstack([d.refPath for d in datasets]),
        wayPts=np.vstack([d.wayPts for d in datasets]),
        qi=d0.qi, qf=d0.qf,
        R=d0.R, v=d0.v, dt=d0.dt,
        refLen=max(d.refLen for d in datasets),
        bbox=(min(d.bbox[0] for d in datasets), max(d.bbox[1] for d in datasets),
              min(d.bbox[2] for d in datasets), max(d.bbox[3] for d in datasets)),
    )
    return make_arena(merged, opts)


def makeOpts(swirl):
    """One place to build the options for a run.  swirl=0 -> plain SAC."""
    opts = SimOptions()
    opts.capSwirl = swirl                    # 0 = Adam's spec as written; >0 = right-of-way
    return opts


def describe(fleets, opts, label):
    minD, idx, series = captainSeparation(fleets)
    print('\n--- %s ---' % label)
    print('Closest leader-leader approach : %.3f  (collision radius %.2f)'
          % (minD, opts.collisionRadius))
    print('Crashed?                       : %s'
          % ('YES' if minD < opts.collisionRadius else 'no'))
    for fl in fleets:
        dev = pathDeviation(fl)
        nAvoid = int(np.sum(fl.log['avoiding']))
        print('%-12s : max path deviation %.2f, final deviation %.2f, '
              'SAC engaged %d steps'
              % (fl.name, dev.max(), dev[-1], nAvoid))
    return minD, series


def plotFleets(fleets, title):
    plt.figure(figsize=(9, 8))
    for fl in fleets:
        # the path he was SUPPOSED to walk
        plt.plot(fl.data.refPath[:, 0], fl.data.refPath[:, 1], ':', color=fl.color,
                 linewidth=1.2, alpha=0.6, label='%s ref path' % fl.name)
        # the path he ACTUALLY walked
        plt.plot(fl.log['x'], fl.log['y'], '-', color=fl.color, linewidth=2.0,
                 label='%s actual' % fl.name)
        # highlight the stretch where SAC was firing
        avoiding = fl.log['avoiding'].astype(bool)
        if np.any(avoiding):
            plt.plot(fl.log['x'][avoiding], fl.log['y'][avoiding], '.',
                     color='k', markersize=3, zorder=5,
                     label='%s SAC engaged' % fl.name)
        # his squadron
        if fl.log['fanX'].size > 0:
            for b in range(fl.log['fanX'].shape[1]):
                plt.plot(fl.log['fanX'][:, b], fl.log['fanY'][:, b], '-',
                         color=fl.color, linewidth=0.5, alpha=0.35, zorder=1,
                         label='%s squadron' % fl.name.replace('Leader', 'Squadron')
                               if b == 0 else None)
    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.legend(fontsize=8)


def leaders(swirl, label):
    """Experiment 1: just the two leaders crossing."""
    opts = makeOpts(swirl)
    dataR, dataG = crossingPaths(halfLen=40.0)   # the X
    apply_dataset(opts, dataR)
    arena = buildArena([dataR, dataG], opts)

    fleets = [Fleet('Red Leader', dataR, nFan=0, color=RED),
              Fleet('Gold Leader', dataG, nFan=0, color=GOLD)]
    runFleets(fleets, arena, opts)
    minD, series = describe(fleets, opts, label)
    return fleets, opts, minD, series


def squadrons(swirl, nFan, label):
    """Experiment 2: each leader brings a squadron. SAC applies cross-squadron."""
    opts = makeOpts(swirl)
    dataR, dataG = crossingPaths(halfLen=40.0)
    apply_dataset(opts, dataR)
    arena = buildArena([dataR, dataG], opts)

    fleets = [Fleet('Red Leader', dataR, nFan=nFan, color=RED),
              Fleet('Gold Leader', dataG, nFan=nFan, color=GOLD)]
    runFleets(fleets, arena, opts)
    minD, series = describe(fleets, opts, label)
    return fleets, opts, minD, series


def run(show=False):
    # ================= Experiment 1: leader interaction =================
    flPlain, opts, minPlain, serPlain = leaders(
        swirl=0.0, label='Exp 1a: leaders, PLAIN SAC (as specced)')
    flRow, _, minRow, serRow = leaders(
        swirl=SimOptions().capSwirl,
        label='Exp 1b: leaders, SAC + right-of-way')

    plotFleets(flPlain, 'Exp 5.1a: plain SAC -- mirror-locked, still collide')
    plotFleets(flRow, 'Exp 5.1b: SAC + right-of-way -- dodge and recover')

    # ================= Experiment 2: wingmen interaction =================
    flSqP, _, minSqP, serSqP = squadrons(
        swirl=0.0, nFan=6, label='Exp 2a: squadrons, PLAIN SAC')
    flSqR, _, minSqR, serSqR = squadrons(
        swirl=SimOptions().capSwirl, nFan=6,
        label='Exp 2b: squadrons, SAC + right-of-way')

    plotFleets(flSqR, 'Exp 5.2: Red vs Gold squadrons, SAC cross-squadron')

    # ================= Leader-leader distance over time =================
    plt.figure(figsize=(9, 5))
    plt.plot(serPlain, color='0.5', label='leaders, plain SAC')
    plt.plot(serRow, color='C0', label='leaders, + right-of-way')
    plt.plot(serSqP, color='0.7', ls='--', label='squadrons, plain SAC')
    plt.plot(serSqR, color='C2', label='squadrons, + right-of-way')
    plt.axhline(opts.collisionRadius, ls='--', color='r',
                label='collision radius (%.2f)' % opts.collisionRadius)
    plt.axhline(opts.capPR, ls=':', color='0.4',
                label='leader protected range (%.1f)' % opts.capPR)
    plt.xlabel('timestep')
    plt.ylabel('Red-Gold leader distance')
    plt.title('Exp 5: how close did the leaders get?')
    plt.legend(fontsize=8)

    # ================= Swerve-and-recover =================
    plt.figure(figsize=(9, 5))
    for fl in flRow:
        plt.plot(pathDeviation(fl), color=fl.color, label='%s (leaders only)' % fl.name)
    for fl in flSqR:
        plt.plot(pathDeviation(fl), '--', color=fl.color, label='%s (with squadron)' % fl.name)
    plt.xlabel('timestep')
    plt.ylabel('distance from own reference path')
    plt.title('Exp 5: swerve and recover (right-of-way runs)')
    plt.legend(fontsize=8)

    if show:
        plt.show()
    return flPlain, flRow, flSqP, flSqR


if __name__ == '__main__':
    run(show=True)