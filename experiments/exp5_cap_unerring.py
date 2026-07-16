"""Experiment 5: Cap follows the path unerringly; everyone else avoids HIM.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add repo root so 'grainframe' is importable
import numpy as np
from matplotlib import pyplot as plt
from grainframe import SimOptions, load_dataset, make_arena
from grainframe.data_io import apply_dataset
from grainframe.simulate import runSimulation
from grainframe.metrics import count_collisions, collisions_vs_radius
from grainframe import plotting as gp

# The arbitrary radii to score (vehicle bodyRadius = radius / 2)
RADII = (0.25, 0.5, 1.0, 1.5, 2.0, 2.5)   # the vehicle sizes we'll test; small -> big


def set_cap_unerring(opts):
    # Cap on the path, blind and unswerving.
    opts.falconBoidsWeight = 0.0      # weight 0 = Cap ignores all boid forces, pure path-follower
    opts.falconSeesFanboids = False   # belt-and-suspenders: he doesn't even look at them
    # Everyone else gets Cap in their neighbor list so they can dodge him.
    opts.boidsSeeCaptain = True       # THIS is the new bit: distractors can now see + avoid Cap
    return opts


def report(coll, opts, label):        # just a tidy printer for one collision dict
    print('\n--- Collisions: %s (radius %.2f = 2 x bodyRadius %.2f) ---'
          % (label, opts.collisionRadius, opts.bodyRadius))
    print('Boid-boid          : %d distinct pairs ever collided (%d pair-timesteps)'
          % (coll['uniquePairs'], coll['pairSteps']))
    print('Boid-captain       : %d boids ever hit Cap (%d timesteps)'
          % (coll['capBoids'], coll['capSteps']))
    print('Closest approaches : boid-boid %.3f, boid-cap %.3f'
          % (coll['minSepPair'], coll['minSepCap']))


def run(opts=None, data=None, show=False):
    if opts is None:                  # no options handed in? make defaults
        opts = SimOptions()
    if data is None:                  # no dataset handed in? load #1
        data = load_dataset('1')
    opts = set_cap_unerring(opts)     # flip Cap to unerring + let everyone see him
    apply_dataset(opts, data)         # copy R, v, dt from the CSVs into opts
    arena = make_arena(data, opts)    # build the walls/box for this run

    # Scenario A: fanboids (they chase Cap AND must avoid him -- hardest case)
    runFan = runSimulation(data, arena, opts, nFan=opts.nFanShowcase)   # simulate w/ fanboids only
    collFan = count_collisions(runFan, opts)                            # score the crashes
    print('Cap reached goal (fanboids run)   : %s (miss %.3f)'
          % (runFan['reachedGoal'], runFan['goalMiss']))
    report(collFan, opts, 'fanboids avoiding Cap')

    # Scenario B: distractors (wandering, but now they see Cap and dodge)
    runDis = runSimulation(data, arena, opts, nDistract=opts.N)         # simulate w/ distractors only
    collDis = count_collisions(runDis, opts)
    print('\nCap reached goal (distractors run): %s (miss %.3f)'
          % (runDis['reachedGoal'], runDis['goalMiss']))
    report(collDis, opts, 'distractors avoiding Cap')

    # Trajectory maps
    gp.plotBase(data, 'Exp 5: Cap unerring, fanboids avoid him')   # draw ref path + set up axes
    gp.addFalcon(runFan, 'k--', 'Captain (unerring)')             # Cap's actual track
    gp.addFanboids(runFan)                                        # every fanboid's colored trail
    gp.addPose(data.qf, 'r')                                      # red arrow = goal pose
    plt.legend()

    gp.plotBase(data, 'Exp 5: Cap unerring, distractors avoid him')
    gp.addFalcon(runDis, 'k--', 'Captain (unerring)')
    gp.addDistractors(runDis, label='Distractors (avoiding Cap)')  # grey distractor trails
    gp.addPose(data.qf, 'r')
    plt.legend()

    # ---- Radius study: re-score the SAME runs at several radii ----
    print('\n--- Collisions vs vehicle radius (same runs, re-scored) ---')
    fanPairs, fanCap = collisions_vs_radius(runFan, opts, RADII)   # no re-sim! just re-count at each size
    disPairs, disCap = collisions_vs_radius(runDis, opts, RADII)
    for i, r in enumerate(RADII):                                  # print a row per radius
        print('radius %.2f : fan-run  pairs %3d  cap-hits %2d   |   '
              'dis-run  pairs %3d  cap-hits %2d'
              % (r, fanPairs[i], fanCap[i], disPairs[i], disCap[i]))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)   # two side-by-side plots, shared y-axis
    ax[0].plot(RADII, fanPairs, 'o-', color='C0', label='boid-boid pairs')   # fan run: pair crashes vs size
    ax[0].plot(RADII, fanCap, 's-', color='C1', label='boids hitting Cap')   # fan run: Cap hits vs size
    ax[0].axvline(opts.PR, ls='--', color='0.6', label='PR (avoid zone)')    # mark where "collision" = avoid range
    ax[0].set_title('fanboids run')
    ax[0].set_xlabel('collision radius')
    ax[0].set_ylabel('collisions')
    ax[0].legend()
    ax[1].plot(RADII, disPairs, 'o-', color='C0', label='boid-boid pairs')   # same two curves for distractor run
    ax[1].plot(RADII, disCap, 's-', color='C1', label='boids hitting Cap')
    ax[1].axvline(opts.PR, ls='--', color='0.6', label='PR (avoid zone)')
    ax[1].set_title('distractors run')
    ax[1].set_xlabel('collision radius')
    ax[1].legend()
    fig.suptitle('Collisions vs vehicle radius (smaller radius = closer approach allowed)')

    if show:            # only pop up windows if we asked to (lets run_all stay quiet)
        plt.show()
    return runFan, runDis   # hand the runs back in case a caller wants to poke at them


if __name__ == '__main__':   # only when you launch THIS file directly (not on import)
    run(show=True)