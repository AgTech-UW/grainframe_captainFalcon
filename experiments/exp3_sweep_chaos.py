"""Experiment 3: n vs chi^2 sweep (pos/heading split) + 20-fanboid chaos map."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from matplotlib import pyplot as plt
from grainframe import SimOptions, load_dataset, make_arena
from grainframe.data_io import apply_dataset
from grainframe.simulate import runSimulation
from grainframe.metrics import fanChi2
from grainframe import plotting as gp


def run(opts=None, data=None, show=False):
    if opts is None:
        opts = SimOptions()
    if data is None:
        data = load_dataset('1')
    apply_dataset(opts, data)
    arena = make_arena(data, opts)

    print('\n--- n vs chi^2 sweep (%d random clumped starts per n) ---' % opts.fanTrials)
    sweepPos = {n: [] for n in opts.fanSweepN}
    sweepAng = {n: [] for n in opts.fanSweepN}

    for n in opts.fanSweepN:
        for k in range(opts.fanTrials):
            trialSeed = opts.seed + 101*k + 7*n
            r = runSimulation(data, arena, opts, nFan=n, seed=trialSeed)
            c2p, c2a = fanChi2(r, opts)
            sweepPos[n].append(c2p)
            sweepAng[n].append(c2a)
        print('n = %2d : chi^2(pos) mean %8.1f   chi^2(ang) mean %8.1f'
              % (n, np.mean(sweepPos[n]), np.mean(sweepAng[n])))

    nArr = np.array(opts.fanSweepN, dtype=float)
    chiPosAvg = np.array([np.mean(sweepPos[n]) for n in opts.fanSweepN])
    chiPosStd = np.array([np.std(sweepPos[n])  for n in opts.fanSweepN])
    chiAngAvg = np.array([np.mean(sweepAng[n]) for n in opts.fanSweepN])
    chiAngStd = np.array([np.std(sweepAng[n])  for n in opts.fanSweepN])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(nArr, chiPosAvg, yerr=chiPosStd, fmt='o-', color='C0',
                capsize=3, label='Pos chi^2')
    ax.errorbar(nArr, chiAngAvg, yerr=chiAngStd, fmt='s-', color='C1',
                capsize=3, label='Heading chi^2')
    ax.set_xlabel('n fanboids')
    ax.set_ylabel('chi^2')
    ax.set_title('N vs chi^2 (Position and Heading)')
    ax.legend()

    # ---- 20-fanboid chaos visualization ----
    print('\n--- Generating 20-Fanboid Trajectory Map ---')
    runChaos = runSimulation(data, arena, opts, nFan=20, seed=123)

    gp.plotBase(data, '20 Fanboids: Structural Bottleneck & Swarm Chaos')
    gp.addFalcon(runChaos)
    gp.addFanboids(runChaos)
    gp.addPose(data.qf, 'r')
    plt.legend()

    if show:
        plt.show()
    return sweepPos, sweepAng, runChaos


if __name__ == '__main__':
    run(show=True)
