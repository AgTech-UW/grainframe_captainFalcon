"""Experiment 2: the Fanboids showcase (unperturbed Captain)."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from matplotlib import pyplot as plt
from grainframe import SimOptions, load_dataset, make_arena
from grainframe.data_io import apply_dataset
from grainframe.simulate import runSimulation
from grainframe.metrics import (crossTrackError, fanFinalErrors, fanChi2,
                                flockFinalReport)
from grainframe import plotting as gp


def run(opts=None, data=None, baselineXY=None, show=False):
    if opts is None:
        opts = SimOptions()
    if data is None:
        data = load_dataset('1')
    apply_dataset(opts, data)
    arena = make_arena(data, opts)

    if baselineXY is None:
        runA = runSimulation(data, arena, opts, nDistract=0)
        baselineXY = np.column_stack([runA['x'], runA['y']])

    runC = runSimulation(data, arena, opts, nFan=opts.nFanShowcase)

    errCa = crossTrackError(runC['x'], runC['y'], baselineXY)
    print('(C) captain reached goal: %s   captain-vs-baseline max %.2e  (unperturbed check)'
          % (runC['reachedGoal'], errCa.max()))

    flockFinalReport(runC, opts)
    chi2_pos, chi2_ang = fanChi2(runC, opts)
    chi2C = chi2_pos + chi2_ang
    print('(C) chi^2 = %.1f   reduced chi^2/(3n) = %.1f'
          % (chi2C, chi2C / (3 * opts.nFanShowcase)))

    gp.plotBase(data, 'Captain Falcon & the Fanboids')
    gp.addFalcon(runC)
    cols = gp.addFanboids(runC)
    gp.addPose(data.qf, 'r')
    plt.legend()

    # Final-pose error panels
    dxE, dyE, dE, dthE = fanFinalErrors(runC)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ax[0].scatter(dxE, dyE, c=cols, s=45, zorder=3)
    ax[0].add_patch(plt.Circle((0, 0), opts.sigmaPos, fill=False, ls='--',
                               color='r', label='capture tol. (sigmaPos)'))
    ax[0].axhline(0, color='0.8', lw=0.8)
    ax[0].axvline(0, color='0.8', lw=0.8)
    ax[0].set_xlabel('x error')
    ax[0].set_ylabel('y error')
    ax[0].axis('equal')
    ax[0].legend()
    ax[0].set_title('final position error vs captain')

    idx = np.arange(opts.nFanShowcase)
    ax[1].bar(idx, dE, color=cols, label='|position error|')
    ax[1].set_xlabel('fanboid #')
    ax[1].set_ylabel('distance error')
    ax2 = ax[1].twinx()
    ax2.plot(idx, np.degrees(dthE), 'k^', label='heading error')
    ax2.set_ylabel('heading error [deg]')
    ax[1].set_title('per-fanboid final errors')
    ax[1].legend(loc='upper left')
    ax2.legend(loc='upper right')
    fig.suptitle('Fanboid final-pose errors, evaluated just at the end')

    if show:
        plt.show()
    return runC


if __name__ == '__main__':
    run(show=True)
