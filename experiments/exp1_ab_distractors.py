"""Experiment 1: the original A/B — baseline vs random distractors."""
import numpy as np
from matplotlib import pyplot as plt
from grainframe import SimOptions, load_dataset, make_arena
from grainframe.data_io import apply_dataset
from grainframe.simulate import runSimulation
from grainframe.metrics import crossTrackError
from grainframe import plotting as gp


def run(opts=None, data=None, show=False):
    if opts is None:
        opts = SimOptions()
    if data is None:
        data = load_dataset('1')
    apply_dataset(opts, data)
    arena = make_arena(data, opts)

    runA = runSimulation(data, arena, opts, nDistract=0)
    runB = runSimulation(data, arena, opts, nDistract=opts.N)

    baselineXY = np.column_stack([runA['x'], runA['y']])
    errB  = crossTrackError(runB['x'], runB['y'], baselineXY)
    errAr = crossTrackError(runA['x'], runA['y'], data.refPath[:, :2])

    print('(A) reached goal: %s (miss %.3f)   baseline-vs-reference: mean %.4f  max %.4f'
          % (runA['reachedGoal'], runA['goalMiss'], errAr.mean(), errAr.max()))
    print('(B) reached goal: %s (miss %.3f)   perturbed-vs-baseline: mean %.4f  RMS %.4f  max %.4f'
          % (runB['reachedGoal'], runB['goalMiss'],
             errB.mean(), np.sqrt(np.mean(errB**2)), errB.max()))

    gp.plotBase(data, 'Captain Falcon vs the (turn-limited) distractors')
    gp.addFalcon(runA, 'k--', '(A) Baseline')
    gp.addFalcon(runB, 'g-', '(B) Perturbed')
    gp.addWaypoints(data)
    gp.addPose(data.qi, 'g')
    gp.addPose(data.qf, 'r')
    gp.addDistractors(runB)
    plt.legend()

    if show:
        plt.show()
    return runA, runB


if __name__ == '__main__':
    run(show=True)
