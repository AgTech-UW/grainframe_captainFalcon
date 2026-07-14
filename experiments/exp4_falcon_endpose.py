"""Experiment 4: Captain Falcon perturbed by the crowd.

chi^2 here is CAPTAIN FALCON's end pose vs the GOAL pose qf (Adam's request),
not the fanboids vs the captain.

Two scenarios:
  1. Fanboid convoy — followers chase him AND he reacts to them
     (opts.falconSeesFanboids = True). Simulates a group of vehicles all
     trying to get to the same place.
  2. Random interferers — wanderers following nothing, just in the way.
     Falcon doing something independent while stuff happens around him:
     can he walk between the raindrops?
"""
import numpy as np
from matplotlib import pyplot as plt
from grainframe import SimOptions, load_dataset, make_arena
from grainframe.data_io import apply_dataset
from grainframe.simulate import runSimulation
from grainframe.metrics import falconFinalErrors, falconChi2
from grainframe import plotting as gp


def endPoseTrials(data, arena, opts, nDistract=0, nFan=0, nTrials=5, label=''):
    """Average Falcon's end-pose error over several random starts."""
    stats = {'miss': [], 'dth': [], 'c2p': [], 'c2a': [], 'reached': []}
    for k in range(nTrials):
        trialSeed = opts.seed + 331*k + 13*(nDistract + nFan)
        r = runSimulation(data, arena, opts, nDistract=nDistract, nFan=nFan,
                          seed=trialSeed)
        c2p, c2a = falconChi2(r, data, opts)
        _, _, miss, dth = falconFinalErrors(r, data)
        stats['miss'].append(miss)
        stats['dth'].append(abs(dth))
        stats['c2p'].append(c2p)
        stats['c2a'].append(c2a)
        stats['reached'].append(r['reachedGoal'])

    print('\n--- Falcon end-pose report: %s (%d trials) ---' % (label, nTrials))
    print('Reached goal      : %d / %d' % (sum(stats['reached']), nTrials))
    print('Position miss     : mean %.3f  std %.3f  worst %.3f'
          % (np.mean(stats['miss']), np.std(stats['miss']), np.max(stats['miss'])))
    print('Heading error     : mean %.2f deg  worst %.2f deg'
          % (np.degrees(np.mean(stats['dth'])), np.degrees(np.max(stats['dth']))))
    print('chi^2 (pos / ang) : mean %.2f / %.2f'
          % (np.mean(stats['c2p']), np.mean(stats['c2a'])))
    return stats


def plotScenario(data, run, title, fanLike):
    gp.plotBase(data, title)
    gp.addFalcon(run, 'k--', 'Captain Falcon (perturbed)')
    if fanLike:
        gp.addFanboids(run)
    else:
        gp.addDistractors(run, label='Interferers')
    gp.addPose(data.qf, 'r', 'Goal pose')
    gp.addPose((run['x'][-1], run['y'][-1], run['theta'][-1]), 'm',
               'Falcon final pose')
    plt.legend()


def run(opts=None, data=None, show=False):
    if opts is None:
        opts = SimOptions()
    if data is None:
        data = load_dataset('1')
    apply_dataset(opts, data)
    arena = make_arena(data, opts)

    # ---- Scenario 1: convoy (the switch) ----
    opts.falconSeesFanboids = True
    runFan = runSimulation(data, arena, opts, nFan=opts.nFanShowcase)
    c2p, c2a = falconChi2(runFan, data, opts)
    _, _, miss, dth = falconFinalErrors(runFan, data)
    print('(Fanboids) Falcon final: miss %.3f, heading err %.2f deg, '
          'chi^2 pos %.2f, ang %.2f'
          % (miss, np.degrees(abs(dth)), c2p, c2a))
    plotScenario(data, runFan,
                 'Falcon perturbed by his own fanboid convoy', fanLike=True)
    fanStats = endPoseTrials(data, arena, opts, nFan=opts.nFanShowcase,
                             nTrials=opts.fanTrials, label='fanboid convoy')

    # ---- Scenario 2: raindrops ----
    opts.falconSeesFanboids = False  # irrelevant here, restore default
    runInt = runSimulation(data, arena, opts, nDistract=opts.N)
    c2p, c2a = falconChi2(runInt, data, opts)
    _, _, miss, dth = falconFinalErrors(runInt, data)
    print('(Interferers) Falcon final: miss %.3f, heading err %.2f deg, '
          'chi^2 pos %.2f, ang %.2f'
          % (miss, np.degrees(abs(dth)), c2p, c2a))
    plotScenario(data, runInt,
                 'Falcon walking between the raindrops (random interferers)',
                 fanLike=False)
    intStats = endPoseTrials(data, arena, opts, nDistract=opts.N,
                             nTrials=opts.fanTrials, label='random interferers')

    # ---- Side-by-side average comparison ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    labels = ['Fanboids', 'Interferers']
    ax[0].bar(labels, [np.mean(fanStats['miss']), np.mean(intStats['miss'])],
              yerr=[np.std(fanStats['miss']), np.std(intStats['miss'])],
              color=['C0', 'C1'], capsize=4)
    ax[0].axhline(opts.collectRadius, ls='--', color='r', label='collect radius')
    ax[0].set_ylabel('final position miss')
    ax[0].legend()
    ax[1].bar(labels,
              [np.degrees(np.mean(fanStats['dth'])),
               np.degrees(np.mean(intStats['dth']))],
              yerr=[np.degrees(np.std(fanStats['dth'])),
                    np.degrees(np.std(intStats['dth']))],
              color=['C0', 'C1'], capsize=4)
    ax[1].set_ylabel('final heading error [deg]')
    fig.suptitle("Captain Falcon's average end-pose error (mean +/- std over trials)")

    if show:
        plt.show()
    return fanStats, intStats


if __name__ == '__main__':
    run(show=True)
