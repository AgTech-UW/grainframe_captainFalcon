"""Run everything, equivalent to the old monolithic captainFalcon.py

Usage:  python run_all.py [dataTag]
"""
import sys
from matplotlib import pyplot as plt
from grainframe import SimOptions, load_dataset

sys.path.insert(0, 'experiments')
from experiments import exp1_ab_distractors, exp2_fanboids, exp3_sweep_chaos
import numpy as np

dataTag = sys.argv[1] if len(sys.argv) > 1 else '1'
opts = SimOptions()
data = load_dataset(dataTag)

runA, runB = exp1_ab_distractors.run(opts, data)
print()
baselineXY = np.column_stack([runA['x'], runA['y']])
exp2_fanboids.run(opts, data, baselineXY=baselineXY)
exp3_sweep_chaos.run(opts, data)

plt.show()
