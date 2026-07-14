# grainframe: Captain Falcon & the Fanboids

Boids + Dubins simulation: a turn-limited leader (Captain Falcon) tracks a
Dubins reference path through waypoints while boid followers (fanboids) and
random interferers move around him.

## Layout

```
grainframe/            the library (back end)
  config.py            ALL tunables live here (SimOptions dataclass)
  data_io.py           load the MATLAB-exported CSV datasets
  arena.py             bounding box + safe walls
  utils.py             wrapToPi, speed clamping
  dynamics.py          boids rules, Dubins steering, Ackermann clamp, spawning
  simulate.py          runSimulation() main loop
  metrics.py           cross-track error, fanboid chi^2, Falcon end-pose chi^2
  plotting.py          reusable plot helpers (front end)
experiments/           runnable scripts, one per experiment
  exp1_ab_distractors.py    baseline vs distractors A/B
  exp2_fanboids.py          fanboid showcase + final-pose error panels
  exp3_sweep_chaos.py       n vs chi^2 sweep + 20-fanboid chaos map
  exp4_falcon_endpose.py    Falcon perturbed by the crowd; chi^2 = his end
                            pose vs the goal pose (fanboid convoy vs
                            random interferers)
run_all.py             run everything (like the old monolithic script)
captainFalcon.py       the original monolith, kept for reference
```

## Usage

```bash
python run_all.py            # dataset 1
python run_all.py 2          # dataset 2
python experiments/exp4_falcon_endpose.py   # just one experiment
```

Tune parameters by editing `grainframe/config.py`, or in code:

```python
from grainframe import SimOptions, load_dataset, make_arena
from grainframe.data_io import apply_dataset
from grainframe.simulate import runSimulation

opts = SimOptions(falconBoidsWeight=0.6, falconSeesFanboids=True)
data = load_dataset('1')
apply_dataset(opts, data)
arena = make_arena(data, opts)
run = runSimulation(data, arena, opts, nFan=10)
```

Notes vs the old script:
- `opts.Rboid`, `opts.hysteresis`, and `tMax` are now derived properties.
- CPA waypoint state moved from `opts.minDist` into a `WaypointTracker`,
  so runs can't leak state into each other.
