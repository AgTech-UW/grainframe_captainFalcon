"""Experiment 5: a parameter study of collision incidence in crossing squadrons.

WHAT THIS MEASURES
------------------
Three factors, fully crossed:

  geometry  : 'headon' (reciprocal courses, the symmetric case that pure
              repulsion provably cannot resolve) vs 'cross' (the X crossing).
  guidance  : 'waypoint' (aim at the next waypoint) vs 'vfield' (Nelson et
              al. vector field). This is the before/after the paper argues.
  nFan      : wingmen per squadron. The scaling claim.

and repeated over seeds, which vary the random spawn placement only -- the
leaders' geometry is deterministic, so seed variation isolates the effect of
initial conditions on the swarm.

Collisions are broken out BY PAIR TYPE, which is the point of the study:
"how many collisions" is much less informative than "collisions between what".
Earlier ad-hoc runs showed the dominant failure was a wingman against its OWN
captain, not the cross-squadron contacts one would expect, and that distinction
drove the formation-slot design. The breakdown is:

    leader-leader          the two captains
    fan-own-captain        wingman vs the captain it is following
    fan-foreign-captain    wingman vs the OTHER squadron's captain
    fan-fan-same           wingmen of the same squadron
    fan-fan-cross          wingmen of opposing squadrons

USAGE
-----
    python experiments/exp5_study.py sweep          # run the grid (slow)
    python experiments/exp5_study.py sweep --quick  # small grid, for a smoke test
    python experiments/exp5_study.py figures        # plots from the saved CSV
    python experiments/exp5_study.py sweep --resume # continue an interrupted run

The sweep appends to the CSV after EVERY run, so an interrupted job loses at
most one result, and --resume skips cells already present. Figures are a
separate pass over the CSV, so you can re-plot without re-simulating.
"""

import os
import sys
import csv
import json
import time
import hashlib
import argparse
import importlib.util

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, 'exp5_priority.py')
OUTDIR = os.path.join(os.path.dirname(HERE), 'results')
CSVPATH = os.path.join(OUTDIR, 'exp5_study.csv')

# Figures carry no titles: the captions do that work in the paper, and a
# title duplicated in both is a copy-editing liability.
plt.rcParams.update({
    'font.size': 9,
    'axes.grid': True,
    # A light grey grid rather than an alpha-blended black one: the EPS
    # backend cannot do transparency, so grid.alpha would be silently
    # promoted to opaque black in the figures that go in the paper.
    'grid.color': '#cccccc',
    'grid.linewidth': 0.5,
    # matplotlib's default legend frame is alpha 0.8, which alone is enough to
    # trip the PostScript transparency warning on every figure. Opaque frame.
    'legend.framealpha': 1.0,
    'figure.autolayout': True,
})

RED, GOLD = 'crimson', 'goldenrod'

# Parameters recorded with EVERY row. A results file that does not record the
# configuration that produced it is not a result, it is a number: two sweeps
# run under different tuning are structurally identical files that mean
# different things, and nothing in the data reveals the difference. This bit
# us once already -- an early partial sweep ran under a spawn band too small
# to hold its own wingmen, and only recollection prevented it being merged
# with good data. configHash is the cheap guard: any row whose hash differs
# came from a different configuration, and the figures refuse to mix them.
PROVENANCE = ['SPAWN_MODE', 'SPAWN_MIN_R', 'SPAWN_MAX_R', 'SPAWN_MIN_SEP',
              'SPAWN_ARC', 'SLOT_SHAPE', 'SLOT_SPACING', 'SLOT_MIN_R',
              'USE_FAN_BRAKE', 'BRAKE_RANGE', 'BRAKE_STOP', 'BRAKE_FLOOR',
              'USE_CORRIDOR', 'CORR_RANGE', 'CORR_HALF', 'CORR_GAIN',
              'USE_SLOTS', 'USE_GLOBAL_PRIORITY', 'SEP_ONLY_ON_ENCOUNTER',
              'PR_RED', 'PR_GOLD', 'REP_GAIN', 'HALF',
              'USE_INFEASIBILITY_STOP', 'USE_RULE_17B',
              'SLOT_PULL_CAP', 'FAN_SWIRL', 'FAN_SWIRL_HEADON_ONLY',
              'SLOT_SHAPE', 'SLOT_SWEEP', 'GEOMETRY_LANE_OFFSET']

FIELDS = ['geometry', 'guidance', 'nFan', 'seed', 'status', 'configHash',
          'leaderLeader', 'fanOwnCap', 'fanForeignCap',
          'fanFanSame', 'fanFanCross', 'totalCollisions',
          'minLeaderGap', 'minFanCapGap', 'minFanFanGap',
          'redRMS', 'goldRMS', 'redPeak', 'goldPeak',
          'stopsRed', 'stopsGold', 'steps', 'firstCollisionRelCPA',
          'runtimeSec']


# ---------------------------------------------------------------------
#  Running one configuration
# ---------------------------------------------------------------------

def captureConfig(m):
    """The tunable state of an experiment module, as a plain dict."""
    cfg = {}
    for k in PROVENANCE:
        if hasattr(m, k):
            v = getattr(m, k)
            cfg[k] = float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v
    cfg['OVERRIDES'] = dict(getattr(m, 'OVERRIDES', {}) or {})

    # Record the SimOptions fields that actually govern the physics. These
    # live in the shared config.py, so a change there would otherwise leave
    # the config hash unmoved and two incomparable sweeps would look
    # identical -- which has already caught us once.
    try:
        from grainframe.config import SimOptions
        probe = SimOptions()
        for k, v in vars(probe).items():
            cfg['opts.' + k] = cfg['OVERRIDES'].get(k, v)
    except Exception:
        pass
    return cfg


def configHash(cfg):
    """Short stable digest of a configuration, for tagging rows."""
    blob = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def loadExperiment():
    """Fresh module each call.

    Deliberately re-imported per run rather than reused: the experiment keeps
    its parameters as module-level globals, so a stale module would carry the
    previous cell's settings into the next one. Re-importing costs
    milliseconds and removes a whole class of silent cross-contamination.
    """
    spec = importlib.util.spec_from_file_location('exp5run', EXP)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def runOne(geometry, guidance, nFan, seed, overrides=None):
    """One simulation, returned as a flat dict of metrics."""
    t0 = time.time()
    m = loadExperiment()
    m.GEOMETRY = geometry
    m.GUIDANCE = guidance
    m.N_FAN = nFan
    m.SEED = seed
    if overrides:
        # Overrides land in one of TWO places and it matters which.
        #
        #   module globals  -- SLOT_SPACING, BRAKE_RANGE, SPAWN_ARC, ...
        #                      read directly by run() in the experiment file.
        #   SimOptions      -- PR, VR, SF, CF, AF, fanLeaderFactor, ...
        #                      read from `opts` inside the physics functions.
        #
        # setattr(m, 'SF', 20) creates a module attribute that NOTHING reads,
        # so the run proceeds with the original value and the sweep silently
        # reports identical results for every value tried. That is exactly
        # what happened to an earlier sensitivity sweep. Route by inspecting
        # SimOptions for the field name, and fail loudly on anything that
        # matches neither, since a silently ignored override is worse than a
        # crash.
        from grainframe.config import SimOptions
        probe = SimOptions()
        optFields = set(vars(probe).keys())

        for k, v in overrides.items():
            if k in optFields:
                # merge into the experiment's own OVERRIDES dict, which run()
                # applies to opts after the dataset is loaded
                m.OVERRIDES = dict(getattr(m, 'OVERRIDES', {}) or {})
                m.OVERRIDES[k] = v
            elif hasattr(m, k):
                setattr(m, k, v)
            else:
                raise KeyError(
                    'override %r matches neither a SimOptions field nor a '
                    'module-level parameter of the experiment. Check the '
                    'spelling: silently ignoring it would produce a sweep in '
                    'which every value gives the same answer.' % k)

    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):          # the run() print()s a report
        r = m.run(plot=False)

    row = analyse(r)
    row.update({'geometry': geometry, 'guidance': guidance,
                'nFan': nFan, 'seed': seed, 'status': 'ok',
                'configHash': configHash(captureConfig(m)),
                'runtimeSec': round(time.time() - t0, 2)})
    return row


def analyse(r):
    """Collision and tracking metrics from one finished run."""
    fx = np.array(r['log']['fanX'])
    fy = np.array(r['log']['fanY'])
    squad = np.asarray(r['squad'])
    nF = fx.shape[1]
    radius = r['opts'].collisionRadius

    capX = [np.asarray(r['xR']), np.asarray(r['xG'])]
    capY = [np.asarray(r['yR']), np.asarray(r['yG'])]
    T = min(fx.shape[0], len(capX[0]), len(capX[1]))

    dLead = r['dLeaders']
    tCPA = int(np.argmin(dLead))               # the encounter instant

    counts = dict(leaderLeader=0, fanOwnCap=0, fanForeignCap=0,
                  fanFanSame=0, fanFanCross=0)
    minLead = float(dLead.min())
    minFanCap = np.inf
    minFanFan = np.inf
    firstHit = None

    if minLead < radius:
        counts['leaderLeader'] = 1
        firstHit = int(np.argmax(dLead < radius))

    # wingman vs wingman. Count PAIRS that ever touch, not timesteps: a single
    # sustained contact is one event, and counting steps would score a slow
    # graze far worse than a fast clean hit.
    for i in range(nF):
        for j in range(i + 1, nF):
            d = np.hypot(fx[:T, i] - fx[:T, j], fy[:T, i] - fy[:T, j])
            dm = float(d.min())
            minFanFan = min(minFanFan, dm)
            if dm < radius:
                key = 'fanFanSame' if squad[i] == squad[j] else 'fanFanCross'
                counts[key] += 1
                hit = int(np.argmax(d < radius))
                firstHit = hit if firstHit is None else min(firstHit, hit)

    # wingman vs captain, split by whether it is their own
    for i in range(nF):
        for k in (0, 1):
            d = np.hypot(fx[:T, i] - capX[k][:T], fy[:T, i] - capY[k][:T])
            dm = float(d.min())
            minFanCap = min(minFanCap, dm)
            if dm < radius:
                key = 'fanOwnCap' if squad[i] == k else 'fanForeignCap'
                counts[key] += 1
                hit = int(np.argmax(d < radius))
                firstHit = hit if firstHit is None else min(firstHit, hit)

    dt = r['opts'].dt
    row = dict(counts)
    row['totalCollisions'] = sum(counts.values())
    row['minLeaderGap'] = round(minLead, 4)
    row['minFanCapGap'] = round(float(minFanCap), 4)
    row['minFanFanGap'] = round(float(minFanFan), 4) if nF > 1 else ''
    # time of the first collision RELATIVE to the leaders' closest approach.
    # Negative means it happened on the approach, positive on the way out.
    row['firstCollisionRelCPA'] = ('' if firstHit is None
                                   else round((firstHit - tCPA) * dt, 3))
    row['stopsRed'] = r['stopCount'][0]
    row['stopsGold'] = r['stopCount'][1]
    row['steps'] = len(dLead)

    from grainframe.guidance import crossTrackError
    for name, xs, ys, data in [('red', r['xR'], r['yR'], r['dataR']),
                               ('gold', r['xG'], r['yG'], r['dataG'])]:
        e = crossTrackError(np.asarray(xs), np.asarray(ys), data.refPath)
        row[name + 'RMS'] = round(float(np.sqrt(np.mean(e ** 2))), 4)
        row[name + 'Peak'] = round(float(np.abs(e).max()), 4)
    return row


# ---------------------------------------------------------------------
#  The sweep
# ---------------------------------------------------------------------

def loadDone():
    """Cells already in the CSV, so --resume can skip them."""
    if not os.path.exists(CSVPATH):
        return set()
    done = set()
    with open(CSVPATH) as f:
        for row in csv.DictReader(f):
            # Only successful cells count as done, so --resume retries
            # anything that failed rather than baking the gap in permanently.
            if row.get('status', 'ok') != 'ok':
                continue
            done.add((row['geometry'], row['guidance'],
                      int(row['nFan']), int(row['seed'])))
    return done


def migrateCSV():
    """Bring an older CSV up to the current schema before appending to it.

    Appending rows written against a NEW field list onto a file whose header
    is the OLD one silently misaligns every subsequent column -- the data
    looks fine and is wrong. So: if the header on disk does not match FIELDS,
    rewrite the file with the current header, filling absent columns. Runs
    completed under the old schema are preserved and marked ok.
    """
    if not os.path.exists(CSVPATH):
        return
    with open(CSVPATH, newline='') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        if header == FIELDS:
            return
        rows = list(reader)

    backup = CSVPATH + '.bak'
    os.replace(CSVPATH, backup)
    with open(CSVPATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            row.setdefault('status', 'ok')
            for k in FIELDS:
                row.setdefault(k, '')
            writer.writerow(row)
    print('migrated %d existing rows to the current schema (backup at %s)'
          % (len(rows), os.path.basename(backup)))


def sweep(quick=False, resume=False):
    os.makedirs(OUTDIR, exist_ok=True)
    if resume:
        migrateCSV()

    # Snapshot the configuration BEFORE running anything, write it beside the
    # CSV, and print it. If a resumed sweep would mix configurations, stop --
    # silently appending incomparable rows is the failure this exists to
    # prevent, and it is not recoverable after the fact.
    cfg = captureConfig(loadExperiment())
    chash = configHash(cfg)
    cfgPath = os.path.join(OUTDIR, 'exp5_config_%s.json' % chash)
    with open(cfgPath, 'w') as f:
        json.dump(cfg, f, indent=2, sort_keys=True, default=str)
    print('config %s -> %s' % (chash, os.path.basename(cfgPath)))
    for k in sorted(cfg):
        print('    %-24s %s' % (k, cfg[k]))
    print()

    if resume and os.path.exists(CSVPATH):
        existing = set()
        with open(CSVPATH) as f:
            for row in csv.DictReader(f):
                if row.get('configHash'):
                    existing.add(row['configHash'])
        other = existing - {chash}
        if other:
            sys.exit(
                'REFUSING TO RESUME: %s already contains rows from a '
                'different configuration (%s), and the current config is %s.\n'
                'Those rows were produced under different parameters and are '
                'not comparable to new ones. Either restore the old settings, '
                'or move the file aside and start a fresh sweep.'
                % (os.path.basename(CSVPATH), ', '.join(sorted(other)), chash))

    if quick:
        geometries = ['headon']
        guidances = ['vfield']
        fanCounts = [2, 4]
        seeds = [8, 11]
    else:
        geometries = ['headon', 'cross']
        guidances = ['waypoint', 'vfield']
        fanCounts = [1, 2, 3, 4, 6, 8, 12]
        seeds = [8, 11, 23, 42, 77, 101, 144, 200, 256, 314,
                 400, 512, 613, 700, 808, 900, 1024, 1111, 1234, 1337]

    grid = [(g, gu, n, s) for g in geometries for gu in guidances
            for n in fanCounts for s in seeds]

    done = loadDone() if resume else set()
    todo = [c for c in grid if c not in done]

    fresh = not os.path.exists(CSVPATH) or (not resume)
    mode = 'a' if (resume and os.path.exists(CSVPATH)) else 'w'

    print('grid: %d cells, %d already done, %d to run'
          % (len(grid), len(done), len(todo)))
    if not todo:
        print('nothing to do')
        return

    with open(CSVPATH, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
        if mode == 'w':
            writer.writeheader()

        t0 = time.time()
        failures = 0
        for i, (g, gu, n, s) in enumerate(todo, 1):
            # A single bad cell must not destroy an hour of completed work.
            # Failures are RECORDED rather than swallowed: a row with
            # status='failed' is visible in the CSV and excluded from the
            # figures, so a partial grid can never be silently mistaken for a
            # complete one.
            try:
                row = runOne(g, gu, n, s)
            except Exception as exc:
                failures += 1
                row = {'geometry': g, 'guidance': gu, 'nFan': n, 'seed': s,
                       'status': 'failed: %s' % type(exc).__name__,
                       'runtimeSec': 0}
                print('    !! %s %s n=%d seed=%d FAILED: %s'
                      % (g, gu, n, s, exc))
            writer.writerow(row)
            f.flush()                     # survive an interrupt
            elapsed = time.time() - t0
            rate = elapsed / i
            eta = rate * (len(todo) - i)
            hits = row.get('totalCollisions', -1)
            print('[%4d/%4d] %-7s %-8s n=%-3d seed=%-5d  hits=%-3s  '
                  '%.1fs  ETA %.0fm'
                  % (i, len(todo), g, gu, n, s,
                     '-' if hits < 0 else '%d' % hits,
                     row['runtimeSec'], eta / 60.0))

    print('\nwrote %s' % CSVPATH)
    if failures:
        print('%d of %d cells FAILED -- see status column' % (failures, len(todo)))


# ---------------------------------------------------------------------
#  Figures
# ---------------------------------------------------------------------

def readCSV():
    if not os.path.exists(CSVPATH):
        sys.exit('no results at %s -- run the sweep first' % CSVPATH)
    rows = []
    skipped = 0
    hashes = set()
    with open(CSVPATH) as f:
        for row in csv.DictReader(f):
            if row.get('status', 'ok') != 'ok':
                skipped += 1
                continue
            if row.get('configHash'):
                hashes.add(row['configHash'])
            for k in row:
                # every non-text column becomes a float; the text columns must
                # be excluded from the conversion
                if k in ('geometry', 'guidance', 'status', 'configHash'):
                    continue
                row[k] = float(row[k]) if row[k] not in ('', None) else np.nan
            rows.append(row)
    if skipped:
        print('note: %d failed runs excluded from the figures' % skipped)
    if len(hashes) > 1:
        sys.exit('REFUSING TO PLOT: %s mixes %d configurations (%s).\n'
                 'Pooling runs from different parameter settings into one '
                 'figure would be a plot of nothing in particular. Split the '
                 'file by configHash and plot each separately.'
                 % (os.path.basename(CSVPATH), len(hashes),
                    ', '.join(sorted(hashes))))
    if hashes:
        print('config %s' % hashes.pop())
    return rows


def _sel(rows, **kw):
    out = rows
    for k, v in kw.items():
        out = [r for r in out if r[k] == v]
    return out


def _meanSem(vals):
    v = np.array([x for x in vals if not np.isnan(x)], dtype=float)
    if v.size == 0:
        return np.nan, np.nan
    return v.mean(), (v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else 0.0)


def _save(fig, name):
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ('eps', 'png'):
        path = os.path.join(OUTDIR, '%s.%s' % (name, ext))
        fig.savefig(path, dpi=200, format=ext)
    print('  wrote %s.{eps,png}' % name)
    plt.close(fig)


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion.

    Used instead of the textbook normal interval because we are estimating
    probabilities near 0 and 1 from ~20 trials, exactly where the normal
    approximation misbehaves -- it produces intervals extending below zero,
    and collapses to zero width when k = 0, claiming certainty from an
    observation that is merely consistent with a small probability. Wilson
    stays inside [0, 1] and keeps sensible width at the boundaries.
    """
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    d = 1.0 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


def logisticFit(x, y, iters=100):
    """Logistic regression by IRLS, on raw 0/1 outcomes.

    Fitted on the individual runs rather than on the per-n proportions,
    because fitting a curve to proportions throws away the differing number of
    trials behind each point and cannot cope with proportions of exactly 0 or
    1 (the logit is infinite there) -- which is precisely what small squadrons
    produce. Returns (intercept, slope) or None if it fails to converge.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    b = np.zeros(2)
    for _ in range(iters):
        eta = X @ b
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        W = np.clip(mu * (1 - mu), 1e-9, None)
        z = eta + (y - mu) / W
        try:
            bNew = np.linalg.solve((X * W[:, None]).T @ X,
                                   (X * W[:, None]).T @ z)
        except np.linalg.LinAlgError:
            return None
        if np.max(np.abs(bNew - b)) < 1e-9:
            return bNew
        b = bNew
    return b


def figCollisionProbability(rows):
    """P(at least one collision) against squadron size.

    This replaces a mean-collisions-per-run plot. A mean over integer counts
    reports quantities like 0.35 collisions, which do not correspond to
    anything observable, and it conflates "many runs with one collision" with
    "one run with many". The probability of a run being clean is directly
    interpretable and is the quantity the question "how many wingmen before
    something hits" is actually asking about.

    The fitted curve gives N50: the squadron size at which a collision becomes
    more likely than not.
    """
    geoms = sorted(set(r['geometry'] for r in rows))
    fig, axes = plt.subplots(1, len(geoms), figsize=(4.4 * len(geoms), 3.6),
                             sharey=True, squeeze=False)
    print('\n  collision probability, N50 (squadron size at P = 0.5):')
    for ax, geom in zip(axes[0], geoms):
        for guid, colour, marker in (('waypoint', 'tab:gray', 's'),
                                     ('vfield', 'tab:blue', 'o')):
            sub = _sel(rows, geometry=geom, guidance=guid)
            if not sub:
                continue
            ns = sorted(set(r['nFan'] for r in sub))
            ps, los, his = [], [], []
            for n in ns:
                cell = _sel(sub, nFan=n)
                k = sum(1 for r in cell if r['totalCollisions'] > 0)
                p, lo, hi = wilson(k, len(cell))
                ps.append(p)
                los.append(p - lo)
                his.append(hi - p)
            ax.errorbar(ns, ps, yerr=[los, his], marker=marker, ms=4.5,
                        lw=0, elinewidth=1.1, capsize=2.5, color=colour,
                        label=guid)

            xs = [r['nFan'] for r in sub]
            ys = [1.0 if r['totalCollisions'] > 0 else 0.0 for r in sub]
            b = logisticFit(xs, ys)
            if b is not None and abs(b[1]) > 1e-9:
                grid = np.linspace(min(xs), max(xs), 200)
                ax.plot(grid, 1 / (1 + np.exp(-(b[0] + b[1] * grid))),
                        '-', lw=1.2, color=colour, alpha=1.0)
                n50 = -b[0] / b[1]
                inRange = min(xs) <= n50 <= max(xs)
                print('    %-7s %-8s N50 = %.1f%s'
                      % (geom, guid, n50,
                         '' if inRange else '  (extrapolated beyond the grid)'))
        ax.axhline(0.5, color='k', ls=':', lw=0.8)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('wingmen per squadron')
        ax.set_title(geom, fontsize=9)
    axes[0][0].set_ylabel('P(at least one collision)')
    axes[0][-1].legend(fontsize=7, loc='lower right')
    _save(fig, 'exp5_collision_probability')


def figCountDistribution(rows):
    """How many collisions per run, as integer counts.

    Stacked bars rather than a mean: collisions come in whole numbers, and the
    shape of the distribution matters. Twenty runs each with one collision and
    one run with twenty are the same mean and completely different failures.
    """
    geoms = sorted(set(r['geometry'] for r in rows))
    fig, axes = plt.subplots(1, len(geoms), figsize=(4.4 * len(geoms), 3.6),
                             sharey=True, squeeze=False)
    bands = [(0, 0, 'none', '#d9d9d9'),
             (1, 1, 'one', '#9ecae1'),
             (2, 3, 'two or three', '#4292c6'),
             (4, 10 ** 9, 'four or more', '#08519c')]
    for ax, geom in zip(axes[0], geoms):
        sub = _sel(rows, geometry=geom, guidance='vfield')
        ns = sorted(set(r['nFan'] for r in sub))
        bottom = np.zeros(len(ns))
        for lo, hi, label, colour in bands:
            frac = []
            for n in ns:
                cell = _sel(sub, nFan=n)
                k = sum(1 for r in cell
                        if lo <= r['totalCollisions'] <= hi)
                frac.append(k / len(cell) if cell else 0.0)
            frac = np.array(frac)
            ax.bar(range(len(ns)), frac, 0.72, bottom=bottom,
                   label=label, color=colour, edgecolor='white', linewidth=0.5)
            bottom += frac
        ax.set_xticks(range(len(ns)))
        ax.set_xticklabels([str(int(n)) for n in ns])
        ax.set_xlabel('wingmen per squadron')
        ax.set_title(geom, fontsize=9)
        ax.grid(False)
    axes[0][0].set_ylabel('fraction of runs')
    axes[0][-1].legend(fontsize=6.5, loc='center left',
                       bbox_to_anchor=(1.01, 0.5), title='collisions')
    _save(fig, 'exp5_count_distribution')


def figTypeComposition(rows):
    """What is colliding with what, as total counts across all runs.

    Absolute integer totals, not per-run averages. The composition is the
    finding: earlier ad-hoc runs showed the dominant failure was a wingman
    against its OWN captain rather than the cross-squadron contacts the
    encounter geometry would suggest, and that is what motivated the
    formation-slot design.
    """
    types = [('fanOwnCap', 'wingman / own leader', '#d62728'),
             ('fanForeignCap', 'wingman / opposing leader', '#ff7f0e'),
             ('fanFanSame', 'wingman / squadmate', '#1f77b4'),
             ('fanFanCross', 'wingman / opposing wingman', '#2ca02c'),
             ('leaderLeader', 'leader / leader', '#000000')]
    geoms = sorted(set(r['geometry'] for r in rows))
    fig, axes = plt.subplots(1, len(geoms), figsize=(4.4 * len(geoms), 3.6),
                             sharey=True, squeeze=False)
    for ax, geom in zip(axes[0], geoms):
        sub = _sel(rows, geometry=geom, guidance='vfield')
        ns = sorted(set(r['nFan'] for r in sub))
        bottom = np.zeros(len(ns))
        for key, label, colour in types:
            tot = np.array([sum(r[key] for r in _sel(sub, nFan=n))
                            for n in ns], dtype=float)
            ax.bar(range(len(ns)), tot, 0.72, bottom=bottom, label=label,
                   color=colour, edgecolor='white', linewidth=0.5)
            bottom += tot
        ax.set_xticks(range(len(ns)))
        ax.set_xticklabels([str(int(n)) for n in ns])
        ax.set_xlabel('wingmen per squadron')
        ax.set_title(geom, fontsize=9)
        ax.grid(False)
    axes[0][0].set_ylabel('total collisions (all runs)')
    axes[0][-1].legend(fontsize=6.5, loc='center left',
                       bbox_to_anchor=(1.01, 0.5))
    _save(fig, 'exp5_type_composition')


def figGuidanceComparison(rows):
    """Waypoint vs vector field: tracking error, and clean-run fraction."""
    geoms = sorted(set(r['geometry'] for r in rows))
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))

    labels, wp, vf, wpE, vfE = [], [], [], [], []
    for geom in geoms:
        for guid, store, err in (('waypoint', wp, wpE), ('vfield', vf, vfE)):
            sub = _sel(rows, geometry=geom, guidance=guid)
            mu, se = _meanSem([r['redRMS'] for r in sub] +
                              [r['goldRMS'] for r in sub])
            store.append(mu)
            err.append(se)
        labels.append(geom)
    x = np.arange(len(labels))
    axes[0].bar(x - 0.18, wp, 0.36, yerr=wpE, capsize=3,
                label='waypoint', color='white', edgecolor='k', hatch='///')
    axes[0].bar(x + 0.18, vf, 0.36, yerr=vfE, capsize=3,
                label='vector field', color='white', edgecolor='tab:blue')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel('cross-track RMS')
    axes[0].legend(fontsize=7)

    # right: fraction of runs with NO collision at all. A proportion, which
    # is what the underlying data supports -- not a mean of integer counts.
    for guid, colour, marker in (('waypoint', 'tab:gray', 's'),
                                 ('vfield', 'tab:blue', 'o')):
        sub = _sel(rows, guidance=guid)
        ns = sorted(set(r['nFan'] for r in sub))
        ps, los, his = [], [], []
        for n in ns:
            cell = _sel(sub, nFan=n)
            k = sum(1 for r in cell if r['totalCollisions'] == 0)
            p, lo, hi = wilson(k, len(cell))
            ps.append(p)
            los.append(p - lo)
            his.append(hi - p)
        axes[1].errorbar(ns, ps, yerr=[los, his], marker=marker, ms=4,
                         lw=1.0, capsize=2.5, color=colour, label=guid)
    axes[1].set_xlabel('wingmen per squadron')
    axes[1].set_ylabel('fraction of collision-free runs')
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(fontsize=7)
    _save(fig, 'exp5_guidance_comparison')


def figClearance(rows):
    """Closest approach distributions against the collision radius."""
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    panels = [('minLeaderGap', 'leader-leader'),
              ('minFanCapGap', 'wingman-leader'),
              ('minFanFanGap', 'wingman-wingman')]
    for ax, (key, label) in zip(axes, panels):
        for guid, colour in (('waypoint', 'tab:gray'), ('vfield', 'tab:blue')):
            vals = [r[key] for r in _sel(rows, guidance=guid)
                    if not np.isnan(r[key])]
            if vals:
                # step outlines, not filled bars: the EPS backend has no
                # transparency, so overlapping alpha-blended histograms would
                # render opaque in the paper and hide whichever series is
                # drawn second. Outlines read correctly in both formats.
                ax.hist(vals, bins=28, histtype='step', lw=1.4,
                        label=guid, color=colour)
        ax.axvline(1.0, color='red', ls='--', lw=1.2)
        ax.set_xlabel('closest approach (%s)' % label)
    axes[0].set_ylabel('runs')
    axes[0].legend(fontsize=7)
    _save(fig, 'exp5_clearance')


def figTiming(rows):
    """When do collisions happen, relative to the leaders' closest approach?"""
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for geom, colour in (('headon', 'tab:purple'), ('cross', 'tab:olive')):
        vals = [r['firstCollisionRelCPA'] for r in _sel(rows, geometry=geom)
                if not np.isnan(r['firstCollisionRelCPA'])]
        if vals:
            ax.hist(vals, bins=30, histtype='step', lw=1.4,
                    label=geom, color=colour)
    ax.axvline(0.0, color='red', ls='--', lw=1.2)
    ax.set_xlabel('time of first collision relative to leader CPA (s)')
    ax.set_ylabel('runs')
    ax.legend(fontsize=7)
    _save(fig, 'exp5_collision_timing')


def figPairScaling(rows):
    """Is collision risk driven by agent count or by PAIR count?

    Plotted against the number of ordered pairs on the field, C(2n+2, 2),
    rather than against n. The distinction is mechanistic: if each additional
    wingman carries a fixed independent risk, incidence is linear in AGENTS;
    if the risk lives in the interactions, it is linear in PAIRS, which grows
    quadratically. Section 3.1 already reports a quadratic trend in pose
    error, so a pair-driven result here would make the same statement about
    collisions and the two findings would reinforce each other.

    Total counts across all runs, on linear axes. Nothing here is averaged.
    """
    sub = _sel(rows, guidance='vfield')
    ns = sorted(set(r['nFan'] for r in sub))
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    for geom, colour, marker in (('headon', 'tab:purple', 'o'),
                                 ('cross', 'tab:olive', 's')):
        g = _sel(sub, geometry=geom)
        if not g:
            continue
        pairs, tot = [], []
        for n in ns:
            cell = _sel(g, nFan=n)
            if not cell:
                continue
            nAgents = 2 * n + 2                 # both squadrons plus captains
            pairs.append(nAgents * (nAgents - 1) / 2.0)
            tot.append(sum(r['totalCollisions'] for r in cell))
        ax.plot(pairs, tot, marker=marker, ms=4.5, lw=1.2, color=colour,
                label=geom)
        if len(pairs) >= 2:
            p = np.polyfit(pairs, tot, 1)
            xs = np.array([min(pairs), max(pairs)])
            # no alpha: the PostScript backend renders it opaque anyway and
            # warns, so a lighter dash weight does the job instead
            ax.plot(xs, np.polyval(p, xs), '--', lw=0.8, color=colour)
            resid = np.array(tot) - np.polyval(p, pairs)
            ss = 1 - resid.var() / (np.var(tot) if np.var(tot) > 0 else 1)
            print('    %-7s collisions vs pairs: slope %.4f, R2 %.3f'
                  % (geom, p[0], ss))
    ax.set_xlabel('number of agent pairs on the field')
    ax.set_ylabel('total collisions (all runs)')
    ax.legend(fontsize=7)
    _save(fig, 'exp5_pair_scaling')


def summaryTable(rows):
    """A LaTeX table body, ready to paste into the paper."""
    path = os.path.join(OUTDIR, 'exp5_summary.tex')
    with open(path, 'w') as f:
        f.write('%% auto-generated by exp5_study.py -- do not hand-edit\n')
        f.write('\\begin{tabular}{llcccc}\n\\hline\n')
        f.write('Geometry & Guidance & Collisions & Leader gap & '
                'Red RMS & Gold RMS \\\\\n\\hline\n')
        for geom in sorted(set(r['geometry'] for r in rows)):
            for guid in sorted(set(r['guidance'] for r in rows)):
                sub = _sel(rows, geometry=geom, guidance=guid)
                if not sub:
                    continue
                c, _ = _meanSem([r['totalCollisions'] for r in sub])
                g, _ = _meanSem([r['minLeaderGap'] for r in sub])
                rr, _ = _meanSem([r['redRMS'] for r in sub])
                gg, _ = _meanSem([r['goldRMS'] for r in sub])
                f.write('%s & %s & %.2f & %.2f & %.2f & %.2f \\\\\n'
                        % (geom, guid, c, g, rr, gg))
        f.write('\\hline\n\\end{tabular}\n')
    print('  wrote exp5_summary.tex')


def figures():
    rows = readCSV()
    print('read %d runs' % len(rows))
    figCollisionProbability(rows)
    figCountDistribution(rows)
    figTypeComposition(rows)
    figGuidanceComparison(rows)
    figClearance(rows)
    figTiming(rows)
    figPairScaling(rows)
    summaryTable(rows)
    print('\nall figures in %s' % OUTDIR)


# ---------------------------------------------------------------------

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['sweep', 'figures'])
    ap.add_argument('--quick', action='store_true',
                    help='tiny grid, to check the pipeline before committing '
                         'to the full run')
    ap.add_argument('--resume', action='store_true',
                    help='skip cells already present in the CSV')
    args = ap.parse_args()

    if args.mode == 'sweep':
        matplotlib.use('Agg')          # no windows during a long batch
        sweep(quick=args.quick, resume=args.resume)
    else:
        figures()