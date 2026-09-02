"""Experiment 5: parameter sensitivity and squadron-size capacity.

TWO QUESTIONS, TWO MODES
------------------------
  sensitivity : how does collision incidence vary as one parameter is swept?
                Reports the MAP, not a winner. This is the honest output when
                the paper's argument is about mechanisms rather than about a
                particular operating point.

  capacity    : what is the largest squadron the current configuration can
                carry while keeping collision probability under a threshold?
                Reports N_max with a confidence interval.

WHY THE SEED SPLIT MATTERS
--------------------------
Selecting parameters because they scored well, then reporting that score, is
circular: the winner of a search over noisy estimates is partly winning on
luck, and the reported number will not reproduce. So the seeds are split.

  TUNE_SEEDS    : used to choose values. Look at these all you like.
  HOLDOUT_SEEDS : never used for selection. Every number that goes in the
                  paper comes from these.

That lets you write one sentence a reviewer will care about: "parameters were
selected on a disjoint set of initial conditions." The cost is wider error
bars on the reported figure, which is cheap for what it buys.

WHAT IS AND IS NOT WORTH SWEEPING
---------------------------------
Some values are CONSTRAINTS derived from the geometry, not free parameters,
and sweeping them is a category error:

    SLOT_SPACING  > opts.PR   or the formation is self-repelling
    SPAWN_MIN_SEP > opts.PR   or wingmen spawn already in violation
    SLOT_SHAPE    = 'vee'     keeps wingmen out of the captain's corridor
                              (measured: 'column' puts 3 of 6 inside it)

Those are justified by argument and fixed. Genuinely free parameters, needing
selection, are the brake geometry, the corridor geometry, and REP_GAIN.

USAGE
-----
    python experiments/exp5_tune.py sensitivity --dry-run   # show the grid
    python experiments/exp5_tune.py sensitivity             # run it
    python experiments/exp5_tune.py capacity
    python experiments/exp5_tune.py figures
"""

import os
import sys
import csv
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp5_study import (runOne, wilson, logisticFit, captureConfig,
                        configHash, loadExperiment, OUTDIR, _save)

TUNECSV = os.path.join(OUTDIR, 'exp5_tune.csv')
CAPCSV = os.path.join(OUTDIR, 'exp5_capacity.csv')

plt.rcParams.update({'font.size': 9, 'axes.grid': True,
                     'grid.color': '#cccccc', 'grid.linewidth': 0.5,
                     'legend.framealpha': 1.0, 'figure.autolayout': True})

# --- seeds. Disjoint by construction; do not let these overlap. ---
#
# Seed count is usually the binding constraint, not the parameter grid. With
# n seeds, the tightest 95% upper bound achievable even at ZERO observed
# collisions is roughly:
#
#     10 seeds -> 0.28      30 seeds -> 0.11      100 seeds -> 0.04
#     20 seeds -> 0.16      50 seeds -> 0.07
#
# So "we observed no collisions" over 10 runs supports only "probability is
# under 28%", which is a much weaker claim than it sounds. If there is spare
# compute, more seeds buys more than more parameter values.
def makeSeeds(nTune, nHoldout):
    """Deterministic disjoint seed sets. Tune seeds are even, holdout odd,
    so the two can never collide however the counts are chosen."""
    tune = [2 * i + 2 for i in range(nTune)]
    hold = [2 * i + 1001 for i in range(nHoldout)]
    assert not (set(tune) & set(hold))
    return tune, hold


TUNE_SEEDS, HOLDOUT_SEEDS = makeSeeds(10, 10)

# --- what to sweep, and why each range. Free parameters only. ---
SWEEPS = {
    # Put first because the timing data points here. Collisions cluster AFTER
    # the captains pass, not at the encounter, which means the squadrons
    # survive the pass and then hit each other while re-forming. This gate is
    # what governs re-forming: cohesion and alignment are suppressed while any
    # foreign agent is inside ENCOUNTER_RANGE, then restored the moment the
    # last one leaves. Default None means the gate range equals opts.VR (8.0)
    # -- flocking resumes at exactly the distance wingmen can still see each
    # other, which is the worst available choice. Larger values keep the gate
    # shut until the squadrons are genuinely clear of one another.
    'ENCOUNTER_RANGE': dict(
        values=[8.0, 12.0, 16.0, 20.0, 26.0, 32.0],
        why='Distance at which cohesion and alignment switch back on after a '
            'pass. Currently None, i.e. opts.VR = 8.0. The measured failure is '
            'post-pass re-formation, so this is the primary suspect.'),
    # --- the pull-vs-separation ratio. This is where the evidence points. ---
    # Flat Reynolds separation is F = SF*d, which SHRINKS to zero at contact.
    # The slot pull is fanLeaderFactor*(distance to slot) and does not shrink.
    # Inside d = k*L/SF the pull wins and keeps winning, so a wingman that
    # drifts inside is drawn all the way in. These four govern that balance
    # and none of them were in the earlier 2820-run sweep, which is why it
    # found nothing.
    'SLOT_PULL_CAP': dict(
        values=[1.0, 2.0, 4.0, 8.0, 1e9],
        labels=['1', '2', '4', '8', 'none'],
        why='Ceiling on the slot pull magnitude. Directly tests whether the '
            'ratio is what traps wingmen against their own captain. 1e9 is '
            'effectively uncapped, i.e. current behaviour.'),
    'BRAKE_RANGE': dict(
        values=[2.0, 3.0, 4.0, 5.0, 6.0, 8.0],
        why='Two wingmen closing at maxSpeed 2.0 each have a 4.0 closing '
            'rate, so the current 3.0 is 0.75 s of reaction. Widening cannot '
            'deadlock, because the total ordering means only one of any pair '
            'brakes.'),
    'BRAKE_STOP': dict(
        values=[1.2, 1.4, 1.8, 2.2, 2.6],
        why='Must stay above collisionRadius (1.0) to be a margin at all. '
            'Too large and wingmen stop for traffic that was never a threat.'),
    'CORR_HALF': dict(
        values=[3.0, 4.5, 6.0, 7.5, 9.0],
        why='Lane half-width. A narrow lane over a long look-ahead pushes '
            'weakly and early; wider clears more decisively but disturbs more '
            'wingmen.'),
    'CORR_RANGE': dict(
        values=[6.0, 9.0, 12.0, 16.0, 20.0],
        why='How far up the captain track to look. Interacts with SLOT_MIN_R '
            '(8.0): slots sit astern, so a long corridor mostly catches '
            'wingmen that have overshot.'),
    'CORR_GAIN': dict(
        values=[20.0, 40.0, 60.0, 90.0, 130.0],
        why='Lateral push strength.'),
    'REP_GAIN': dict(
        values=[25.0, 45.0, 70.0, 90.0, 130.0],
        why='Captain-captain Khatib strength. This, not PR, moves the point '
            'at which a captain actually reacts: at PR 12 the force is 0.66 '
            'against a guidance pull of ~10, reaching parity only at d = 8.4.'),
    'SPAWN_MIN_SEP': dict(
        values=[6.0, 7.0, 8.0, 10.0, 12.0],
        why='Spawn separation. Must exceed opts.PR (5.0). Larger values need '
            'a wider band, which the spawn auto-expands to provide.'),
    'fanLeaderFactor': dict(
        values=[0.25, 0.5, 1.0, 2.0],
        why='Strength of the slot pull. Lower means separation wins at a '
            'shorter distance, but also a looser formation.'),
    'SF': dict(
        values=[5.0, 10.0, 20.0, 40.0],
        why='Separation strength. Raising it moves the balance point inward, '
            'but cannot fix the profile: the force still vanishes at contact.'),
    'SPAWN_ARC': dict(
        values=[np.deg2rad(a) for a in (40, 60, 80, 100, 120)],
        labels=['40', '60', '80', '100', '120'],
        why='Half-width of the rear spawn sector, in degrees. Wider spreads '
            'wingmen over more area at the same radius.'),
}

FIELDS = ['mode', 'param', 'value', 'label', 'split', 'nFan', 'seed',
          'geometry', 'guidance', 'configHash', 'status',
          'totalCollisions', 'fanOwnCap', 'fanForeignCap',
          'fanFanSame', 'fanFanCross', 'leaderLeader',
          'minFanFanGap', 'minFanCapGap', 'minLeaderGap',
          'redRMS', 'goldRMS', 'redArrived', 'goldArrived',
          # WHEN the first collision happened, in seconds relative to the
          # leaders' closest approach. Negative = on the approach, positive =
          # on the way out, near zero = at the encounter itself. This is what
          # separates "the encounter is too hard" from "the squadrons arrive
          # badly formed": the collision TYPE tells you who hit whom, but only
          # the timing tells you whether the encounter caused it.
          'firstCollisionRelCPA', 'steps', 'runtimeSec']


def _writer(path, mode):
    f = open(path, mode, newline='')
    w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
    if mode == 'w':
        w.writeheader()
    return f, w


def _run(param, value, label, split, nFan, seed, geometry, guidance,
         overrides=None):
    ov = dict(overrides) if overrides else {}
    if param:
        ov[param] = value
    try:
        row = runOne(geometry, guidance, nFan, seed, overrides=ov)
    except Exception as exc:
        row = {'status': 'failed: %s' % type(exc).__name__, 'runtimeSec': 0}
    row.update({'param': param or '', 'value': value if param else '',
                'label': label, 'split': split, 'nFan': nFan, 'seed': seed,
                'geometry': geometry, 'guidance': guidance})
    return row


# ---------------------------------------------------------------------
#  Sensitivity: one parameter at a time
# ---------------------------------------------------------------------

def _doneKeys(path, keyFn):
    """Cells already recorded, so a long run can be resumed after a crash."""
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get('status', 'ok') != 'ok':
                continue          # retry anything that failed
            done.add(keyFn(r))
    return done


def sensitivity(nFan, geometry, guidance, params, dryRun=False, resume=False):
    os.makedirs(OUTDIR, exist_ok=True)
    jobs = []
    for name in params:
        spec = SWEEPS[name]
        labels = spec.get('labels') or ['%g' % v for v in spec['values']]
        for v, lab in zip(spec['values'], labels):
            for split, seeds in (('tune', TUNE_SEEDS),
                                 ('holdout', HOLDOUT_SEEDS)):
                for s in seeds:
                    jobs.append((name, v, lab, split, s))

    print('sensitivity: %d parameters, nFan=%d, %s/%s'
          % (len(params), nFan, geometry, guidance))
    for name in params:
        spec = SWEEPS[name]
        labels = spec.get('labels') or ['%g' % v for v in spec['values']]
        print('  %-14s %s' % (name, ', '.join(labels)))
        print('      %s' % spec['why'])
    print('\n  %d runs total (%d values x %d/%d seeds x 2 splits)'
          % (len(jobs), sum(len(SWEEPS[p]['values']) for p in params),
             len(TUNE_SEEDS), len(HOLDOUT_SEEDS)))

    done = set()
    if resume:
        done = _doneKeys(TUNECSV, lambda r: (r['param'], r['label'],
                                             r['split'], int(r['seed'])))
        before = len(jobs)
        jobs = [j for j in jobs
                if (j[0], j[2], j[3], j[4]) not in done]
        print('  %d already recorded, %d remaining' % (before - len(jobs),
                                                       len(jobs)))
    print('  at ~7 s per run that is about %.0f minutes (%.1f hours)'
          % (len(jobs) * 7.0 / 60.0, len(jobs) * 7.0 / 3600.0))

    if dryRun:
        print('\n--dry-run: nothing executed')
        return
    if not jobs:
        print('nothing to do')
        return

    f, w = _writer(TUNECSV, 'a' if (resume and os.path.exists(TUNECSV)) else 'w')
    t0 = time.time()
    with f:
        for i, (name, v, lab, split, s) in enumerate(jobs, 1):
            row = _run(name, v, lab, split, nFan, s, geometry, guidance)
            row['mode'] = 'sensitivity'
            w.writerow(row)
            f.flush()
            el = time.time() - t0
            print('[%4d/%4d] %-14s %-6s %-8s seed=%-5d hits=%-3s ETA %.0fm'
                  % (i, len(jobs), name, lab, split, s,
                     row.get('totalCollisions', '-'),
                     (el / i) * (len(jobs) - i) / 60.0))
    print('\nwrote %s' % TUNECSV)


# ---------------------------------------------------------------------
#  Capacity: the largest squadron that stays under a collision threshold
# ---------------------------------------------------------------------

def capacity(geometry, guidance, sizes, threshold, dryRun=False,
             resume=False):
    """Largest nFan whose collision probability stays at or below threshold.

    Run on HOLDOUT seeds only. The reported bound uses the UPPER Wilson limit,
    not the point estimate: claiming a squadron size is safe because 9 of 10
    runs were clean overstates what ten trials can support. Requiring the
    upper limit to clear the threshold is the conservative reading, and it is
    the one that survives a reviewer asking how many runs you did.
    """
    os.makedirs(OUTDIR, exist_ok=True)
    jobs = [(n, s) for n in sizes for s in HOLDOUT_SEEDS]
    print('capacity: sizes %s, %s/%s, threshold P(collision) <= %.2f'
          % (sizes, geometry, guidance, threshold))
    print('  %d holdout seeds -> tightest achievable upper bound is %.2f'
          % (len(HOLDOUT_SEEDS), wilson(0, len(HOLDOUT_SEEDS))[2]))
    if resume:
        done = _doneKeys(CAPCSV, lambda r: (int(r['nFan']), int(r['seed'])))
        before = len(jobs)
        jobs = [j for j in jobs if j not in done]
        print('  %d already recorded, %d remaining' % (before - len(jobs),
                                                       len(jobs)))
    print('  %d runs, about %.0f minutes' % (len(jobs), len(jobs) * 7.0 / 60))
    if dryRun:
        print('\n--dry-run: nothing executed')
        return
    if not jobs:
        reportCapacity(threshold)
        return

    f, w = _writer(CAPCSV, 'a' if (resume and os.path.exists(CAPCSV)) else 'w')
    t0 = time.time()
    with f:
        for i, (n, s) in enumerate(jobs, 1):
            row = _run(None, None, '', 'holdout', n, s, geometry, guidance)
            row['mode'] = 'capacity'
            w.writerow(row)
            f.flush()
            el = time.time() - t0
            print('[%4d/%4d] nFan=%-3d seed=%-5d hits=%-3s ETA %.0fm'
                  % (i, len(jobs), n, s, row.get('totalCollisions', '-'),
                     (el / i) * (len(jobs) - i) / 60.0))
    print('\nwrote %s' % CAPCSV)
    reportCapacity(threshold)


GRIDCSV = os.path.join(OUTDIR, 'exp5_grid.csv')


def grid(sizes, prValues, ratios, nSeeds, geometry, guidance, dryRun=False,
         resume=False):
    """2-D sweep over protected range and the slot-spacing RATIO.

    Spacing is swept as a multiple of PR rather than as an absolute, because
    the constraint that matters is relational: slot spacing must exceed the
    protected range or adjacent slots sit inside one another's protected zone
    and the formation is self-repelling. Sweeping them independently wastes
    most of the grid on configurations that are broken by construction, and
    makes the one real relationship hard to see.

    Spawn separation is tied to spacing for the same reason.
    """
    os.makedirs(OUTDIR, exist_ok=True)
    jobs = [(n, pr, r, s)
            for n in sizes for pr in prValues for r in ratios
            for s in HOLDOUT_SEEDS[:nSeeds]]
    print('grid: PR %s x spacing ratio %s x sizes %s' % (prValues, ratios, sizes))
    print('  %d cells x %d seeds = %d runs' %
          (len(sizes) * len(prValues) * len(ratios), nSeeds, len(jobs)))

    if resume:
        done = _doneKeys(GRIDCSV, lambda r: (int(r['nFan']), r['label'],
                                             int(r['seed'])))
        jobs = [j for j in jobs
                if (j[0], 'PR%g_x%g' % (j[1], j[2]), j[3]) not in done]
        print('  %d remaining after resume' % len(jobs))
    print('  about %.0f minutes (%.1f hours)'
          % (len(jobs) * 7.0 / 60.0, len(jobs) * 7.0 / 3600.0))
    if dryRun:
        print('\n--dry-run: nothing executed')
        return
    if not jobs:
        print('nothing to do')
        return

    f, w = _writer(GRIDCSV, 'a' if (resume and os.path.exists(GRIDCSV)) else 'w')
    t0 = time.time()
    with f:
        for i, (n, pr, ratio, s) in enumerate(jobs, 1):
            spacing = pr * ratio
            # PR and VR are SimOptions fields; SLOT_SPACING, SPAWN_MIN_SEP
            # and SLOT_MIN_R are module globals. runOne routes each to the
            # right place. Before that routing existed the PR and VR entries
            # were silently dropped, so any grid run predating the fix varied
            # only the three module-level values.
            ov = {'PR': pr, 'VR': max(8.0, pr),
                  'SLOT_SPACING': spacing, 'SPAWN_MIN_SEP': spacing,
                  'SLOT_MIN_R': max(8.0, pr)}
            row = _run(None, None, 'PR%g_x%g' % (pr, ratio), 'holdout',
                       n, s, geometry, guidance, overrides=ov)
            row['mode'] = 'grid'
            w.writerow(row)
            f.flush()
            el = time.time() - t0
            print('[%4d/%4d] n=%-3d PR=%-5.1f x%-4.2f seed=%-5d hits=%-3s '
                  'ETA %.0fm' % (i, len(jobs), n, pr, ratio, s,
                                 row.get('totalCollisions', '-'),
                                 (el / i) * (len(jobs) - i) / 60.0))
    print('\nwrote %s' % GRIDCSV)
    reportGrid()


def reportGrid():
    rows = _read(GRIDCSV)
    if not rows:
        sys.exit('no grid results')
    sizes = sorted(set(int(r['nFan']) for r in rows))
    labels = sorted(set(r['label'] for r in rows))
    print('\nP(collision) by PR and spacing ratio, holdout seeds')
    for n in sizes:
        print('\n  nFan = %d' % n)
        prs = sorted(set(float(L.split('_')[0][2:]) for L in labels))
        rts = sorted(set(float(L.split('_x')[1]) for L in labels))
        print('    PR \\ ratio  ' + '  '.join('%6.2f' % r for r in rts))
        for pr in prs:
            cells = []
            for rt in rts:
                lab = 'PR%g_x%g' % (pr, rt)
                c = [r for r in rows
                     if int(r['nFan']) == n and r['label'] == lab]
                if not c:
                    cells.append('     -')
                    continue
                k = sum(1 for r in c if r['totalCollisions'] > 0)
                cells.append('%6.2f' % (k / len(c)))
            print('    %8.1f    %s' % (pr, '  '.join(cells)))
    print('\n  ratio = SLOT_SPACING / PR. Values at or below 1.0 are '
          'self-repelling\n  by construction and should be visibly worse.')


def _read(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get('status', 'ok') != 'ok':
                continue
            for k in ('value', 'nFan', 'seed', 'totalCollisions', 'fanOwnCap',
                      'fanForeignCap', 'fanFanSame', 'fanFanCross',
                      'leaderLeader', 'minFanFanGap', 'minFanCapGap',
                      'minLeaderGap', 'redRMS', 'goldRMS',
                      'firstCollisionRelCPA', 'steps'):
                if r.get(k) not in ('', None):
                    r[k] = float(r[k])
                else:
                    r[k] = np.nan
            rows.append(r)
    return rows


def reportCapacity(threshold=0.1):
    rows = _read(CAPCSV)
    if not rows:
        sys.exit('no capacity results -- run `capacity` first')
    sizes = sorted(set(int(r['nFan']) for r in rows))
    print('\ncapacity, holdout seeds only, threshold %.2f' % threshold)
    print('  nFan   clean/runs   P(collision)   95% CI          verdict')
    best = None
    for n in sizes:
        cell = [r for r in rows if int(r['nFan']) == n]
        k = sum(1 for r in cell if r['totalCollisions'] > 0)
        p, lo, hi = wilson(k, len(cell))
        ok = hi <= threshold
        if ok:
            best = n
        print('  %4d   %5d/%-5d  %.2f           [%.2f, %.2f]   %s'
              % (n, len(cell) - k, len(cell), p, lo, hi,
                 'within' if ok else 'EXCEEDS'))
    # --- what is actually colliding, and how far the spawn band was pushed ---
    from grainframe.priority import radiusForCount
    m = loadExperiment()
    halfAngle = getattr(m, 'SPAWN_ARC', np.deg2rad(60.0))
    minR = getattr(m, 'SPAWN_MIN_R', 8.0)
    minSep = getattr(m, 'SPAWN_MIN_SEP', 6.0)
    maxR = getattr(m, 'SPAWN_MAX_R', 18.0)

    print()
    print('  collision composition (totals over all holdout runs), and the')
    print('  spawn radius each size actually needed:')
    print('  nFan   ownCap  foreignCap  fanSame  fanCross  leadLead   '
          'spawnR  expanded?')
    for n in sizes:
        cell = [r for r in rows if int(r['nFan']) == n]
        tot = lambda k: int(np.nansum([r[k] for r in cell]))
        need = radiusForCount(n, minR, halfAngle, minSep)
        print('  %4d   %6d  %10d  %7d  %8d  %8d   %6.1f  %s'
              % (n, tot('fanOwnCap'), tot('fanForeignCap'), tot('fanFanSame'),
                 tot('fanFanCross'), tot('leaderLeader'), need,
                 'YES' if need > maxR else 'no'))
    print('  (SPAWN_MAX_R is %.1f; sizes marked YES had the band auto-widened,'
          % maxR)
    print('   so those wingmen start further back and have less runway to')
    print('   form up before the encounter.)')

    # --- WHEN did the first collision happen, relative to the encounter? ---
    tim = [r for r in rows
           if not np.isnan(r.get('firstCollisionRelCPA', np.nan))]
    print()
    if not tim:
        print('  collision timing: not recorded in this CSV (predates the')
        print('  firstCollisionRelCPA column). Re-run `capacity` to collect it.')
    else:
        print('  timing of the FIRST collision, in seconds relative to the')
        print('  leaders\' closest approach. Negative = before the encounter.')
        print('  nFan   n    median    IQR                near encounter')
        for n in sizes:
            v = np.array([r['firstCollisionRelCPA'] for r in tim
                          if int(r['nFan']) == n])
            if v.size == 0:
                continue
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            near = int(np.sum(np.abs(v) <= 5.0))
            print('  %4d  %3d  %+7.1f    [%+.1f, %+.1f]      %d/%d within 5 s'
                  % (n, v.size, med, q1, q3, near, v.size))
        allv = np.array([r['firstCollisionRelCPA'] for r in tim])
        before = float(np.mean(allv < -5.0))
        near = float(np.mean(np.abs(allv) <= 5.0))
        after = float(np.mean(allv > 5.0))
        print()
        print('  phase split:  before %.0f%%   at encounter %.0f%%   after %.0f%%'
              % (100 * before, 100 * near, 100 * after))
        print()
        # The SIGN is the whole message here, and an earlier version of this
        # report discarded it by testing only |t| <= 5. Before and after the
        # encounter are different failures with different fixes; collapsing
        # them into "not near the encounter" points at the wrong one.
        dom = max((before, 'before'), (near, 'near'), (after, 'after'))[1]
        if dom == 'near':
            print('  Collisions cluster AT the encounter. The corridor and')
            print('  separation parameters are the right targets.')
        elif dom == 'before':
            print('  Collisions happen mostly on the APPROACH, before the')
            print('  captains meet. The squadrons never formed up: more runway')
            print('  (HALF) and slot assignment matter more than avoidance gain.')
        else:
            print('  Collisions happen mostly AFTER the captains pass. The')
            print('  squadrons survive the encounter and then hit each other')
            print('  while RE-FORMING, still interpenetrated. Suspect the')
            print('  encounter gate: SEP_ONLY_ON_ENCOUNTER suppresses cohesion')
            print('  and alignment during the pass, then restores them as a')
            print('  hard switch the moment the last foreign agent leaves')
            print('  ENCOUNTER_RANGE -- so every wingman is yanked back toward')
            print('  its slot simultaneously, through a volume still full of')
            print('  the other squadron. Widening ENCOUNTER_RANGE (keeping the')
            print('  gate shut longer) or ramping it smoothly are the targets,')
            print('  not the corridor.')

    print()
    if best is None:
        print('  No squadron size meets the threshold on its UPPER confidence '
              'limit.\n  With %d seeds the tightest bound achievable even at '
              'zero observed\n  collisions is about %.2f -- so either run more '
              'seeds or relax the\n  threshold. Reporting the point estimate '
              'instead would overstate\n  what this many trials can support.'
              % (len(HOLDOUT_SEEDS), wilson(0, len(HOLDOUT_SEEDS))[2]))
    else:
        print('  N_max = %d wingmen per squadron.' % best)
        print('  Read as: with 95%% confidence the collision probability at '
              'this size\n  is at or below %.2f. Larger squadrons are not '
              'ruled safe by this data.' % threshold)


# ---------------------------------------------------------------------
#  Figures
# ---------------------------------------------------------------------

def figures():
    rows = _read(TUNECSV)
    if not rows:
        sys.exit('no sensitivity results -- run `sensitivity` first')
    params = sorted(set(r['param'] for r in rows if r['param']))
    print('read %d runs over %d parameters' % (len(rows), len(params)))

    ncol = 3
    nrow = int(np.ceil(len(params) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 2.9 * nrow),
                             squeeze=False)
    print('\n  best value per parameter, chosen on TUNE seeds,')
    print('  with the holdout result at that value:')
    for ax, name in zip([a for r in axes for a in r], params):
        sub = [r for r in rows if r['param'] == name]
        labels = sorted(set(r['label'] for r in sub),
                        key=lambda L: float(L))
        for split, colour, marker in (('tune', 'tab:gray', 's'),
                                      ('holdout', 'tab:blue', 'o')):
            ps, los, his = [], [], []
            for lab in labels:
                cell = [r for r in sub
                        if r['label'] == lab and r['split'] == split]
                k = sum(1 for r in cell if r['totalCollisions'] > 0)
                p, lo, hi = wilson(k, len(cell))
                ps.append(p)
                los.append(p - lo)
                his.append(hi - p)
            ax.errorbar(range(len(labels)), ps, yerr=[los, his], marker=marker,
                        ms=4, lw=1.0, capsize=2.5, color=colour, label=split)
            if split == 'tune':
                bestI = int(np.nanargmin(ps))
        hold = [r for r in sub
                if r['label'] == labels[bestI] and r['split'] == 'holdout']
        kh = sum(1 for r in hold if r['totalCollisions'] > 0)
        ph, _, hh = wilson(kh, len(hold))
        print('    %-14s best %-6s   holdout P = %.2f (upper %.2f)'
              % (name, labels[bestI], ph, hh))
        ax.axvline(bestI, color='tab:red', ls=':', lw=1.0)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(name, fontsize=8.5)
        ax.set_ylim(-0.05, 1.05)
    for a in [a for r in axes for a in r][len(params):]:
        a.axis('off')
    axes[0][0].set_ylabel('P(at least one collision)')
    axes[0][-1].legend(fontsize=6.5, loc='upper right')
    _save(fig, 'exp5_sensitivity')

    print('\n  The dotted line marks the tune-set minimum. Where the holdout '
          'curve\n  disagrees with it, the tune-set minimum was noise, not '
          'signal -- which\n  is exactly what the split exists to reveal.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['sensitivity', 'capacity', 'figures',
                                     'report', 'grid', 'gridreport'])
    ap.add_argument('--nFan', type=int, default=6)
    ap.add_argument('--geometry', default='headon')
    ap.add_argument('--guidance', default='vfield')
    ap.add_argument('--params', default='',
                    help='comma-separated subset; default is all of them')
    ap.add_argument('--sizes', default='2,3,4,6,8,10,12')
    ap.add_argument('--threshold', type=float, default=0.10)
    ap.add_argument('--prs', default='8,10,13,16,20',
                    help='protected range values for the grid')
    ap.add_argument('--ratios', default='1.0,1.25,1.5,2.0',
                    help='SLOT_SPACING / PR ratios for the grid')
    ap.add_argument('--dry-run', dest='dryRun', action='store_true',
                    help='print the grid and the time estimate, run nothing')
    ap.add_argument('--resume', action='store_true',
                    help='skip cells already in the CSV. Use this for any long '
                         'unattended run: without it a crash at hour six loses '
                         'everything.')
    ap.add_argument('--tune-seeds', dest='nTune', type=int, default=10)
    ap.add_argument('--holdout-seeds', dest='nHold', type=int, default=10)
    args = ap.parse_args()

    TUNE_SEEDS, HOLDOUT_SEEDS = makeSeeds(args.nTune, args.nHold)
    import exp5_tune as _self
    _self.TUNE_SEEDS, _self.HOLDOUT_SEEDS = TUNE_SEEDS, HOLDOUT_SEEDS

    if args.mode == 'sensitivity':
        matplotlib.use('Agg')
        chosen = ([p.strip() for p in args.params.split(',') if p.strip()]
                  or list(SWEEPS))
        bad = [p for p in chosen if p not in SWEEPS]
        if bad:
            sys.exit('unknown parameter(s): %s\nknown: %s'
                     % (', '.join(bad), ', '.join(sorted(SWEEPS))))
        sensitivity(args.nFan, args.geometry, args.guidance, chosen,
                    dryRun=args.dryRun, resume=args.resume)
    elif args.mode == 'capacity':
        matplotlib.use('Agg')
        sizes = [int(s) for s in args.sizes.split(',')]
        capacity(args.geometry, args.guidance, sizes, args.threshold,
                 dryRun=args.dryRun, resume=args.resume)
    elif args.mode == 'grid':
        matplotlib.use('Agg')
        grid([int(s) for s in args.sizes.split(',')],
             [float(v) for v in args.prs.split(',')],
             [float(v) for v in args.ratios.split(',')],
             args.nHold, args.geometry, args.guidance,
             dryRun=args.dryRun, resume=args.resume)
    elif args.mode == 'gridreport':
        reportGrid()
    elif args.mode == 'report':
        reportCapacity(args.threshold)
    else:
        figures()