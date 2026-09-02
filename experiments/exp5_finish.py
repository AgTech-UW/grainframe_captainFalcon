"""Experiment 5: the runs that stand between the draft and a submission.

Three jobs the existing runners cannot do, plus reports.

  ablation   Contribution 1 has no ablation. Runs the full scheme against
             variants with one mechanism removed each, same seeds, and
             reports collision AND arrival probability per variant. Arrival
             matters because several mechanisms exist to prevent deadlock
             rather than collision, so removing them may leave collisions
             flat while runs stop finishing. Judging variants on collisions
             alone would call that a success.

  capacity2  Table 1 (tab:capacity), both columns, one recorded config hash
             per column, same seed count. The satisfied column runs the
             experiment file's own configuration. The violated column is the
             SAME configuration with SLOT_SPACING overridden below PR, so
             the columns differ in exactly one number.

  grid2      The PR x (delta/PR) grid, as exp5_tune.py grid, but accepting
             arbitrary --set KEY=VALUE overrides (SEP_PROFILE=norm,
             RboidFactor=0.5, ...) and writing to a --tag'd CSV so variant
             grids never mix with the baseline in one file.

  report     Wilson tables from any CSV this script wrote.

USAGE (see the runbook for the full publication sequence)
-----
  python experiments/exp5_finish.py ablation  --sizes 4,8 --seeds 50
  python experiments/exp5_finish.py capacity2 --sizes 3,4,6,8,12 --seeds 50
  python experiments/exp5_finish.py grid2 --prs 8,10,13,16,20 \
         --ratios 0.5,0.75,1.0,1.25,1.5,2.0 --sizes 3,4 --seeds 30 --tag flat
  python experiments/exp5_finish.py grid2 ... --set SEP_PROFILE=norm --tag norm
  python experiments/exp5_finish.py grid2 ... --set RboidFactor=0.5 --tag halfR
  python experiments/exp5_finish.py report --csv results/exp5_ablation.csv

Every mode supports --resume (skip rows already recorded) and --dry-run
(print the job list and time estimate, run nothing). Results append after
every run, so an interrupted job loses at most one simulation.
"""

import os
import sys
import csv
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(os.path.dirname(HERE), 'results')

# Reuse the run machinery and the CSV schema from the tuning harness, so
# every row this script writes carries the same provenance columns
# (configHash above all) as every other results file in the repository.
sys.path.insert(0, HERE)
import exp5_tune as T                      # noqa: E402  (path shim above)

FIELDS = T.FIELDS
SEC_PER_RUN = 7.0                          # coarse planning figure


# ---------------------------------------------------------------------
#  The ablation variants
# ---------------------------------------------------------------------
#
# One mechanism removed per variant, everything else identical. The override
# names route through runOne: booleans and FAN_SWIRL are module globals of
# exp5_priority.py, so a typo fails loudly there rather than silently here.
#
#   full        the scheme as described in Secs. 5.2 and 5.3
#   noswirl     FAN_SWIRL = 0, no shared handedness in the head-on
#   noorder     USE_GLOBAL_PRIORITY = False, no total order (also disables
#               the brake, which is gated on the order in the experiment)
#   nocorridor  USE_CORRIDOR = False, no clearing ahead of a leader
#   nobrake     USE_FAN_BRAKE = False, order retained, brake action removed
#
# noorder removing the brake as a side effect is a fact about the
# implementation worth keeping: the brake without an order is exactly the
# both-agents-brake deadlock of Sec. 5.2, so the meaningful ablations are
# order+brake together (noorder) and brake alone (nobrake).

VARIANTS = {
    'full':       {},
    'noswirl':    {'FAN_SWIRL': 0.0},
    'noorder':    {'USE_GLOBAL_PRIORITY': False},
    'nocorridor': {'USE_CORRIDOR': False},
    'nobrake':    {'USE_FAN_BRAKE': False},
}

ABLCSV = os.path.join(OUTDIR, 'exp5_ablation.csv')
CAP2CSV = os.path.join(OUTDIR, 'exp5_capacity2.csv')


def _openAppend(path, resume):
    exists = os.path.exists(path)
    mode = 'a' if (resume and exists) else 'w'
    f = open(path, mode, newline='')
    w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
    if mode == 'w':
        w.writeheader()
    return f, w


def _done(path, keyFn):
    if not os.path.exists(path):
        return set()
    out = set()
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            if r.get('status', 'ok') != 'ok':
                continue
            out.add(keyFn(r))
    return out


def _estimate(nJobs):
    print('  %d runs, about %.0f minutes (%.1f hours)'
          % (nJobs, nJobs * SEC_PER_RUN / 60.0, nJobs * SEC_PER_RUN / 3600.0))


def _loop(jobs, path, resume, runFn):
    f, w = _openAppend(path, resume)
    t0 = time.time()
    with f:
        for i, job in enumerate(jobs, 1):
            row = runFn(job)
            w.writerow(row)
            f.flush()
            el = time.time() - t0
            print('[%4d/%4d] %s hits=%-3s arrived=%s/%s ETA %.0fm'
                  % (i, len(jobs), row.get('label', ''),
                     row.get('totalCollisions', '-'),
                     row.get('redArrived', '?'), row.get('goldArrived', '?'),
                     (el / i) * (len(jobs) - i) / 60.0))
    print('\nwrote %s' % path)


# ---------------------------------------------------------------------
#  Modes
# ---------------------------------------------------------------------

def ablation(sizes, seeds, geometry, guidance, dryRun, resume,
             extra=None, tag=''):
    # Tagged path so an ablation under a different geometry or a harder
    # configuration never lands in the same file as the baseline one. The
    # resume key is (label, nFan, seed), which does NOT include geometry, so
    # sharing a file across geometries would silently skip every row.
    path = (ABLCSV if not tag
            else os.path.join(OUTDIR, 'exp5_ablation_%s.csv' % tag))
    extra = dict(extra or {})
    jobs = [(v, n, s) for v in VARIANTS for n in sizes for s in seeds]
    print('ablation: %s x sizes %s x %d holdout seeds, %s/%s'
          % (list(VARIANTS), sizes, len(seeds), geometry, guidance))
    if extra:
        print('  extra overrides applied to EVERY variant: %s' % extra)
    print('  writing %s' % path)
    _estimate(len(jobs))
    if resume:
        done = _done(path, lambda r: (r['label'], int(r['nFan']),
                                      int(r['seed'])))
        jobs = [j for j in jobs if (j[0], j[1], j[2]) not in done]
        print('  %d remaining after resume' % len(jobs))
    if dryRun or not jobs:
        return report(path) if not dryRun else None

    def one(job):
        v, n, s = job
        ov = dict(extra)
        ov.update(VARIANTS[v])      # variant flags win over --set
        row = T._run(None, None, v, 'holdout', n, s, geometry, guidance,
                     overrides=ov)
        row['mode'] = 'ablation'
        return row

    _loop(jobs, path, resume, one)
    report(path)


def capacity2(sizes, seeds, spacingViolated, geometry, guidance,
              dryRun, resume, extra=None, tag=''):
    path = (CAP2CSV if not tag
            else os.path.join(OUTDIR, 'exp5_capacity2_%s.csv' % tag))
    extra = dict(extra or {})
    cols = [('satisfied', dict(extra)),
            ('violated', dict(extra, **{'SLOT_SPACING': spacingViolated,
                                        'SPAWN_MIN_SEP': spacingViolated}))]
    jobs = [(lab, ov, n, s) for lab, ov in cols for n in sizes for s in seeds]
    print('capacity2: satisfied (config as-is) and violated '
          '(SLOT_SPACING=%g) x sizes %s x %d seeds' %
          (spacingViolated, sizes, len(seeds)))
    if extra:
        print('  extra overrides applied to BOTH columns: %s' % extra)
    print('  writing %s' % path)
    print('  spawn separation follows spacing in the violated column, as it '
          'does in the grid,\n  so the two columns differ only in the '
          'quantity the condition is about.')
    _estimate(len(jobs))
    if resume:
        done = _done(path, lambda r: (r['label'], int(r['nFan']),
                                      int(r['seed'])))
        jobs = [j for j in jobs if (j[0], j[2], j[3]) not in done]
        print('  %d remaining after resume' % len(jobs))
    if dryRun or not jobs:
        return report(path) if not dryRun else None

    def one(job):
        lab, ov, n, s = job
        row = T._run(None, None, lab, 'holdout', n, s, geometry, guidance,
                     overrides=dict(ov))
        row['mode'] = 'capacity2'
        return row

    _loop(jobs, path, resume, one)
    report(path)


def grid2(prs, ratios, sizes, seeds, extra, tag, geometry, guidance,
          dryRun, resume):
    path = os.path.join(OUTDIR, 'exp5_grid2_%s.csv' % tag)
    jobs = [(n, pr, r, s) for n in sizes for pr in prs for r in ratios
            for s in seeds]
    print('grid2 [%s]: PR %s x ratio %s x sizes %s x %d seeds'
          % (tag, prs, ratios, sizes, len(seeds)))
    if extra:
        print('  extra overrides: %s' % extra)
    _estimate(len(jobs))
    if resume:
        done = _done(path, lambda r: (int(r['nFan']), r['label'],
                                      int(r['seed'])))
        jobs = [j for j in jobs
                if (j[0], 'PR%g_x%g' % (j[1], j[2]), j[3]) not in done]
        print('  %d remaining after resume' % len(jobs))
    if dryRun or not jobs:
        return report(path) if not dryRun else None

    def one(job):
        n, pr, ratio, s = job
        spacing = pr * ratio
        ov = {'PR': pr, 'VR': max(8.0, pr),
              'SLOT_SPACING': spacing, 'SPAWN_MIN_SEP': spacing,
              'SLOT_MIN_R': max(8.0, pr)}
        ov.update(extra)
        row = T._run(None, None, 'PR%g_x%g' % (pr, ratio), 'holdout',
                     n, s, geometry, guidance, overrides=ov)
        row['mode'] = 'grid2:%s' % tag
        return row

    _loop(jobs, path, resume, one)
    report(path)


# ---------------------------------------------------------------------
#  Reporting: collision AND arrival, Wilson intervals throughout
# ---------------------------------------------------------------------

def report(path):
    if not os.path.exists(path):
        sys.exit('no results at %s' % path)
    with open(path, newline='') as f:
        rows = [r for r in csv.DictReader(f) if r.get('status') == 'ok']
    if not rows:
        sys.exit('no ok rows in %s' % path)

    groups = {}
    for r in rows:
        groups.setdefault((r['label'], int(r['nFan'])), []).append(r)

    hashes = sorted(set(r['configHash'] for r in rows))
    print('\n%s' % path)
    print('config hashes present: %s' % ', '.join(hashes))
    print('%-12s %4s %6s  %-22s %-22s %s'
          % ('label', 'nFan', 'n', 'P(collision) [95%]',
             'P(both arrive) [95%]', 'collFree&noArrive'))
    for (lab, n) in sorted(groups):
        g = groups[(lab, n)]
        N = len(g)
        kC = sum(1 for r in g if int(r['totalCollisions']) > 0)
        haveArr = all(r.get('redArrived') not in ('', None) for r in g)
        if haveArr:
            kA = sum(1 for r in g
                     if r['redArrived'] == '1' and r['goldArrived'] == '1')
            # clean of collisions yet not both arrived: the deadlock-shaped
            # outcome that collision counts alone cannot see
            kD = sum(1 for r in g
                     if int(r['totalCollisions']) == 0
                     and not (r['redArrived'] == '1'
                              and r['goldArrived'] == '1'))
            _, aLo, aHi = T.wilson(kA, N)
            arrTxt = '%2d/%-3d [%.2f, %.2f]' % (kA, N, aLo, aHi)
            dlTxt = '%d' % kD
        else:
            arrTxt, dlTxt = 'not recorded', '-'
        _, cLo, cHi = T.wilson(kC, N)
        print('%-12s %4d %6d  %2d/%-3d [%.2f, %.2f]   %-22s %s'
              % (lab, n, N, kC, N, cLo, cHi, arrTxt, dlTxt))


# ---------------------------------------------------------------------

def _parseSet(pairs):
    out = {}
    for p in pairs or []:
        if '=' not in p:
            sys.exit('--set needs KEY=VALUE, got %r' % p)
        k, v = p.split('=', 1)
        for cast in (int, float):
            try:
                out[k] = cast(v)
                break
            except ValueError:
                continue
        else:
            if v.lower() in ('true', 'false'):
                out[k] = v.lower() == 'true'
            else:
                out[k] = v
    return out


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['ablation', 'capacity2', 'grid2',
                                     'report'])
    ap.add_argument('--sizes', default='4,8')
    ap.add_argument('--seeds', type=int, default=50,
                    help='holdout seeds per cell')
    ap.add_argument('--geometry', default='headon')
    ap.add_argument('--guidance', default='vfield')
    ap.add_argument('--violated-spacing', dest='vspace', type=float,
                    default=15.0,
                    help='SLOT_SPACING for the violated capacity column; the '
                         'satisfied column uses the config as written')
    ap.add_argument('--prs', default='8,10,13,16,20')
    ap.add_argument('--ratios', default='0.5,0.75,1.0,1.25,1.5,2.0')
    ap.add_argument('--set', dest='sets', action='append', default=[],
                    metavar='KEY=VALUE',
                    help='extra overrides for grid2, repeatable '
                         '(SEP_PROFILE=norm, RboidFactor=0.5, ...)')
    ap.add_argument('--tag', default='flat',
                    help='suffix for the grid2 CSV so variants never mix')
    ap.add_argument('--csv', default='',
                    help='CSV path for report mode')
    ap.add_argument('--dry-run', dest='dryRun', action='store_true')
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()

    # Holdout seeds from the same generator as every other results file, so
    # numbers are comparable across studies.
    _, HOLD = T.makeSeeds(10, args.seeds)
    seeds = HOLD
    sizes = [int(s) for s in args.sizes.split(',')]

    if args.mode == 'ablation':
        ablation(sizes, seeds, args.geometry, args.guidance,
                 args.dryRun, args.resume,
                 extra=_parseSet(args.sets),
                 tag=('' if args.tag == 'flat' else args.tag))
    elif args.mode == 'capacity2':
        capacity2(sizes, seeds, args.vspace, args.geometry, args.guidance,
                  args.dryRun, args.resume,
                  extra=_parseSet(args.sets),
                  tag=('' if args.tag == 'flat' else args.tag))
    elif args.mode == 'grid2':
        grid2([float(v) for v in args.prs.split(',')],
              [float(v) for v in args.ratios.split(',')],
              sizes, seeds, _parseSet(args.sets), args.tag,
              args.geometry, args.guidance, args.dryRun, args.resume)
    else:
        report(args.csv or ABLCSV)
