"""Sandbox: two squads approach each other. One run, one plot.

ALL the knobs live in the PARAMETERS block below -- edit and rerun.

Each leader is a Dubins car (constant speed, bounded turn rate) doing:

    desired velocity  =  GUIDANCE (follow my path)  +  REPULSION (dodge the
                                                       other leader)

then bang-bang steering toward that vector. The two terms are deliberately
separate so you can change one without disturbing the other.

  GUIDANCE  -- vector field path following, Nelson/Barber/McLain/Beard,
               IEEE T-RO 23(3), 2007.
  REPULSION -- Khatib obstacle potential, IJRR 5(1), 1986.

Run it:
    python experiments/exp5_simple.py            static plot
    python experiments/exp5_simple.py anim       live playback
    python experiments/exp5_simple.py collide    collision report only
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from matplotlib import pyplot as plt
from grainframe import SimOptions
from grainframe.data_io import apply_dataset
from grainframe.paths import straightPath
from grainframe.dynamics import (falconSteering, ackermannClamp,
                                 WaypointTracker)
from grainframe.guidance import (guidanceWaypoint, guidanceVectorField,
                                 crossTrackError)
from grainframe.avoidance import separationFlat, repulsionKhatib
from grainframe.deadlock import canAvoid, yieldDecision
from grainframe.flocking import flockStep
from grainframe.priority import (buildRanks, fanBrakeRanked, corridorClear,
                                 spawnFormation, spawnRandomBehind,
                                 formationSlots)


# =====================================================================
#  PARAMETERS: edit me
# =====================================================================

SEED  = 1051            # change for a different random squadron layout
N_FAN = 3            # fanboids per squadron

GEOMETRY = 'headon'  # 'headon' = straight at each other on the same line
                     # 'cross'  = the X crossing at the origin

HALF = 40.0          # how far from the middle each leader starts, so the
                     # path is 2*HALF long. tMax scales off the path length
                     # automatically, so raising this buys form-up time at no
                     # other cost: at HALF = 40 and v = 1 the squadrons have
                     # only ~40 s before the leaders meet at the origin, which
                     # is not enough for a scattered spawn to settle. Raise to
                     # 80-120 if wingmen are still tangled at the crossing.
LANE_OFFSET = 0.0    # shove Gold's lane sideways. 0 = perfectly symmetric.

# Gain overrides from config.py. None = keep the stock value.
OVERRIDES = {
    # 'SF': 5.0,              # fanboid separation strength
    # 'CF': 0.8,              # fanboid cohesion
    # 'AF': 2.0,              # fanboid alignment
    # 'fanLeaderFactor': 1.0, # fanboid pull toward own leader
    # 'R': 5.0,               # leader turn radius (bigger = clumsier)
    'PR': 16.0,
    'VR': 16.0,
}

# ---------------------------------------------------------------------
#  GUIDANCE: how each leader follows his own path
# ---------------------------------------------------------------------
# 'vfield'   -- vector field. For every point in the plane it defines the
#               heading you SHOULD have. Far off the line you fly in at
#               CHI_INF; as you close, the desired heading rotates to match
#               the path. Cross-track error and heading error therefore null
#               at the same moment, which is what stops the oscillation.
#
# 'waypoint' -- the old behaviour: aim straight at the next waypoint. Kept
#               so you can reproduce the limit cycle for comparison. It
#               oscillates because it nulls POSITION but leaves you pointed
#               sideways: arriving at a waypoint from off-line means arriving
#               with ~90 deg of heading error, every single time.
GUIDANCE = 'vfield'

K_CROSS  = 0.5                   # how sharply the field bends back to the line.
                                 # Too small = lazy recovery. Too large = the
                                 # field demands more curvature than the car
                                 # has and you get chatter. 0.4-0.6 is the
                                 # sweet spot at R=5.
CHI_INF  = np.deg2rad(70.0)      # approach angle when far off the line (deg).
                                 # 90 would be "fly straight at it", which is
                                 # untrackable. 70-85 is the useful range.
GUID_MAG = 10.0                  # magnitude of the guidance vector. MUST be a
                                 # constant. The old code used the distance to
                                 # the waypoint, which slid from 10 down to 0
                                 # across each leg -- meaning the balance
                                 # between path-following and avoidance
                                 # silently changed depending on where in the
                                 # leg the encounter happened.

# Per-leader path pull. Scales GUID_MAG, so it sets how strongly each leader
# resists being pushed off his line relative to the repulsion.
LEADER_FACTOR_RED  = 1.0
LEADER_FACTOR_GOLD = 1.0

# ---------------------------------------------------------------------
#  REPULSION: how each leader dodges the other
# ---------------------------------------------------------------------
# Per-leader protected range: how early each leader reacts to the OTHER.
# Different values break the mirror symmetry -- the one with the bigger range
# gives way, the one with the smaller range mostly holds his line.
PR_RED  = 12.0
PR_GOLD = 10.0

# True  -- Khatib form. Force is zero at the range boundary and rises steeply
#          toward contact. Smooth, no discontinuity.
# False -- legacy flat form, SF * displacement. This was BACKWARDS: strongest
#          at the boundary (5 * 12 = 60, versus a path pull of ~7) and it
#          jumped from 0 to 60 the instant the other leader crossed PR.
# IMPORTANT: this flag applies to the LEADERS ONLY. The fanboids keep the
# stock Reynolds separation rule (see fanSeparation below) so the boids model
# under study is not altered by a change to the leader controller.
SMOOTH_REPULSION = True

REP_GAIN = 45.0      # repulsion force at HALF the protected range.
                     # NOTE: opts.SF does NOT transfer to the Khatib form --
                     # the two laws have wildly different scales. This is its
                     # own independent knob. 45 reproduces roughly the same
                     # swerve magnitude the old flat rule produced.

# ---------------------------------------------------------------------
#  INFEASIBILITY STOP (paper Section 4.1)
# ---------------------------------------------------------------------
# If a leader physically cannot turn hard enough to avoid the other, he STOPS
# rather than orbiting uselessly. Stopping is always feasible regardless of
# turn radius.
USE_INFEASIBILITY_STOP = True
STOP_RANGE     = 6.0    # start worrying about a stop inside this distance
STOP_LOOKAHEAD = 3200   # steps to roll forward when testing feasibility.
                        # This is a DISTANCE in disguise: steps * v * dt.
                        # 3200 * 1.0 * 0.01 = 32 units = one full 2*pi*R turn
                        # at R=5. The old value of 40 was 0.4 units of travel,
                        # i.e. no lookahead at all -- canAvoid() collapsed to
                        # "am I inside STOP_MARGIN right now".
STOP_MARGIN    = 1.5    # stop if the best-case future gap stays below this

# WHO YIELDS. Deadlock cannot be resolved by symmetric reactive rules alone
# without some global information (Duhaut et al. 2007); the standard fix is a
# priority rule, because the alternative -- analysing the deadlock equilibrium
# directly -- scales roughly exponentially in the number of agents (Grover,
# Liu & Sycara, IJRR 42(6), 2023). So each leader carries a static rank.
#
# LOWER number = STAND-ON (holds course and speed, COLREGs Rule 17).
# HIGHER number = GIVE-WAY (yields first, COLREGs Rule 16).
# Written as a list indexed by leader so adding a third squadron is just a
# third entry -- nothing below is hard-coded to two ranks.
LEADER_PRIORITY = [0, 1]        # Red stands on, Gold gives way

# COLREGs Rule 17(b): if the give-way vessel's action alone cannot prevent the
# collision, the stand-on vessel must also act.
#
# OFF BY DEFAULT, and the reason is instructive. Rule 17(b) says the stand-on
# vessel takes "such action as will best aid to avoid collision" -- in a
# head-on that means TURN, not stop. Implemented below as a freeze (the only
# escalation currently available), it produces permanent gridlock: once the
# give-way leader halts, he becomes a stationary obstacle dead ahead, which
# never stops being infeasible, so the stand-on leader freezes too and neither
# recovers. Measured at PR_RED = PR_GOLD = 3.0: both leaders frozen for ~11000
# steps, zero cross-track, neither reaches its goal.
#
# The correct escalation is a hard turn (bypass the bang-bang deadband and
# command max rate away from the threat), which is a change to the steering
# path rather than to this flag. Until that exists, leaving this False gives a
# clean one-sided yield.
USE_RULE_17B = False

# ---------------------------------------------------------------------
#  RULE PRIORITY DURING AN ENCOUNTER
# ---------------------------------------------------------------------
# Cross-squad coupling is already separation-only: cohesion and alignment are
# fed same-squad neighbours only. The counter-productive term is the fan's OWN
# squadron. While the squads interpenetrate, cohesion pulls each fan back to
# its squadmates' centre of mass and alignment pulls its heading back to the
# squad average -- both of which point straight back into the lane it is
# trying to separate out of. The two rules cancel and the swerve never
# executes.
#
# Reynolds gives separation strict priority for exactly this reason. While any
# foreign agent (other squad's fans, or the other leader) is within
# ENCOUNTER_RANGE, alignment and cohesion switch OFF for that fanboid and it
# flies on separation + leader pull alone. Outside the encounter the full rule
# is unchanged, so squadrons still form up before and after the pass.
SEP_ONLY_ON_ENCOUNTER = True
ENCOUNTER_RANGE = None   # None -> opts.VR, the range cohesion/alignment act
                         # over anyway. Set a number to decouple the two.

# ---------------------------------------------------------------------
#  GLOBAL PRIORITY: every agent on the field is ranked
# ---------------------------------------------------------------------
# The leaders needed a rank because two vehicles that both stop are deadlocked
# forever. The same argument applies to the wingmen. A speed floor -- letting
# a jammed pair crawl instead of freeze -- makes deadlock slow rather than
# resolving it, and leaves the outcome dependent on whatever asymmetry happens
# to sit in the initial conditions.
#
# So: ONE total order over everything. key = (tier, squadRank, indexInSquad).
# Lower sorts first and means STAND-ON. Captains are tier 0 and therefore
# outrank every wingman of every squadron -- "captains trump fans, always" --
# which is what keeps a captain's path-following exact.
USE_GLOBAL_PRIORITY = True

SQUAD_RANK = [0, 1]     # ordering over squadrons. [0, 1] = Red squadron
                        # outranks Gold. Swap to [1, 0] for Gold over Red.
                        # NOTE this is separate from LEADER_PRIORITY above,
                        # which orders the two captains against each other;
                        # keep them consistent or the captains and their
                        # squadrons will disagree about who defers.

# --- wingman vs wingman: BRAKE. Both are manoeuvrable and the rank
#     guarantees only one of any pair yields, so nobody parks in anybody's
#     path and a full stop is safe.
# --- wingman vs wingman, HEAD-ON: swirl, not brake ---
# Braking is the right yield for an OVERTAKING pair: slowing genuinely opens
# the gap. It is the wrong yield for a HEAD-ON pair, because the closing rate
# is dominated by the other agent, and since the ranking means only one of the
# pair brakes, the braking wingman becomes a stationary target for the one
# that does not. Measured: a cross-squadron pair passed at 0.037 with the
# brake active.
#
# A head-on is symmetric, so no gain of pure repulsion separates the pair --
# the force points along the line joining them. Rotating it gives both a
# consistent handedness and they pass starboard-to-starboard (COLREGs Rule
# 14, which prescribes a mutual starboard turn for a head-on rather than a
# give-way/stand-on hierarchy). It needs no ranking, which is why it works
# between squadrons that share no ordering.
#
# FAN_SWIRL = 0.0 reproduces exactly the previous behaviour, so this is a
# clean ablation.
FAN_SWIRL = 1.2
FAN_SWIRL_HEADON_ONLY = True   # False = swirl every repulsion, not just
                               # head-on pairs (the multifleet capSwirl style)

USE_FAN_BRAKE = True
BRAKE_RANGE = 3.0       # start easing off inside this gap
BRAKE_STOP  = 1.4       # fully stopped at this gap (> collisionRadius = 1.0)
BRAKE_FLOOR = 0.0       # 0.0 is now safe: with a total order only one of any
                        # pair brakes, so a full stop cannot deadlock. Raise
                        # it to reproduce the old symmetric scheme, where the
                        # floor was load-bearing.

# --- wingman vs captain: NEVER brake. A captain is a constant-speed vehicle
#     that does not yield to wingmen, so a stopped wingman in front of one is
#     not avoiding a collision, it is guaranteeing one. Get out of the lane
#     instead, at full speed.
USE_CORRIDOR = True
CORR_RANGE  = 12.0      # look this far up the captain's track
CORR_HALF   = 4.5       # lane half-width to clear
CORR_GAIN   = 60.0

# --- formation slots: the measured collision source is a wingman hitting its
#     OWN captain, because one shared attractor makes every wingman converge
#     on a single point they are all also required to avoid. Distinct slots
#     are separated by construction.
USE_SLOTS   = True
SLOT_SPACING = 20.0      # MUST exceed opts.PR (5.0) or the formation is
                        # self-repelling: separation fights the slot pull and
                        # the formation cannot hold. Measured at 3.0 the slots
                        # made collisions WORSE than no slots at all.
SLOT_ROWS    = 2        # only used by shape='column'
SLOT_SHAPE   = 'vee'    # 'column' | 'vee' | 'echelon'.
                        # 'column' puts wingmen directly astern of the
                        # captain, i.e. inside the very corridor CORR_HALF is
                        # trying to clear -- the slot pull and the corridor
                        # push then fight each other. Measured with 6 wingmen
                        # and CORR_HALF = 4.5: column places 3 of 6 in the
                        # lane, vee and echelon place none.
SLOT_SWEEP   = np.deg2rad(45.0)   # angle back from abeam for vee/echelon.
                                  # Smaller = wider flatter V.
SLOT_MIN_R   = 8.0      # every wingman starts at least this far from its
                        # captain, so nobody begins inside a protected range

# Deterministic spawn. On a real field the start and end states are known --
# vehicles are placed, not sprinkled. Random gaussian scatter puts some
# fraction of wingmen inside the captain's protected range at t = 0, so they
# open the run resolving a conflict that should not exist and the collision
# statistics pick up artefacts of the initial condition. Set > 0 only to test
# robustness to placement error.
SPAWN_JITTER = 0.0

# SPAWN_MODE selects how wingmen are placed at t = 0. In BOTH modes the slot
# attractors are unchanged and every wingman starts on its captain's heading
# -- only the starting POSITIONS differ. This separation matters: the slots
# are what eliminated wingman-captain collisions, and they keep working under
# random placement, so the formation genuinely assembles itself rather than
# being handed to the controller.
#
#   'formation' -- every wingman starts exactly in its slot. Fully
#                  deterministic, zero transient. Use for repeatability.
#   'random'    -- uniform in an annular sector astern, subject to a minimum
#                  radius from the captain and a minimum separation between
#                  wingmen. Use to show the formation forming.
SPAWN_MODE = 'random'

SPAWN_MIN_R  = 8.0      # inner radius: keeps everyone outside the captain's
                        # protected range at t = 0
SPAWN_MAX_R  = 18.0     # outer radius of the spawn band
SPAWN_ARC    = np.deg2rad(60.0)   # half-width of the rear sector.
                                  # 90 deg = the whole rear hemisphere.
SPAWN_MIN_SEP = 13.0     # minimum gap between wingmen at spawn. Keep above
                        # opts.PR (5.0) or they start inside each other's
                        # protected range and spend the opening seconds
                        # unpiling instead of forming up.

FAN_SPREAD = 3.0     # how loosely each squadron spawns around its leader
CIRCLE_R   = 2.5     # radius of the drawn COLLISION circles
# =====================================================================

RED, GOLD = 'crimson', 'goldenrod'


# ---------------------------------------------------------------------
#  Guidance law
# ---------------------------------------------------------------------

def guidance(x, y, segA, segB, wpX, wpY, gain):
    """Dispatch to the configured guidance law (grainframe.guidance).

    Thin wrapper: it exists only to bind this experiment's PARAMETERS block to
    the shared implementation, so the knobs stay visible at the top of this
    file while the law itself is shared and testable.
    """
    if GUIDANCE == 'waypoint':
        return guidanceWaypoint(x, y, wpX, wpY, gain)
    out = guidanceVectorField(x, y, segA, segB, gain,
                              kCross=K_CROSS, chiInf=CHI_INF,
                              magnitude=GUID_MAG)
    if out is None:                       # degenerate segment
        return guidanceWaypoint(x, y, wpX, wpY, gain)
    return out


def leaderRepulsion(xb, yb, xOthers, yOthers, opts, range_=None):
    """Leader-vs-leader repulsion. Khatib when SMOOTH_REPULSION, else the
    legacy flat rule, so the paper can report the cost of each."""
    if range_ is None:
        range_ = opts.PR
    if SMOOTH_REPULSION:
        return repulsionKhatib(xb, yb, xOthers, yOthers,
                               range_=range_, gain=REP_GAIN)
    return separationFlat(xb, yb, xOthers, yOthers,
                          strength=opts.SF, range_=range_)


def makePaths():
    if GEOMETRY == 'headon':
        dataR = straightPath(-HALF, 0, +HALF, 0)                     # left -> right
        dataG = straightPath(+HALF, LANE_OFFSET, -HALF, LANE_OFFSET) # right -> left
    else:                                                            # 'cross'
        dataR = straightPath(-HALF, -HALF, +HALF, +HALF)
        dataG = straightPath(-HALF, +HALF + LANE_OFFSET, +HALF, -HALF + LANE_OFFSET)
    return dataR, dataG


# ---------------------------------------------------------------------
#  Simulation
# ---------------------------------------------------------------------

def run(plot=True):
    opts = SimOptions()
    dataR, dataG = makePaths()
    apply_dataset(opts, dataR)
    for name, val in OVERRIDES.items():
        if val is not None:
            setattr(opts, name, val)
    np.random.seed(SEED)

    # Leader state: index 0 = Red, index 1 = Gold
    X  = np.array([dataR.qi[0], dataG.qi[0]])
    Y  = np.array([dataR.qi[1], dataG.qi[1]])
    TH = np.array([dataR.qi[2], dataG.qi[2]])
    wayPts   = [dataR.wayPts, dataG.wayPts]
    trackers = [WaypointTracker(opts), WaypointTracker(opts)]
    idP  = [1, 1]
    done = [False, False]
    goalMin     = [np.inf, np.inf]
    stoppedFlag = [False, False]     # frozen this step?
    stopCount   = [0, 0]             # how many steps each spent frozen

    # Squadrons: deterministic formation spawn. Every wingman starts in its
    # slot, outside SLOT_MIN_R of its captain, on its captain's heading.
    spawnSpeed = float(np.clip(opts.v, opts.minSpeed, opts.maxSpeed))
    if SPAWN_MODE == 'formation':
        (fanX, fanY, fanVx, fanVy, squad,
         slotA, slotL) = spawnFormation(X, Y, TH, N_FAN,
                                        spacing=SLOT_SPACING, rows=SLOT_ROWS,
                                        minRadius=SLOT_MIN_R,
                                        speed=spawnSpeed,
                                        jitter=SPAWN_JITTER)
    else:
        fanX, fanY, fanVx, fanVy, squad = spawnRandomBehind(
            X, Y, TH, N_FAN, minRadius=SPAWN_MIN_R, maxRadius=SPAWN_MAX_R,
            halfAngle=SPAWN_ARC, minSep=SPAWN_MIN_SEP, speed=spawnSpeed)
        # the slot ATTRACTORS are the same either way -- only where the
        # wingmen begin differs, so the formation has to assemble itself
        sA, sL = formationSlots(N_FAN, SLOT_SPACING, SLOT_ROWS, SLOT_MIN_R,
                               shape=SLOT_SHAPE, sweep=SLOT_SWEEP)
        slotA, slotL = np.tile(sA, 2), np.tile(sL, 2)

    # ---- the global ordering, built once ----
    capKeys, fanKeys = buildRanks(squad, nLeaders=2, squadRank=SQUAD_RANK)

    dt, v = opts.dt, opts.v
    log = {'x': [[], []], 'y': [[], []], 'fanX': [], 'fanY': [],
           'fanGate': []}

    t = 0.0
    tMax = opts.tMax(dataR.refLen)
    while t <= tMax and not all(done):

        # ------------------------- LEADERS -------------------------
        # Snapshot the freeze flags BEFORE anyone decides. The old code read
        # stoppedFlag[other] while the same list was being written inside the
        # loop, so leader 0 saw leader 1's flag from the PREVIOUS step while
        # leader 1 saw leader 0's flag from THIS step. Loop index silently
        # acted as the priority rule. Now every leader decides from the same
        # snapshot and the ranking below is the only tiebreak.
        prevStopped = list(stoppedFlag)
        newStopped = [False] * 2
        newLead = []
        for k in range(2):
            if done[k]:
                newLead.append((X[k], Y[k], TH[k]))
                continue

            wp = wayPts[k]
            idP[k] = trackers[k].advance(X[k], Y[k], idP[k], wp)
            wpX, wpY = wp[idP[k], 0], wp[idP[k], 1]
            dGoal = np.hypot(wpX - X[k], wpY - Y[k])

            # done at closest approach to the last waypoint
            if idP[k] == wp.shape[0] - 1:
                goalMin[k] = min(goalMin[k], dGoal)
                if goalMin[k] < opts.passWindow and dGoal > goalMin[k] + opts.hysteresis:
                    done[k] = True
                    newLead.append((X[k], Y[k], TH[k]))
                    continue

            # --- term 1: follow my path ---
            segA = wp[max(idP[k] - 1, 0)]
            segB = wp[idP[k]]
            myGain = LEADER_FACTOR_RED if k == 0 else LEADER_FACTOR_GOLD
            vx, vy = guidance(X[k], Y[k], segA, segB, wpX, wpY, myGain)

            # --- term 2: dodge the other leader (leaders ignore fanboids) ---
            other = 1 - k
            myPR = PR_RED if k == 0 else PR_GOLD
            sx, sy = leaderRepulsion(X[k], Y[k],
                                np.array([X[other]]), np.array([Y[other]]),
                                opts, range_=myPR)
            vx += sx
            vy += sy

            # --- bang-bang Dubins steering toward the summed vector ---
            vTh = falconSteering(TH[k], vx, vy, opts)

            distOther = np.hypot(X[other] - X[k], Y[other] - Y[k])

            # DEADLOCK GUARD, priority-based (COLREGs Rules 16/17).
            mustStop = False
            if USE_INFEASIBILITY_STOP and distOther < STOP_RANGE:
                # An already-frozen leader is a stationary obstacle, not one
                # closing at speed v. Telling canAvoid() that is what lets the
                # stand-on leader steer past instead of freezing in sympathy.
                vOther = 0.0 if prevStopped[other] else v
                trapped = not canAvoid(X[k], Y[k], TH[k],
                                       X[other], Y[other], TH[other], opts,
                                       lookahead=STOP_LOOKAHEAD,
                                       margin=STOP_MARGIN, vOther=vOther)
                mustStop = yieldDecision(k, other, trapped,
                                         priority=LEADER_PRIORITY,
                                         prevStopped=prevStopped,
                                         useRule17b=USE_RULE_17B)

            if mustStop:
                newLead.append((X[k], Y[k], TH[k]))          # hold position
                newStopped[k] = True
            else:
                newLead.append((X[k] + v * np.cos(TH[k]) * dt,
                                Y[k] + v * np.sin(TH[k]) * dt,
                                TH[k] + vTh * dt))

        # ------------------------- FANBOIDS -------------------------
        vxDes, vyDes = flockStep(fanX, fanY, fanVx, fanVy, squad,
                                 X, Y, TH, opts,
                                 sepOnlyOnEncounter=SEP_ONLY_ON_ENCOUNTER,
                                 encounterRange=ENCOUNTER_RANGE,
                                 slotAlong=slotA if USE_SLOTS else None,
                                 slotLateral=slotL if USE_SLOTS else None,
                                 swirl=FAN_SWIRL,
                                 swirlHeadOnOnly=FAN_SWIRL_HEADON_ONLY)

        # wingman vs captain: lateral exit from the lane, at speed.
        if USE_CORRIDOR:
            cgx, cgy = corridorClear(fanX, fanY, X, Y, TH,
                                     range_=CORR_RANGE, halfWidth=CORR_HALF,
                                     gain=CORR_GAIN)
            vxDes = vxDes + cgx * dt
            vyDes = vyDes + cgy * dt

        fanVx, fanVy, _ = ackermannClamp(fanVx, fanVy, vxDes, vyDes, opts)

        # wingman vs wingman: the lower-ranked one of each pair brakes.
        if USE_FAN_BRAKE and USE_GLOBAL_PRIORITY:
            gate = fanBrakeRanked(fanX, fanY, fanVx, fanVy, fanKeys,
                                  brakeRange=BRAKE_RANGE,
                                  brakeStop=BRAKE_STOP, floor=BRAKE_FLOOR)
        else:
            gate = np.ones(fanX.size)

        fanX = fanX + fanVx * gate * dt
        fanY = fanY + fanVy * gate * dt

        # ------------------------- commit + log -------------------------
        stoppedFlag = newStopped
        for k in range(2):
            X[k], Y[k], TH[k] = newLead[k]
            if stoppedFlag[k]:
                stopCount[k] += 1
            log['x'][k].append(X[k])
            log['y'][k].append(Y[k])
        log['fanX'].append(fanX.copy())
        log['fanY'].append(fanY.copy())
        log['fanGate'].append(gate.copy())
        t += dt

    # ------------------------- report -------------------------
    xR, yR = np.array(log['x'][0]), np.array(log['y'][0])
    xG, yG = np.array(log['x'][1]), np.array(log['y'][1])
    dLeaders = np.hypot(xR - xG, yR - yG)

    print('guidance: %s | leader repulsion: %s | fanboid rule: stock Reynolds'
          % (GUIDANCE, 'Khatib' if SMOOTH_REPULSION else 'flat SAC'))
    print('closest leader-leader approach %.3f  (collision radius %.2f)'
          % (dLeaders.min(), opts.collisionRadius))
    print('finished: Red %s, Gold %s' % (done[0], done[1]))
    if USE_INFEASIBILITY_STOP:
        print('infeasibility stops: Red %d steps, Gold %d steps'
              % (stopCount[0], stopCount[1]))

    # Path-tracking quality: cross-track error against each reference line.
    # This is the number the guidance law is supposed to improve.
    for k, (name, xs, ys, d) in enumerate([('Red', xR, yR, dataR),
                                           ('Gold', xG, yG, dataG)]):
        e = crossTrackError(xs, ys, d.refPath)
        print('  %-5s cross-track: peak %.2f, RMS %.2f' % (name, np.abs(e).max(),
                                                           np.sqrt(np.mean(e ** 2))))

    # collision episodes: contiguous stretches below collisionRadius
    below = dLeaders < opts.collisionRadius
    circles = []
    inRun = False
    for tt in range(len(dLeaders)):
        if below[tt] and not inRun:
            circles.append([tt, tt]); inRun = True
        elif below[tt]:
            circles[-1][1] = tt
        elif inRun:
            inRun = False

    results = {'log': log, 'squad': squad, 'dataR': dataR, 'dataG': dataG,
               'opts': opts, 'dLeaders': dLeaders, 'circles': circles,
               'xR': xR, 'yR': yR, 'xG': xG, 'yG': yG,
               'stopCount': stopCount}
    if not plot:
        return results

    # ------------------------- plot -------------------------
    plt.figure(figsize=(9, 8))
    for k, (c, name, d) in enumerate([(RED, 'Red Leader', dataR),
                                      (GOLD, 'Gold Leader', dataG)]):
        plt.plot(d.refPath[:, 0], d.refPath[:, 1], ':', color=c, alpha=0.6,
                 label='%s ref path' % name)
        plt.plot(log['x'][k], log['y'][k], '-', color=c, linewidth=2,
                 label='%s actual' % name)

    fx = np.array(log['fanX'])
    fy = np.array(log['fanY'])
    for i in range(2 * N_FAN):
        c = RED if squad[i] == 0 else GOLD
        plt.plot(fx[:, i], fy[:, i], '-', color=c, linewidth=0.5, alpha=0.35,
                 zorder=1)

    for n_, (t0, t1) in enumerate(circles):
        tw = t0 + int(np.argmin(dLeaders[t0:t1 + 1]))
        cx = 0.5 * (xR[tw] + xG[tw])
        cy = 0.5 * (yR[tw] + yG[tw])
        plt.gca().add_patch(plt.Circle((cx, cy), CIRCLE_R, fill=False,
                                       color='red', lw=2.0, zorder=6,
                                       label='COLLISION' if n_ == 0 else None))

    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Two squads, %s approach (seed %d, %d fans each, guidance=%s)'
              % (GEOMETRY, SEED, N_FAN, GUIDANCE))
    plt.legend(fontsize=8)
    plt.show()
    return results


# ---------------------------------------------------------------------
#  Extras
# ---------------------------------------------------------------------

def collisions(radius=None):
    """Run the sim, then report EVERY collision: leader-leader, leader-fan,
    fan-fan (same squad and cross squad)."""
    from grainframe.collisions import agents_from_arrays, collision_report

    r = run(plot=False)
    log, squad = r['log'], r['squad']
    fx = np.array(log['fanX'])
    fy = np.array(log['fanY'])

    leaders = [(r['xR'], r['yR']), (r['xG'], r['yG'])]
    fans = [(fx[:, i], fy[:, i]) for i in range(fx.shape[1])]

    agents = agents_from_arrays(leaders, fans, fanTeams=squad,
                                leaderNames=['Red Leader', 'Gold Leader'],
                                teamNames={0: 'Red', 1: 'Gold'})
    if radius is None:
        radius = r['opts'].collisionRadius
    return collision_report(agents, radius, dt=r['opts'].dt)


_CACHE = {}


def animate(frameSkip=20, tail=400, save=None, fps=60, reuse=True):
    """Play back the sim.

    The simulation is fully computed BEFORE playback starts -- but the frames
    are not. FuncAnimation calls update() live and matplotlib redraws each
    frame on demand, so playback speed is bounded by rendering, not by the
    sim. Three knobs, in order of effect:

      blit      (in animate_tracks) 19.5 -> 3.0 ms/frame. Biggest single win.
      fps       sets interval = 1000//fps, a hard floor on frame time. At the
                old fps=20 that was a 50 ms pause per frame no matter how fast
                the render was.
      frameSkip how many sim steps each frame advances. 8391 steps at the old
                frameSkip=25 is 336 frames; at 60 it is 140, and you cannot
                see the difference at playback speed.

    reuse: keep the finished run in memory so replaying does not re-simulate.
    Set False after changing any parameter, or the old run will be replayed.
    """
    from grainframe.animate import tracks_from_arrays, animate_tracks

    if reuse and 'r' in _CACHE:
        r = _CACHE['r']
    else:
        r = run(plot=False)
        _CACHE['r'] = r
    log, squad = r['log'], r['squad']
    fx = np.array(log['fanX'])
    fy = np.array(log['fanY'])

    leaders = [(r['xR'], r['yR']), (r['xG'], r['yG'])]
    fans = [(fx[:, i], fy[:, i]) for i in range(fx.shape[1])]
    fanColors = [RED if squad[i] == 0 else GOLD for i in range(fx.shape[1])]

    dLeaders = r['dLeaders']
    dt = r['opts'].dt
    subtitle = lambda t: 't = %.1f s   leader gap = %.2f' % (t * dt, dLeaders[t])
    refs = [(r['dataR'].refPath[:, 0], r['dataR'].refPath[:, 1], RED),
            (r['dataG'].refPath[:, 0], r['dataG'].refPath[:, 1], GOLD)]

    tracks = tracks_from_arrays(leaders, fans, [RED, GOLD], fanColors,
                                leaderLabels=['Red Leader', 'Gold Leader'])
    return animate_tracks(tracks, dt=dt, frameSkip=frameSkip, tail=tail,
                          title='Two squads, %s (seed %d)' % (GEOMETRY, SEED),
                          subtitleFn=subtitle, refPaths=refs, save=save,
                          fps=fps)


def compare():
    """Side by side: old waypoint chasing vs the vector field. For the paper."""
    global GUIDANCE, SMOOTH_REPULSION
    saveG, saveS = GUIDANCE, SMOOTH_REPULSION

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    settings = [('waypoint', False, 'BEFORE: chase next waypoint, flat repulsion'),
                ('vfield',   True,  'AFTER: vector field, Khatib repulsion')]
    for ax, (g, sm, label) in zip(axes, settings):
        GUIDANCE, SMOOTH_REPULSION = g, sm
        r = run(plot=False)
        ax.axhline(0, color='k', lw=0.5, ls=':')
        fx, fy = np.array(r['log']['fanX']), np.array(r['log']['fanY'])
        for i in range(fx.shape[1]):
            ax.plot(fx[:, i], fy[:, i], lw=0.4, alpha=0.3,
                    color=RED if r['squad'][i] == 0 else GOLD)
        ax.plot(r['xR'], r['yR'], color=RED,  lw=2, label='Red')
        ax.plot(r['xG'], r['yG'], color=GOLD, lw=2, label='Gold')
        ax.set_title('%s   |  closest gap %.2f' % (label, r['dLeaders'].min()),
                     fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc='upper left')

    GUIDANCE, SMOOTH_REPULSION = saveG, saveS
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'plot'
    if mode.startswith('anim'):
        animate()
    elif mode.startswith('coll'):
        collisions()
    elif mode.startswith('comp'):
        compare()
    else:
        run()