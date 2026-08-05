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
from grainframe.dynamics import (boidsRules, falconSteering, ackermannClamp,
                                 WaypointTracker)


# =====================================================================
#  PARAMETERS: edit me
# =====================================================================

SEED  = 8            # change for a different random squadron layout
N_FAN = 6            # fanboids per squadron

GEOMETRY = 'headon'  # 'headon' = straight at each other on the same line
                     # 'cross'  = the X crossing at the origin

HALF = 40.0          # how far from the middle each leader starts
LANE_OFFSET = 0.0    # shove Gold's lane sideways. 0 = perfectly symmetric.

# Gain overrides from config.py. None = keep the stock value.
OVERRIDES = {
    # 'SF': 5.0,              # fanboid separation strength
    # 'CF': 0.8,              # fanboid cohesion
    # 'AF': 2.0,              # fanboid alignment
    # 'fanLeaderFactor': 1.0, # fanboid pull toward own leader
    # 'R': 5.0,               # leader turn radius (bigger = clumsier)
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
PR_GOLD = 4.0

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

# ---------------------------------------------------------------------
#  FANBOID SAFETY LAYER  (the swarm analogue of the leader stop, Sec 4.1)
# ---------------------------------------------------------------------
# A fanboid is an Ackermann point mass with turn radius Rboid and a speed
# floor. When Rboid is large compared to the collision radius, STEERING alone
# cannot resolve a close encounter -- no gain on the separation rule fixes a
# kinematic impossibility, it just saturates the heading clamp. BRAKING is the
# one authority that does not depend on turn radius, which is exactly the
# argument the paper already makes for the leaders in Section 4.1.
#
# This runs AFTER boids + Ackermann, so it never touches the Reynolds rules:
# separation/cohesion/alignment produce the velocity, this only decides how
# much of it is executed this step.
# Fans give leaders a WIDER berth than they give each other. A leader is a
# constant-speed Dubins car that never yields (falconSeesFanboids = False), so
# a fan that only reacts at the peer protected range has no runway. Braking is
# also the wrong move against an oncoming leader -- you become a stationary
# target -- so the brake below ignores leaders and only meters peer traffic.
FAN_LEADER_PR = 9.0    # fan's separation range against the OTHER squad's leader.
                       # Against its OWN leader a fan keeps the stock PR, or the
                       # squadron just gets blown apart and stops being a squadron.

# The leaders are the residual. A leader is a constant-speed Dubins car with
# falconSeesFanboids = False and a repulsion term that only ever looks at the
# OTHER LEADER, so it cannot yield to a fanboid at all -- the fan must do 100%
# of the avoidance with turn radius Rboid against a collision radius of 1.0.
# Two ways out, both switchable so the paper can report the cost of each:
#   FAN_OWN_PR      -- fan-side only. Widen the fan's berth around its OWN
#                      leader. Costs squadron cohesion, leaves leaders exact.
#   LEADER_FAN_PR   -- leader-side. Give each leader a Khatib term against
#                      fanboids. Costs cross-track error, i.e. it perturbs the
#                      guidance result the paper is actually measuring.
FAN_OWN_PR    = 5.0    # None -> stock opts.PR
LEADER_FAN_PR = 0.0    # 0 -> leaders ignore fanboids entirely.
                       # KEPT AT 0 BY DESIGN: a captain yields only to another
                       # captain. Fanboids yield to captains, never the reverse.
LEADER_FAN_GAIN = 8.0  # Khatib eta scale for the above

# CORRIDOR CLEARING. Braking against a captain is measurably self-defeating --
# a stopped fanboid in front of a constant-speed car that never yields is a
# stationary target. This term instead pushes the fan SIDEWAYS out of the
# captain's lane at full speed: get off the road rather than stop in it.
# ---------------------------------------------------------------------
#  RULES OF THE ROAD  (the tiebreaker -- config.capSwirl, never wired in here)
# ---------------------------------------------------------------------
# A symmetric head-on is unsolvable by repulsion alone. Both agents lie on the
# mirror line, the repulsion points ALONG that line, and no gain breaks the
# tie -- they just push each other backwards and lock. The fix is handedness:
# everyone also veers consistently to one side.
#
# Implemented as a 90 degree CCW rotation of the repulsion vector in the WORLD
# frame. Check it on the head-on case: Red at the origin heading +x sees Gold
# at +d, so his repulsion is (-1,0) and rot90 gives (0,-1) -- he goes to his
# own right. Gold's repulsion is (+1,0) and rot90 gives (0,+1) -- also his own
# right. A starboard-to-starboard pass, which is what you want.
SWIRL_LEADER = 0.8     # 0 = off (the old symmetric deadlock)
SWIRL_FAN    = 0.8

# A fully frozen pair is its own deadlock: two fans nose to nose both brake to
# zero and neither ever moves again. The floor keeps everyone crawling so the
# geometry can still resolve.
BRAKE_FLOOR = 0.15

USE_CORRIDOR = True
CORR_RANGE = 12.0    # look this far up the captain's track
CORR_HALF  = 4.5     # lane half-width to clear
CORR_GAIN  = 60.0

USE_FAN_BRAKE = True
BRAKE_RANGE   = 3.0    # start easing off inside this gap
BRAKE_STOP    = 1.4    # fully frozen at this gap (> collisionRadius = 1.0)

# ---------------------------------------------------------------------
#  SPAWN: deterministic formation, laid out in each leader's body frame
# ---------------------------------------------------------------------
# Random spawn was the wrong model. Boids started at random headings and random
# speeds with a turn radius of Rboid, so the opening seconds were spent
# untangling an arrangement nobody chose -- and occasionally starting already
# inside the collision radius. A squadron forms up BEFORE it flies, so the
# initial condition should be a formation, not a cloud.
#
# Slots are defined in the leader's body frame:  +x = the way he is pointing,
# +y = his left. That makes the layout identical for 'headon' and 'cross'.
#
#   'vee'      classic wedge trailing behind the leader, alternating sides
#   'echelon'  single diagonal line off one side
#   'column'   single file directly astern
#   'line'     abreast, one rank, leader centred
#   'grid'     rows astern, FORM_COLS wide
#   'ring'     evenly spaced all the way around him, at a random bearing
FORMATION     = 'ring'
FORM_SPACING  = 3.0    # gap between consecutive slots
FORM_STANDOFF = 3.0    # gap from the leader to the first slot
FORM_SWEEP    = 0.7    # vee/echelon: lateral offset per unit of trail
FORM_COLS     = 3      # 'grid' only
FORM_SIDE     = +1     # 'echelon' only: +1 = left, -1 = right

# Each squadron's pattern gets its own random rotation about its captain, so
# Red and Gold are NOT mirror images of each other. A symmetric setup is the
# worst possible test case: mirrored agents get shoved along the same mirror
# line and the tie never breaks, which is the same reason capSwirl exists for
# the leaders. Rotating each squad independently kills that artefact.
# Set to False for a fixed, fully deterministic layout.
FORM_RANDOM_ROT = True
FORM_JITTER   = 0.0    # optional gaussian noise on each slot. 0 = fully
                       # deterministic, and note that with 0 the whole sim
                       # stops depending on SEED at all -- if you want a seed
                       # sweep again, this is the knob that gives you one.
CIRCLE_R   = 3.0     # radius of the drawn COLLISION circles
# =====================================================================

RED, GOLD = 'crimson', 'goldenrod'


# ---------------------------------------------------------------------
#  Guidance law
# ---------------------------------------------------------------------

def guidance(x, y, segA, segB, wpX, wpY, gain):
    """Desired velocity vector for a leader at (x, y).

    segA -> segB is the path segment currently being tracked.
    wpX, wpY is the active waypoint (only used by the 'waypoint' mode).
    """
    if GUIDANCE == 'waypoint':
        return (wpX - x) * gain, (wpY - y) * gain

    # --- vector field ---
    seg = segB - segA
    L = np.linalg.norm(seg)
    if L < 1e-9:
        return (wpX - x) * gain, (wpY - y) * gain

    u = seg / L                              # along-track unit vector
    n = np.array([-u[1], u[0]])              # left normal
    e = (x - segA[0]) * n[0] + (y - segA[1]) * n[1]   # signed cross-track error

    chiPath = np.arctan2(u[1], u[0])
    # Far off the line (|e| large) atan saturates and chiD = chiPath -+ CHI_INF.
    # On the line (e = 0) atan is 0 and chiD = chiPath exactly.
    chiD = chiPath - CHI_INF * (2.0 / np.pi) * np.arctan(K_CROSS * e)

    return GUID_MAG * gain * np.cos(chiD), GUID_MAG * gain * np.sin(chiD)


# ---------------------------------------------------------------------
#  Repulsion
# ---------------------------------------------------------------------

def fanSeparation(xb, yb, xOthers, yOthers, opts, range_=None):
    """STOCK Reynolds/SAC separation -- used by the FANBOIDS. Do not change.

    This is part of the boids model being studied, so it stays exactly as it
    was: a flat push proportional to displacement, cut off at the protected
    range. Note this is the same family of object as the Khatib potential
    below -- both are repulsive artificial potential fields -- they just have
    different distance profiles. Keeping them separate means a change to the
    leader controller cannot silently alter the swarm results.
    """
    if range_ is None:
        range_ = opts.PR
    dx = xb - xOthers
    dy = yb - yOthers
    D = np.hypot(dx, dy)
    tooClose = (D > 1e-9) & (D <= range_)
    return (float(np.sum(dx[tooClose]) * opts.SF),
            float(np.sum(dy[tooClose]) * opts.SF))


def swirl(rx, ry, k):
    """Add a consistent handedness to a repulsion vector.

    Returns r + k * rot90(r). Scales with the repulsion itself, so it vanishes
    the moment the repulsion does -- it can never push anybody around when
    there is nothing to avoid.
    """
    if k == 0.0:
        return rx, ry
    return rx - k * ry, ry + k * rx


def leaderRepulsion(xb, yb, xOthers, yOthers, opts, range_=None):
    """Leader-vs-leader repulsion ONLY. Not used by the fanboids.

    Khatib (1986):  F = eta * (1/d - 1/range) / d^2, directed away.
    Zero at d = range, singular at d = 0. eta is scaled so that REP_GAIN is
    literally the force magnitude at d = range/2.
    """
    if range_ is None:
        range_ = opts.PR

    dx = xb - xOthers
    dy = yb - yOthers
    D = np.hypot(dx, dy)
    tooClose = (D > 1e-9) & (D <= range_)
    if not np.any(tooClose):
        return 0.0, 0.0

    if not SMOOTH_REPULSION:                      # legacy flat rule
        return (float(np.sum(dx[tooClose]) * opts.SF),
                float(np.sum(dy[tooClose]) * opts.SF))

    d = np.clip(D[tooClose], 0.25, None)          # clip so it stays finite
    eta = REP_GAIN * range_ ** 3 / 4.0
    w = eta * (1.0 / d - 1.0 / range_) / d ** 2   # magnitude
    ux, uy = dx[tooClose] / d, dy[tooClose] / d   # unit vector, away from them
    rx, ry = float(np.sum(w * ux)), float(np.sum(w * uy))
    return swirl(rx, ry, SWIRL_LEADER)


def canAvoid(x, y, th, xOther, yOther, thOther, opts):
    """Can this leader clear the other by turning as hard as he legally can?

    Roll both forward STOP_LOOKAHEAD steps: the other holds heading, I turn at
    my maximum rate (v/R) in whichever direction opens the gap. If even that
    best case stays closer than STOP_MARGIN, no feasible avoidance exists and
    the caller should STOP.

    Vectorised: the whole rollout is closed-form under a constant turn rate,
    so it is one cumsum instead of a Python loop. Same answer, ~100x faster,
    which matters because STOP_LOOKAHEAD has to be a real horizon (thousands
    of steps) to mean anything.
    """
    dt, v, R = opts.dt, opts.v, opts.R
    dpsiMax = (v / R) * dt                  # most I can rotate per step

    # Which way opens the gap? Sign of cross(heading, toward-other).
    toX, toY = xOther - x, yOther - y
    cross = np.cos(th) * toY - np.sin(th) * toX
    turn = -np.sign(cross) * dpsiMax        # turn AWAY

    n = np.arange(1, STOP_LOOKAHEAD + 1)
    thi = th + turn * n                     # my heading each step
    xi = x + np.cumsum(v * np.cos(thi) * dt)
    yi = y + np.cumsum(v * np.sin(thi) * dt)
    xo = xOther + v * np.cos(thOther) * dt * n   # he just keeps going straight
    yo = yOther + v * np.sin(thOther) * dt * n

    return float(np.min(np.hypot(xi - xo, yi - yo))) >= STOP_MARGIN


def fanBrake(fanX, fanY, fanVx, fanVy, X, Y, TH, opts):
    """Per-fanboid speed scale in [0, 1]: ease off when closing on somebody.

    Returns a multiplier applied to the DISPLACEMENT only. The velocity state
    itself is left alone so the heading stays well defined and the Ackermann
    clamp keeps working normally on the next step -- same trick the leaders
    use when they freeze.
    """
    n = fanX.size
    lvx, lvy = opts.v * np.cos(TH), opts.v * np.sin(TH)

    scale = np.ones(n)
    for i in range(n):
        # Peers AND both captains. Right of way is one-directional by design:
        # a fanboid yields to any captain, a captain yields only to another
        # captain (LEADER_FAN_PR = 0). Braking in front of a car that never
        # yields is a genuine risk, which is why FAN_LEADER_PR also gives the
        # fan a wide STEERING berth -- the brake is the last resort, not the
        # first move.
        px  = np.append(fanX,  X)
        py  = np.append(fanY,  Y)
        pvx = np.append(fanVx, lvx)
        pvy = np.append(fanVy, lvy)
        dx, dy = px - fanX[i], py - fanY[i]
        d = np.hypot(dx, dy)
        near = (d > 1e-9) & (d < BRAKE_RANGE)
        if not np.any(near):
            continue
        # range rate along the line of sight; negative = closing
        rdot = ((pvx[near] - fanVx[i]) * dx[near] +
                (pvy[near] - fanVy[i]) * dy[near]) / d[near]
        closing = rdot < 0.0
        if not np.any(closing):
            continue
        dMin = float(d[near][closing].min())
        s = (dMin - BRAKE_STOP) / (BRAKE_RANGE - BRAKE_STOP)
        scale[i] = min(scale[i], float(np.clip(s, BRAKE_FLOOR, 1.0)))
    return scale


def corridorClear(fx, fy, X, Y, TH, opts):
    """Sideways push out of a captain's lane.

    Decompose the fan's offset from the captain into along-track (a) and
    cross-track (c) in the captain's frame. If the fan is AHEAD (a > 0), inside
    CORR_RANGE and inside the lane (|c| < CORR_HALF), push along the captain's
    left/right normal, away from the centreline, hardest when dead centre.
    """
    gx = gy = 0.0
    for k in range(2):
        ux, uy = np.cos(TH[k]), np.sin(TH[k])
        nx, ny = -uy, ux
        dx, dy = fx - X[k], fy - Y[k]
        a = dx * ux + dy * uy                    # along-track
        c = dx * nx + dy * ny                    # cross-track
        if a <= 0.0 or a > CORR_RANGE or abs(c) >= CORR_HALF:
            continue
        side = 1.0 if c >= 0 else -1.0           # exit the near side
        if abs(c) < 1e-6:
            side = 1.0                           # dead centre: consistent handedness
        w = CORR_GAIN * (1.0 - abs(c) / CORR_HALF) * (1.0 - a / CORR_RANGE)
        gx += w * side * nx
        gy += w * side * ny
    return gx, gy


def formationSlots(n):
    """Slot offsets (along, lateral) in the leader's body frame.

    along < 0 is behind him, lateral > 0 is to his left.
    """
    if FORMATION == 'ring':
        # Radius is whichever is larger: the requested standoff, or the radius
        # at which n evenly spaced boids sit FORM_SPACING apart along the arc.
        # That way tightening the ring can never spawn them on top of each
        # other -- it just pushes the ring outward instead.
        rMin = n * FORM_SPACING / (2.0 * np.pi)
        rad = max(FORM_STANDOFF, rMin)
        ang = 2.0 * np.pi * np.arange(n) / n
        return [(rad * np.cos(a), rad * np.sin(a)) for a in ang]

    slots = []
    for j in range(n):
        if FORMATION == 'vee':
            rank = j // 2 + 1
            side = 1.0 if j % 2 == 0 else -1.0
            d = FORM_STANDOFF + (rank - 1) * FORM_SPACING
            slots.append((-d, side * d * FORM_SWEEP))

        elif FORMATION == 'echelon':
            d = FORM_STANDOFF + j * FORM_SPACING
            slots.append((-d, FORM_SIDE * d * FORM_SWEEP))

        elif FORMATION == 'column':
            slots.append((-(FORM_STANDOFF + j * FORM_SPACING), 0.0))

        elif FORMATION == 'line':
            rank = j // 2 + 1
            side = 1.0 if j % 2 == 0 else -1.0
            slots.append((-FORM_STANDOFF, side * rank * FORM_SPACING))

        elif FORMATION == 'grid':
            row, col = divmod(j, FORM_COLS)
            # centre each row on the leader's track
            lat = (col - (FORM_COLS - 1) / 2.0) * FORM_SPACING
            slots.append((-(FORM_STANDOFF + row * FORM_SPACING), lat))

        else:
            raise ValueError('unknown FORMATION %r' % FORMATION)
    return slots


def spawnSquadrons(X, Y, TH, opts):
    """Place each squadron in formation behind its leader, matched to his
    heading and his speed.

    Starting everyone aligned and at the leader's speed removes the opening
    transient entirely: there is nothing to untangle, so whatever the collision
    report shows afterwards is the encounter, not the initial condition.
    """
    slots = formationSlots(N_FAN)
    fanX, fanY, fanVx, fanVy, squad = [], [], [], [], []

    for k in range(2):
        ux, uy = np.cos(TH[k]), np.sin(TH[k])     # forward
        nx, ny = -uy, ux                          # left

        # Independent random spin of this squadron's pattern about its captain.
        # Note this only moves WHERE they stand: every fan still launches on
        # the captain's heading at the captain's speed, so there is no opening
        # transient regardless of how the pattern lands.
        phi = np.random.uniform(0, 2 * np.pi) if FORM_RANDOM_ROT else 0.0
        cphi, sphi = np.cos(phi), np.sin(phi)

        for (along0, lat0) in slots:
            along = along0 * cphi - lat0 * sphi
            lat   = along0 * sphi + lat0 * cphi
            px = X[k] + along * ux + lat * nx
            py = Y[k] + along * uy + lat * ny
            if FORM_JITTER > 0.0:
                px += np.random.normal(0, FORM_JITTER)
                py += np.random.normal(0, FORM_JITTER)
            fanX.append(px)
            fanY.append(py)

        # everyone starts pointed the way the leader is pointed, at his speed
        spd = float(np.clip(opts.v, opts.minSpeed, opts.maxSpeed))
        fanVx.extend([spd * ux] * N_FAN)
        fanVy.extend([spd * uy] * N_FAN)
        squad.append(np.full(N_FAN, k))

    fanX = np.array(fanX)
    fanY = np.array(fanY)

    # sanity: report the tightest pair at t=0 so a bad spacing is obvious
    px = np.concatenate([fanX, X])
    py = np.concatenate([fanY, Y])
    gap = np.inf
    for i in range(px.size):
        for j in range(i + 1, px.size):
            gap = min(gap, float(np.hypot(px[i] - px[j], py[i] - py[j])))
    tag = '  <-- TOO TIGHT' if gap < opts.collisionRadius else ''
    print('spawn: %s, %d per squad, random rot %s, tightest pair at t=0 = %.2f%s'
          % (FORMATION, N_FAN, 'on' if FORM_RANDOM_ROT else 'off', gap, tag))

    return (fanX, fanY, np.array(fanVx), np.array(fanVy),
            np.concatenate(squad))


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

    # Squadrons: N_FAN boids clumped near each leader's start
    fanX, fanY, fanVx, fanVy, squad = spawnSquadrons(X, Y, TH, opts)

    dt, v = opts.dt, opts.v
    log = {'x': [[], []], 'y': [[], []], 'fanX': [], 'fanY': []}

    t = 0.0
    tMax = opts.tMax(dataR.refLen)
    while t <= tMax and not all(done):

        # ------------------------- LEADERS -------------------------
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

            # --- optional term 3: dodge fanboids (off by default) ---
            if LEADER_FAN_PR > 0.0:
                fdx = X[k] - fanX
                fdy = Y[k] - fanY
                fD = np.hypot(fdx, fdy)
                m = (fD > 1e-9) & (fD <= LEADER_FAN_PR)
                if np.any(m):
                    d = np.clip(fD[m], 0.25, None)
                    eta = LEADER_FAN_GAIN * LEADER_FAN_PR ** 3 / 4.0
                    w = eta * (1.0 / d - 1.0 / LEADER_FAN_PR) / d ** 2
                    vx += float(np.sum(w * fdx[m] / d))
                    vy += float(np.sum(w * fdy[m] / d))

            # --- bang-bang Dubins steering toward the summed vector ---
            vTh = falconSteering(TH[k], vx, vy, opts)

            distOther = np.hypot(X[other] - X[k], Y[other] - Y[k])

            # DEADLOCK GUARD: if the other leader is already frozen he is not
            # closing on me, so freezing too would lock us both forever with
            # the gap constant and canAvoid() never able to recover. Only one
            # of us needs to yield.
            mustStop = (USE_INFEASIBILITY_STOP
                        and distOther < STOP_RANGE
                        and not stoppedFlag[other]
                        and not canAvoid(X[k], Y[k], TH[k],
                                         X[other], Y[other], TH[other], opts))

            if mustStop:
                newLead.append((X[k], Y[k], TH[k]))          # hold position
                stoppedFlag[k] = True
            else:
                newLead.append((X[k] + v * np.cos(TH[k]) * dt,
                                Y[k] + v * np.sin(TH[k]) * dt,
                                TH[k] + vTh * dt))
                stoppedFlag[k] = False

        # ------------------------- FANBOIDS -------------------------
        vxDes = np.empty(2 * N_FAN)
        vyDes = np.empty(2 * N_FAN)
        for i in range(2 * N_FAN):
            k = squad[i]                       # my squadron / my leader
            mine = (squad == k)

            # cohesion + alignment with my OWN squadron (plus my leader)
            sameX  = np.append(fanX[mine],  X[k])
            sameY  = np.append(fanY[mine],  Y[k])
            sameVx = np.append(fanVx[mine], v * np.cos(TH[k]))
            sameVy = np.append(fanVy[mine], v * np.sin(TH[k]))
            (_, _, aX, aY, cX, cY) = boidsRules(
                fanX[i], fanY[i], fanVx[i], fanVy[i],
                sameX, sameY, sameVx, sameVy, opts)

            # separation against other fanboids at the stock protected range,
            # and against the two leaders at a wider one (FAN_LEADER_PR).
            other = 1 - k
            sX, sY = fanSeparation(fanX[i], fanY[i], fanX, fanY, opts)
            oX, oY = fanSeparation(fanX[i], fanY[i], X[k:k+1], Y[k:k+1], opts,
                                   range_=FAN_OWN_PR)
            wX, wY = fanSeparation(fanX[i], fanY[i],
                                   X[other:other+1], Y[other:other+1], opts,
                                   range_=FAN_LEADER_PR)
            sX += oX + wX
            sY += oY + wY
            # handedness applied to the SUM, so the Reynolds rule above is
            # still bit-for-bit stock -- this is an added term, not an edit
            sX, sY = swirl(sX, sY, SWIRL_FAN)

            if USE_CORRIDOR:
                cx, cy = corridorClear(fanX[i], fanY[i], X, Y, TH, opts)
                sX += cx
                sY += cy

            # pull toward MY leader only
            lX = (X[k] - fanX[i]) * opts.fanLeaderFactor
            lY = (Y[k] - fanY[i]) * opts.fanLeaderFactor

            vxDes[i] = fanVx[i] + (sX + aX + cX + lX) * dt
            vyDes[i] = fanVy[i] + (sY + aY + cY + lY) * dt

        fanVx, fanVy, _ = ackermannClamp(fanVx, fanVy, vxDes, vyDes, opts)
        gate = (fanBrake(fanX, fanY, fanVx, fanVy, X, Y, TH, opts)
                if USE_FAN_BRAKE else 1.0)
        fanX = fanX + fanVx * gate * dt
        fanY = fanY + fanVy * gate * dt

        # ------------------------- commit + log -------------------------
        for k in range(2):
            X[k], Y[k], TH[k] = newLead[k]
            if stoppedFlag[k]:
                stopCount[k] += 1
            log['x'][k].append(X[k])
            log['y'][k].append(Y[k])
        log['fanX'].append(fanX.copy())
        log['fanY'].append(fanY.copy())
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
        a = d.refPath[0]
        u = (d.refPath[-1] - a) / np.linalg.norm(d.refPath[-1] - a)
        n = np.array([-u[1], u[0]])
        e = (xs - a[0]) * n[0] + (ys - a[1]) * n[1]
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


def animate(frameSkip=25, tail=400, save=None):
    """Run the sim, then play it back using the shared animation utility."""
    from grainframe.animate import tracks_from_arrays, animate_tracks

    r = run(plot=False)
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
                          subtitleFn=subtitle, refPaths=refs, save=save)


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