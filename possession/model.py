"""Who has the ball? A three-state HMM (home, away, dead) decoded jointly with Viterbi.

What the challenge asks for, restated as a model:

  * while the ball is alive one team always has possession, and it keeps it through passes,
    loose balls and contested phases until the other team gains CONTROL;
  * a touch, a deflection or a lost aerial duel is not control;
  * a stoppage ends a possession, and the team that restarts owns the next one.

So each frame contributes evidence and each change of state costs a penalty:

  alive vs dead   the tempered log-likelihood ratio from deadball.py
  home vs away    contact: a player of one side within CONTACT_RADIUS of the ball with the
                  nearest opponent clearly further away. The evidence is zero-sum between home
                  and away (+k/2 for the side in contact, -k/2 for the other), so it never
                  moves the alive/dead decision, only which alive state wins.

A change of possession therefore happens only when the new side's contact evidence outweighs
the switching penalty, i.e. a few frames of clear contact (the "controlled possession" of
CHALLENGE.md), and a one-frame deflection cannot do it. Between contacts there is no evidence
and the state persists, which is the "possession stays with the last team" rule.

Every constant below is a physical or football prior, written down before decoding, not fitted
to any labels.
"""

import numpy as np

import hmm
from deadball import P_RESTART, P_STOP, TEMPERATURE
from features import FPS, run_lengths

HOME, AWAY, DEAD = 0, 1, 2
LETTERS = np.array(["H", "A", "D"])

# A controlling player has the ball at his feet: within a metre, plus tracking error.
CONTACT_RADIUS = 1.5
# ...and nearer to it than any opponent by a clear margin, or it is a duel, not control.
CONTACT_MARGIN = 0.5
# A ball travelling faster than this is in flight: the nearest player is passing under it, not
# controlling it. Such frames still vote, at a fraction of the weight.
FLIGHT_SPEED = 12.0
FLIGHT_WEIGHT = 0.25
# Evidence per frame of clear contact, in nats.
CONTACT_WEIGHT = 1.0
# A possession lasts about a dozen seconds on average in top-flight football; the rest of the
# possession ends are stoppages, which carry their own penalty.
MEAN_POSSESSION_S = 12.0
P_SWITCH = 1.0 / (MEAN_POSSESSION_S * FPS)
# Minimum durations, in frames. A possession can be very short (one controlled touch before a
# tackle); a stoppage cannot.
MIN_POSSESSION = 10
MIN_DEAD = 3 * FPS
# Log-evidence that rules a state out. Finite on purpose: the duration decoder works on
# cumulative sums, where -inf - -inf would be NaN.
FORBIDDEN = -1e3


def contact_evidence(f, radius=CONTACT_RADIUS, margin=CONTACT_MARGIN):
    """Signed per-frame contact weight: > 0 home in control, < 0 away, 0 no evidence."""
    d_home, d_away = f["d_home"], f["d_away"]
    near = np.minimum(d_home, d_away)
    gap = np.abs(d_home - d_away)
    clear = f["vis"] & (near < radius) & (gap > margin)
    weight = np.where(np.nan_to_num(f["ball_speed"], nan=0.0) > FLIGHT_SPEED, FLIGHT_WEIGHT, 1.0)
    sign = np.where(d_home < d_away, 1.0, -1.0)
    return np.where(clear, sign * weight, 0.0)


# The learned proximity evidence (see fit_team_table). Bins have physical meaning: at the feet,
# within reach, a stride or two away, closing in, nearby, far; a contested ball vs a clear one;
# a ball on the ground vs driven vs struck.
DIST_EDGES = [1.0, 1.5, 2.5, 4.0, 7.0]
MARGIN_EDGES = [0.5, 1.5, 3.0]
BALL_EDGES = [10.0, 15.0]
# Consecutive frames of one player near the ball are one observation, not twenty-five, so each
# frame's log-odds is shrunk by 1 / (integrated autocorrelation time of the evidence's errors),
# measured on the match by correlation_time(). This value only seeds the first pass.
TEAM_TEMPERATURE = 0.2
MAX_AGREEMENT = 0.99
# Evidence nearer than this is what the correlation time is measured on; beyond it the table is
# close to zero anyway and would only dilute the estimate.
CORRELATION_RANGE = 7.0
MAX_LAG = 150


def proximity_codes(f):
    """Joint (distance, margin, ball speed) bin per frame with a visible ball; -1 without."""
    near = np.minimum(f["d_home"], f["d_away"])
    margin = np.abs(f["d_home"] - f["d_away"])
    speed = np.nan_to_num(f["ball_speed"], nan=0.0)
    code = ((np.digitize(near, DIST_EDGES) * (len(MARGIN_EDGES) + 1)
             + np.digitize(margin, MARGIN_EDGES)) * (len(BALL_EDGES) + 1)
            + np.digitize(speed, BALL_EDGES))
    return np.where(f["vis"] & np.isfinite(near), code, -1)


def fit_team_table(f, states, codes, temperature=TEAM_TEMPERATURE):
    """Per-bin log-odds that the nearest side is the side in possession, from a decoded match.

    Self-training: the decoded possession is the pseudo-label, the rule the competition allows
    ("pseudo-labels produced by an algorithm"). Agreement below one half is floored to no
    evidence rather than inverted: a ball struck past the nearest player says nothing about
    who has it, not that the other side does.
    """
    n_bins = (len(DIST_EDGES) + 1) * (len(MARGIN_EDGES) + 1) * (len(BALL_EDGES) + 1)
    alive = (states == HOME) | (states == AWAY)
    use = alive & (codes >= 0)
    nearest_home = f["d_home"] < f["d_away"]
    agree = nearest_home == (states == HOME)
    hits = np.bincount(codes[use & agree], minlength=n_bins) + 1.0
    total = np.bincount(codes[use], minlength=n_bins) + 2.0
    p = np.clip(hits / total, 0.5, MAX_AGREEMENT)
    return temperature * np.log(p / (1 - p))


def correlation_time(f, states, codes, max_lag=MAX_LAG):
    """Integrated autocorrelation time, in frames, of the nearest side disagreeing with the
    decoded possession.

    It is how long one tracking error persists (a duel, a swapped track, a ball occluded
    behind a body), and therefore how many consecutive frames amount to one independent piece
    of evidence. The sum is truncated at the first lag where the correlation drops below 0.05,
    the usual initial-positive-sequence cut.
    """
    alive = (states == HOME) | (states == AWAY)
    near = np.minimum(f["d_home"], f["d_away"])
    use = alive & (codes >= 0) & (near < CORRELATION_RANGE)
    disagree = ((f["d_home"] < f["d_away"]) != (states == HOME)).astype(np.float64)
    x = np.where(use, disagree - disagree[use].mean(), np.nan)
    total = 1.0
    for lag in range(1, max_lag + 1):
        a, b = x[:-lag], x[lag:]
        ok = np.isfinite(a) & np.isfinite(b)
        rho = np.sum(a[ok] * b[ok]) / np.sqrt(np.sum(a[ok] ** 2) * np.sum(b[ok] ** 2))
        if rho < 0.05:
            break
        total += 2.0 * rho
    return total


def team_evidence(f, table, codes):
    """Signed per-frame log-odds: > 0 home, < 0 away, 0 without a ball."""
    sign = np.where(f["d_home"] < f["d_away"], 1.0, -1.0)
    return np.where(codes >= 0, sign * table[np.maximum(codes, 0)], 0.0)


def transition_matrix(p_switch=P_SWITCH, p_stop=P_STOP, p_restart=P_RESTART):
    t = np.empty((3, 3))
    t[HOME] = [1 - p_switch - p_stop, p_switch, p_stop]
    t[AWAY] = [p_switch, 1 - p_switch - p_stop, p_stop]
    t[DEAD] = [p_restart / 2, p_restart / 2, 1 - p_restart]
    return np.log(t)


def emissions(dead_llr, contact, contact_weight=CONTACT_WEIGHT, temperature=TEMPERATURE):
    half = 0.5 * contact_weight * contact
    return np.stack([half, -half, temperature * dead_llr], axis=1)


def decode(f, dead, contact=None, contact_weight=CONTACT_WEIGHT, p_switch=P_SWITCH,
           min_possession=MIN_POSSESSION, min_dead=MIN_DEAD, priors=None):
    """Per-frame state codes (HOME/AWAY/DEAD) on graded frames, -1 elsewhere.

    dead is the dict deadball.fit returns: its log-ratio drives the alive/dead emission and its
    learned event bonuses make entering and leaving a stoppage cheap only at stop and restart
    events. priors is the dict build_priors returns: the Laws of the Game and the Smart
    Tagging set pieces, as hard constraints and transition bonuses.
    """
    if contact is None:
        contact = contact_evidence(f)
    priors = priors or {}
    states = np.full(f["n"], -1, dtype=np.int8)
    trans = transition_matrix(p_switch)
    emission = emissions(dead["llr"], contact, contact_weight)
    if "forced_dead" in priors:
        emission[priors["forced_dead"], HOME] = FORBIDDEN
        emission[priors["forced_dead"], AWAY] = FORBIDDEN
    if "forced_alive" in priors:
        emission[priors["forced_alive"], DEAD] = FORBIDDEN
    for start, stop in run_lengths(f["graded"]):
        bonus = np.zeros((stop - start, 3, 3))
        for team in (HOME, AWAY):
            bonus[:, team, DEAD] = dead["stop_bonus"][start:stop]
            bonus[:, DEAD, team] = dead["restart_bonus"][start:stop]
        if "restart_team" in priors:
            # P(dead -> side at t) = P(restart at t) * P(side | t). The restart term is the
            # learned event bonus; the side term turns the summed log-odds r into
            # log(2 sigma(+-r)), relative to the 50/50 split already in the transition matrix,
            # so the two sides' probabilities still sum to one and the evidence decides who
            # restarts, not whether.
            side = priors["restart_team"][start:stop]
            bonus[:, DEAD, HOME] += np.log(2.0) - np.logaddexp(0.0, -side)
            bonus[:, DEAD, AWAY] += np.log(2.0) - np.logaddexp(0.0, side)
        if "restart_block" in priors:
            blocked = priors["restart_block"][start:stop]
            bonus[blocked, DEAD, HOME] += FORBIDDEN
            bonus[blocked, DEAD, AWAY] += FORBIDDEN
        path = hmm.viterbi_duration(emission[start:stop], trans,
                                    [min_possession, min_possession, min_dead], bonus)
        states[start:stop] = path
    return states


def before_kickoff(f, events):
    """Graded frames before each half's tagged kick-off window: the ball cannot be in play.

    Law 8: every half starts with a kick-off. The Smart Tagging marks a ~5 s window containing
    it, so every graded frame of a half before that window is pre-match or halftime (warm-ups,
    players walking out), however lively it looks. Inside the window the decoder still chooses
    the restart frame itself.
    """
    forced = np.zeros(f["n"], dtype=bool)
    halves = run_lengths(f["graded"])
    for event in events:
        if event["code"].strip().lower() != "set pieces":
            continue
        if (event["tags"].get("Type") or "").strip().lower() != "kick off (start)":
            continue
        for start, stop in halves:
            if start <= event["start_frame"] < stop:
                forced[start:event["start_frame"]] = True
    return forced


# Law 9: a ball wholly over a line is out of play. The ATD ball is noisy at the edges and the
# tracker sometimes locks onto a spare ball beside the pitch, so an exit only counts if the ball
# is clearly outside, stays there, and got there continuously rather than by a jump.
OUT_MARGIN = 1.0
OUT_MIN_FRAMES = 5
MAX_STEP = 1.6              # metres per frame: 40 m/s, faster than any kicked ball
OUT_HOLD = FPS              # dead for at least this long after the ball is back in view
# A corner or goal kick is taken from a fixed place: until the ball has been seen there, the
# restart cannot have happened. Bounded, in case the ball is never seen again.
SPOT_CORNER = 4.0
SPOT_GOAL_AREA_X = 8.0
SPOT_GOAL_AREA_Y = 12.0
SPOT_WAIT_MAX = 60 * FPS


def ball_out_of_play(f, pitch):
    """Frames where the laws of the game say the ball is dead because it left the pitch."""
    length, width = pitch
    bx, by, vis = f["bx"], f["by"], f["vis"]
    outside = np.nan_to_num(f["outside"], nan=-99.0)
    step = np.full(f["n"], np.inf)
    step[1:] = np.hypot(np.diff(bx), np.diff(by))
    step = np.nan_to_num(step, nan=np.inf)
    forced = np.zeros(f["n"], dtype=bool)
    for start, stop in run_lengths(vis & (outside > OUT_MARGIN)):
        if stop - start < OUT_MIN_FRAMES or start == 0:
            continue
        if not vis[start - 1] or step[start] > MAX_STEP or step[start + 1:stop].max(initial=0) > MAX_STEP:
            continue                                  # appeared outside: a spare ball, not an exit
        end = min(f["n"], stop + OUT_HOLD)
        peak = start + int(np.argmax(outside[start:stop]))
        over_goal_line = min(bx[peak], length - bx[peak]) < min(by[peak], width - by[peak])
        if over_goal_line:
            near_end = 0.0 if bx[peak] < length / 2 else length
            limit = min(f["n"], stop + SPOT_WAIT_MAX)
            for t in range(end, limit):
                if not vis[t]:
                    continue
                corner = min(np.hypot(bx[t] - near_end, by[t] - cy) for cy in (0.0, width))
                goal_area = (abs(bx[t] - near_end) < SPOT_GOAL_AREA_X
                             and abs(by[t] - width / 2) < SPOT_GOAL_AREA_Y)
                if corner < SPOT_CORNER or goal_area:
                    end = t
                    break
        forced[start:end] = True
    return forced & f["graded"]


# Law 8: a kick-off is taken from the centre mark. While the ball is in view anywhere else, the
# kick-off has not happened yet. This stops a ball being rolled back to the spot from reading
# as the restart. And a tagged kick-off has happened by the end of its window.
KICKOFF_SPOT = 4.0
KICKOFF_PLAY = 5 * FPS
# The tagged side of a corner, goal kick or free kick takes the restart. The tagging is automatic,
# so this is evidence, not a constraint: log-odds of a tag being right about 95 % of the time.
# It acts only on the dead -> alive transition, so if the other side then gains control (a
# corner cleared and kept), the possession still changes; a header won without control does not.
RESTART_TEAM_BONUS = float(np.log(0.95 / 0.05))
RESTART_TEAM_BEFORE = 2 * FPS
RESTART_TEAM_AFTER = 18 * FPS
TAG_TEAMS = {"barcelona": HOME, "atletic club": AWAY}
# Whoever takes a set piece stands on the ball just before it: the side with clear contact in
# the last two seconds before a restart restarts. Measured against the tagged side of corners,
# goal kicks and free kicks it was right 13 times in 14, so its weight is the Laplace-smoothed
# log-odds of that record. Unlike the tag, it also covers throw-ins, which are never tagged.
TAKER_WINDOW = 2 * FPS
TAKER_REACH = 2.0
TAKER_BONUS = float(np.log((13 + 1) / (1 + 1)))


def _set_pieces(events):
    for event in events:
        if event["code"].strip().lower() == "set pieces":
            yield event, (event["tags"].get("Type") or "").strip().lower()


def taker_bias(f):
    """Per-frame log-bonus for a restart by home (> 0) or away (< 0), from who is on the ball.

    At frame t, the clear contacts of the preceding TAKER_WINDOW frames vote; the majority side
    gets TAKER_BONUS on a dead -> alive transition at t. No contact, no bonus.
    """
    near = np.minimum(f["d_home"], f["d_away"])
    clear = f["vis"] & (near < TAKER_REACH) & (np.abs(f["d_home"] - f["d_away"]) > CONTACT_MARGIN)
    vote = np.where(clear, np.where(f["d_home"] < f["d_away"], 1, -1), 0)
    total = np.concatenate(([0], np.cumsum(vote)))
    t = np.arange(len(vote))
    window = total[t + 1] - total[np.maximum(t + 1 - TAKER_WINDOW, 0)]
    return np.sign(window) * TAKER_BONUS


def build_priors(f, events, pitch, f_team=None):
    """Every Law-of-the-Game and Smart Tagging prior the decoder applies, as one dict.

    f_team, if given, is the feature set the possession model reads (with the broadcast's
    observations); the restart-taker evidence is computed from it, everything else from f.
    """
    length, width = pitch
    n = f["n"]
    forced_dead = before_kickoff(f, events) | ball_out_of_play(f, pitch)
    forced_alive = np.zeros(n, dtype=bool)
    restart_block = np.zeros(n, dtype=bool)
    restart_team = np.zeros(n)
    off_spot = f["vis"] & (np.hypot(f["bx"] - length / 2, f["by"] - width / 2) > KICKOFF_SPOT)
    for event, kind in _set_pieces(events):
        start, stop = max(0, event["start_frame"]), min(n, event["end_frame"])
        if kind.startswith("kick off"):
            restart_block[start:stop] |= off_spot[start:stop]
            forced_alive[stop:min(n, stop + KICKOFF_PLAY)] = True
            continue
        team = TAG_TEAMS.get(str(event["tags"].get("Team", "")).strip().lower())
        if team is None:
            continue                                  # throw-ins: no side tagged
        lo, hi = max(0, start - RESTART_TEAM_BEFORE), min(n, start + RESTART_TEAM_AFTER)
        restart_team[lo:hi] = RESTART_TEAM_BONUS if team == HOME else -RESTART_TEAM_BONUS
    restart_team += taker_bias(f_team if f_team is not None else f)
    forced_alive &= ~forced_dead & f["graded"]
    return {"forced_dead": forced_dead, "forced_alive": forced_alive,
            "restart_block": restart_block, "restart_team": restart_team}


def to_letters(states):
    """H/A/D on decoded frames, X wherever nothing was decoded (outside the graded halves)."""
    out = np.full(len(states), "X", dtype="<U1")
    ok = states >= 0
    out[ok] = LETTERS[states[ok]]
    return out


def mean_possession_s(states, graded):
    """Alive time per possession, in seconds: the statistic the switching prior encodes."""
    n_alive, n_spells = 0, 0
    for start, stop in run_lengths(graded):
        seg = states[start:stop]
        alive = (seg == HOME) | (seg == AWAY)
        n_alive += int(alive.sum())
        n_spells += int(np.sum(alive[1:] & (seg[1:] != seg[:-1]))) + int(alive[0])
    return n_alive / float(max(n_spells, 1)) / FPS


def calibrate_temperature(f, dead, table_logit, codes, priors, target_s=MEAN_POSSESSION_S,
                          low=0.01, high=1.0, steps=10):
    """Evidence temperature at which the decoded match agrees with the possession-length prior.

    The model assumes possessions last MEAN_POSSESSION_S on average; with the evidence too hot
    it decodes far shorter ones, too cold far longer ones. Bisection on a log scale finds the
    temperature where its own output is consistent with its own assumption. No labels enter:
    the target is the prior, and the statistic is measured on the decoded full match.
    """
    graded = f["graded"]
    states = None
    for _ in range(steps):
        mid = np.sqrt(low * high)
        states = decode(f, dead, contact=team_evidence(f, mid * table_logit, codes),
                        priors=priors)
        if mean_possession_s(states, graded) < target_s:
            high = mid          # too many possessions: evidence too strong, cool it
        else:
            low = mid
    return np.sqrt(low * high), states


def fit(f, dead, iterations=2, verbose=True, priors=None, target_s=MEAN_POSSESSION_S):
    """Seed with the contact rule, learn the proximity table, calibrate, re-decode.

    Returns (states, table, temperature). The contact rule only seeds the first pass. After it,
    the table's shape (which distances say how much) comes from self-training, and its overall
    scale from calibrate_temperature.
    """
    codes = proximity_codes(f)
    states = decode(f, dead, priors=priors)
    table, temperature = None, TEAM_TEMPERATURE
    for iteration in range(iterations):
        table_logit = fit_team_table(f, states, codes, temperature=1.0)
        temperature, new = calibrate_temperature(f, dead, table_logit, codes, priors, target_s)
        table = temperature * table_logit
        if verbose:
            graded = f["graded"]
            print("  team pass {}: temperature {:.3f}, mean possession {:.1f}s, {:6d} frames "
                  "changed".format(iteration, temperature, mean_possession_s(new, graded),
                                   int(np.sum(new[graded] != states[graded]))))
        states = new
    return states, table, temperature
