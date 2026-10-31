"""Is the ball alive? A two-state HMM whose emission model is learned from the match itself.

No labels are used, from this game or any other. The model is fitted by hard EM over the whole
match:

  1. a crude rule gives an initial alive/dead split (ball missing for a long stretch while the
     players walk, or the ball clearly over a line);
  2. per-feature histograms are estimated for each class from the current split;
  3. the summed log-likelihood ratios are decoded with a sticky two-state Viterbi that has
     minimum durations, giving a new split;
  4. repeat from 2 until the split stops changing.

The features are discretised on purpose. Their class-conditional shapes are far from Gaussian
(ball visibility is binary, the out-of-bounds margin is meaningful only near zero), and a
histogram with Laplace smoothing assumes nothing about them.

Two priors, both from football rather than from this game's labels:

  * a match has on the order of 100-150 stoppages, which sets the switching penalties;
  * a stoppage lasts at least about two seconds, and a spell of live play between two
    stoppages at least about one.

Naive Bayes over-counts correlated evidence, so the summed log-ratio is tempered by TEMPERATURE
before decoding. That shrinks every frame's vote by the same factor; it moves no boundary that
is supported by sustained evidence, and suppresses blips that are not.
"""

import numpy as np

import hmm
from features import FPS, run_lengths

ALIVE, DEAD = 0, 1

# Priors (see module docstring). Expressed per frame.
STOPPAGES_PER_MATCH = 120
GRADED_FRAMES = 158857
DEAD_SHARE = 0.4
P_STOP = STOPPAGES_PER_MATCH / (GRADED_FRAMES * (1 - DEAD_SHARE))
P_RESTART = STOPPAGES_PER_MATCH / (GRADED_FRAMES * DEAD_SHARE)
MIN_DEAD = 2 * FPS
MIN_ALIVE = 1 * FPS
TEMPERATURE = 1.0 / 3.0
LAPLACE = 1.0
MAX_ITER = 8

# Bin edges. Chosen for physical meaning, not fitted: walking vs jogging vs running for the
# players, a ball at rest / rolling / passed / struck, a ball well inside / at / over a line.
SPEED_EDGES = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0]
ACCEL_EDGES = [-1.0, -0.5, -0.2, 0.2, 0.5, 1.0]
BALL_SPEED_EDGES = [0.5, 2.0, 5.0, 10.0, 18.0]
GAP_EDGES = [2 * FPS, 5 * FPS, 10 * FPS]
OUTSIDE_EDGES = [-3.0, -1.0, 0.0, 1.0, 3.0]
CONTACT_EDGES = [1.0, 2.0, 4.0, 8.0]


def _digitize(x, edges, missing_bin):
    out = np.digitize(np.nan_to_num(x, nan=0.0), edges)
    return np.where(np.isfinite(x), out, missing_bin)


def discretize(f):
    """(T, n_views) integer matrix: one joint categorical code per evidence view per frame.

    Features that describe the same thing are binned jointly rather than separately, so naive
    Bayes counts each source of evidence once. Ball visibility in particular would otherwise be
    counted three times (the ball, out-of-bounds and contact features all have a
    "missing" bin), and the model would conclude that a dead ball is never visible, which is
    exactly wrong for throw-ins and goal kicks.

      view 0, the ball:     missing -> how long; visible -> speed x line x contact
      view 1, the players:  collective speed x its rate of change
      view 2, the tagging:  inside a Smart Tagging set-piece window or not
    """
    vis = f["vis"]
    n_speed = len(BALL_SPEED_EDGES) + 1
    n_line = len(OUTSIDE_EDGES) + 1
    n_contact = len(CONTACT_EDGES) + 1
    contact = np.minimum(f["d_home"], f["d_away"])
    seen = (_digitize(f["ball_speed"], BALL_SPEED_EDGES, 0) * n_line
            + _digitize(f["outside"], OUTSIDE_EDGES, 0)) * n_contact         + _digitize(np.where(np.isfinite(contact), contact, np.nan), CONTACT_EDGES, 0)
    missing = n_speed * n_line * n_contact + np.digitize(f["gap"], GAP_EDGES)
    ball = np.where(vis, seen, missing)

    n_accel = len(ACCEL_EDGES) + 2
    players = (_digitize(f["speed_1s"], SPEED_EDGES, len(SPEED_EDGES) + 1) * n_accel
               + _digitize(f["accel"], ACCEL_EDGES, len(ACCEL_EDGES) + 1))
    columns = [ball, players]
    if "set_piece" in f:
        columns.append(f["set_piece"].astype(np.int64))
    return np.stack(columns, axis=1).astype(np.int64)


def initial_split(f):
    """The crude rule EM starts from. True = dead.

    Deliberately not "the ball is missing": players walking is the primary cue, so that EM can
    discover visible-ball stoppages rather than being seeded never to look for them.
    """
    speed = np.nan_to_num(f["speed_2s"], nan=0.0)
    long_gap = f["gap"] > 5 * FPS
    over_line = np.nan_to_num(f["outside"], nan=-99) > 1.0
    return (speed < 1.2) | (long_gap & (speed < 1.8)) | over_line


def _any_within(mask, lo, hi):
    """For each t: is mask True anywhere in [t+lo, t+hi]? (clipped at the ends)"""
    c = np.concatenate(([0], np.cumsum(mask.astype(np.int64))))
    t = np.arange(len(mask))
    a = np.clip(t + lo, 0, len(mask))
    b = np.clip(t + hi + 1, 0, len(mask))
    return (c[b] - c[a]) > 0


def _window_mean(x, lo, hi):
    """NaN-aware mean of x over [t+lo, t+hi] for each t."""
    finite = np.isfinite(x)
    total = np.concatenate(([0.0], np.cumsum(np.where(finite, x, 0.0))))
    count = np.concatenate(([0], np.cumsum(finite)))
    t = np.arange(len(x))
    a = np.clip(t + lo, 0, len(x))
    b = np.clip(t + hi + 1, 0, len(x))
    n = count[b] - count[a]
    return np.where(n > 0, (total[b] - total[a]) / np.maximum(n, 1), np.nan)


def event_codes(f):
    """(stop_code, restart_code): what the frame looks like as a candidate edge of a stoppage.

    A stoppage starts at an event (the ball crosses a line, the ball vanishes, everyone brakes
    at a whistle) and ends at one: the ball is struck from rest, it reappears, everyone
    accelerates. These codes describe frame t as a candidate first-dead (stop) or first-alive
    (restart) frame; deadball.fit learns how much each pattern raises the odds of a transition.
    """
    vis = f["vis"]
    window = 2 * FPS
    before = _window_mean(f["speed"], -window, -1)
    after = _window_mean(f["speed"], 0, window - 1)
    change = np.nan_to_num(after - before, nan=0.0)

    over = vis & (np.nan_to_num(f["outside"], nan=-99) > 0.0)
    crossed = _any_within(over, -5, 5)
    gap_start = np.zeros(len(vis), dtype=bool)
    gap_end = np.zeros(len(vis), dtype=bool)
    for start, stop in run_lengths(~vis):
        if stop - start >= FPS:
            gap_start[start] = True
            if stop < len(vis):
                gap_end[stop] = True
    vanishes = _any_within(gap_start, -5, 2 * FPS)
    stop_code = (crossed * 2 + vanishes) * 3 + np.digitize(change, [-0.6, -0.2])

    moving = vis & (np.nan_to_num(f["ball_speed"], nan=0.0) > 5.0)
    still = (~vis) | (np.nan_to_num(f["ball_speed"], nan=99.0) < 1.5)
    rested = _window_mean(still.astype(np.float64), -window, -3) > 0.8
    kicked = _any_within(moving, -3, 10) & rested
    appears = _any_within(gap_end, -10, FPS)
    tagged = f["set_piece"] if "set_piece" in f else np.zeros(len(vis), dtype=bool)
    restart_code = ((kicked * 2 + appears) * 3 + np.digitize(change, [0.2, 0.6])) * 2 + tagged
    return stop_code.astype(np.int64), restart_code.astype(np.int64)


def transition_frames(dead, runs):
    """(stops, restarts): first dead / first alive frame of every change inside the runs."""
    stops, restarts = [], []
    for start, stop in runs:
        seg = dead[start:stop].astype(np.int8)
        change = np.flatnonzero(np.diff(seg)) + 1
        for c in change:
            (stops if seg[c] else restarts).append(start + int(c))
    return np.array(stops, dtype=np.int64), np.array(restarts, dtype=np.int64)


def fit_event_bonus(code, at, mask):
    """log P(code | transition here) - log P(code | any frame): a per-frame switching bonus."""
    n_bins = int(code.max()) + 1
    on = np.bincount(code[at], minlength=n_bins) + LAPLACE
    off = np.bincount(code[mask], minlength=n_bins) + LAPLACE
    return np.log(on / on.sum()) - np.log(off / off.sum())


def fit_histograms(codes, dead, mask):
    """log P(code | dead) - log P(code | alive), per feature, as a list of lookup arrays."""
    tables = []
    for j in range(codes.shape[1]):
        n_bins = int(codes[:, j].max()) + 1
        on = np.bincount(codes[mask & dead, j], minlength=n_bins) + LAPLACE
        off = np.bincount(codes[mask & ~dead, j], minlength=n_bins) + LAPLACE
        tables.append(np.log(on / on.sum()) - np.log(off / off.sum()))
    return tables


def log_ratio(codes, tables):
    return sum(table[codes[:, j]] for j, table in enumerate(tables))


def decode(llr, runs, temperature=TEMPERATURE, stop_bonus=None, restart_bonus=None):
    """Sticky two-state Viterbi with minimum durations, one graded run at a time.

    stop_bonus / restart_bonus are per-frame log-bonuses on the alive->dead and dead->alive
    transitions (see event_codes); without them the switching penalty is the same everywhere.
    """
    trans = np.log(np.array([[1 - P_STOP, P_STOP], [P_RESTART, 1 - P_RESTART]]))
    dead = np.zeros(len(llr), dtype=bool)
    for start, stop in runs:
        emission = np.stack([np.zeros(stop - start), temperature * llr[start:stop]], axis=1)
        bonus = None
        if stop_bonus is not None:
            bonus = np.zeros((stop - start, 2, 2))
            bonus[:, ALIVE, DEAD] = stop_bonus[start:stop]
            bonus[:, DEAD, ALIVE] = restart_bonus[start:stop]
        path = hmm.viterbi_duration(emission, trans, [MIN_ALIVE, MIN_DEAD], bonus)
        dead[start:stop] = path == DEAD
    return dead


def fit(f, verbose=True, events=True):
    """Run hard EM.

    Returns a dict: dead (mask), llr (per-frame log-ratio), stop_bonus and restart_bonus
    (per-frame transition log-bonuses, zeros when events=False), tables, history.
    """
    codes = discretize(f)
    stop_code, restart_code = event_codes(f)
    graded = f["graded"]
    runs = run_lengths(graded)
    dead = initial_split(f) & graded
    stop_bonus = restart_bonus = None
    history = []
    for iteration in range(MAX_ITER):
        tables = fit_histograms(codes, dead, graded)
        llr = log_ratio(codes, tables)
        if events and iteration > 0:
            # Edges are only worth learning once the split is roughly right, so the first pass
            # decodes with flat penalties and the bonuses are fitted from its transitions.
            stops, restarts = transition_frames(dead, runs)
            stop_bonus = fit_event_bonus(stop_code, stops, graded)[stop_code]
            restart_bonus = fit_event_bonus(restart_code, restarts, graded)[restart_code]
        new = decode(llr, runs, stop_bonus=stop_bonus, restart_bonus=restart_bonus)
        changed = int(np.sum(new[graded] != dead[graded]))
        n_spells = len(run_lengths(new & graded))
        history.append((iteration, changed, n_spells, float(new[graded].mean())))
        if verbose:
            print("  EM {}: {:6d} frames changed, {:4d} dead spells, dead share {:.3f}".format(
                iteration, changed, n_spells, new[graded].mean()))
        dead = new
        if changed == 0 and (not events or iteration > 0):
            break
    zeros = np.zeros(len(llr))
    return {"dead": dead, "llr": llr,
            "stop_bonus": stop_bonus if stop_bonus is not None else zeros,
            "restart_bonus": restart_bonus if restart_bonus is not None else zeros,
            "tables": tables, "history": history}
