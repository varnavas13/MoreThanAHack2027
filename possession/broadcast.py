"""Align the broadcast tracking to the tactical timeline, algorithmically, and use its ball.

The broadcast recording is a second, independent camera with its own automatic tracking, and
so its own ball track, which sees the ball at different moments than the tactical camera does
(close-ups of a corner taker, a ball the wide shot loses in a crowd). No mapping between the two
timelines is provided. This module recovers one from the tracking alone.

Both ATDs are in the same pitch coordinates, so a broadcast frame and the tactical frame showing
the same instant have the same players in the same places. Matching the broadcast's visible
players against every tactical frame (chamfer distance over a one-second clip) finds that the
live broadcast sits at ONE constant offset per half: a TV feed shows replays over live time
rather than pausing it, so the offset never drifts, and replay or close-up frames simply fail to
match. So:

  1. estimate each half's offset by a coarse-to-fine search over a few dozen sampled moments,
     taking the median, which is robust to the samples that land in a replay;
  2. verify every tactical frame individually: the broadcast frame at that offset must show
     enough players, each close to a tactical player;
  3. on verified frames, the broadcast ball is a second observation of the same ball.

No human mapping, no labels: the alignment is a by-product of the tracking.
"""

import numpy as np

from features import FPS, run_lengths

# Bump when anything below changes what the mapping is; it names the cache file.
MAP_VERSION = 1
N_SAMPLES = 20
COARSE_STEP = 8
CLIP = (0, FPS // 2, FPS - 1)          # three frames across one second
MIN_PLAYERS = 8                        # a broadcast frame showing fewer cannot be verified
CAP = 5.0                              # metres: a player further than this counts as this far
# A verified frame's mean player-to-nearest-player distance. Live frames measure 0.2-0.9 m;
# the best wrong frame anywhere in the match measures 1.8 m or more.
VERIFY = 1.2
# A broadcast player within this distance of a tactical player is the same person seen twice;
# further away, he is someone the tactical tracker missed (it sees ~20.5 of 22).
SAME_PLAYER = 1.0


def _positions(data):
    p = np.stack([data["px"], data["py"]], axis=-1).astype(np.float32)
    return p


def _cost(b_points, t_positions):
    """Mean distance from each broadcast player to the nearest tactical player, per frame."""
    d2 = ((t_positions[:, None, :, :] - b_points[None, :, None, :]) ** 2).sum(-1)
    d2 = np.where(np.isfinite(d2), d2, np.inf).min(-1)
    return np.minimum(np.sqrt(d2), CAP).mean(-1)


def _clip_cost(bp, tp, b_frame, t_frames):
    total = 0.0
    for k in CLIP:
        pts = bp[b_frame + k]
        pts = pts[np.isfinite(pts[:, 0])]
        total = total + _cost(pts, tp[t_frames + k])
    return total / len(CLIP)


def estimate_offsets(tactical, broadcast, halves, verbose=True):
    """{half_index: offset} with tactical_frame = broadcast_frame + offset, one per half.

    Sampled broadcast moments are each matched against the whole tactical match (coarse grid,
    then refined to the frame); a match counts only if it is unambiguous, i.e. the clip cost is
    under VERIFY. The median offset of the matches landing in each half is that half's offset.
    """
    tp, bp = _positions(tactical), _positions(broadcast)
    n_b = np.isfinite(broadcast["px"]).sum(axis=1)
    n_t = np.isfinite(tactical["px"]).sum(axis=1)
    candidates = np.arange(0, len(tp) - FPS, COARSE_STEP)
    candidates = candidates[n_t[candidates] >= 12]
    found = {index: [] for index in range(len(halves))}
    for b_frame in np.linspace(FPS, len(bp) - 3 * FPS, N_SAMPLES).astype(int):
        # nudge forward to a moment with enough players in view
        for b_frame in range(b_frame, min(b_frame + 40 * FPS, len(bp) - 2 * FPS), FPS):
            if min(n_b[b_frame + k] for k in CLIP) >= 10:
                break
        else:
            continue
        cost = _clip_cost(bp, tp, b_frame, candidates)
        if cost.min() > VERIFY:
            continue                                   # a replay or a close-up: no live match
        best = int(candidates[np.argmin(cost)])
        fine = np.arange(best - COARSE_STEP, best + COARSE_STEP + 1)
        best = int(fine[np.argmin(_clip_cost(bp, tp, b_frame, fine))])
        for index, (start, stop) in enumerate(halves):
            if start <= best < stop:
                found[index].append(best - b_frame)
    offsets = {}
    for index, values in found.items():
        if values:
            offsets[index] = int(np.median(values))
            if verbose:
                print("  half {}: offset {:+d} frames ({} matched moments, spread {})".format(
                    index + 1, offsets[index], len(values), max(values) - min(values)))
    return offsets


def verified_mapping(tactical, broadcast, halves, offsets):
    """(b_index, cost): the broadcast frame for each tactical frame, -1 where not verified."""
    tp, bp = _positions(tactical), _positions(broadcast)
    n = len(tp)
    b_index = np.full(n, -1, dtype=np.int64)
    cost = np.full(n, np.inf, dtype=np.float32)
    for index, (start, stop) in enumerate(halves):
        if index not in offsets:
            continue
        t = np.arange(start, stop)
        b = t - offsets[index]
        ok = (b >= 0) & (b < len(bp))
        t, b = t[ok], b[ok]
        n_b = np.isfinite(bp[b, :, 0]).sum(axis=1)
        t, b = t[n_b >= MIN_PLAYERS], b[n_b >= MIN_PLAYERS]
        for a in range(0, len(t), 4000):
            tt, bb = t[a:a + 4000], b[a:a + 4000]
            d2 = ((tp[tt][:, None, :, :] - bp[bb][:, :, None, :]) ** 2).sum(-1)   # C x Kb x Kt
            d2 = np.where(np.isfinite(d2), d2, np.inf).min(-1)                       # C x Kb
            seen = np.isfinite(bp[bb][:, :, 0])
            d = np.where(seen, np.minimum(np.sqrt(d2), CAP), 0.0)
            c = d.sum(-1) / np.maximum(seen.sum(-1), 1)
            cost[tt] = c
            good = c < VERIFY
            b_index[tt[good]] = bb[good]
    return b_index, cost


def ball_track(tactical, broadcast, b_index):
    """(bx, by) of the broadcast ball on the tactical timeline, NaN where unverified or unseen."""
    bx = np.full(len(b_index), np.nan)
    by = np.full(len(b_index), np.nan)
    ok = b_index >= 0
    bx[ok] = broadcast["bx"][b_index[ok]]
    by[ok] = broadcast["by"][b_index[ok]]
    return bx, by


def extra_players(tactical, broadcast, b_index):
    """(x, y) of broadcast players with no tactical player within SAME_PLAYER, per frame.

    Shaped (frames, broadcast slots), NaN where there is no such player or the frame is not
    verified. Their teams are broadcast["team"].
    """
    n, slots = len(b_index), broadcast["px"].shape[1]
    ex = np.full((n, slots), np.nan, np.float32)
    ey = np.full((n, slots), np.nan, np.float32)
    frames = np.flatnonzero(b_index >= 0)
    for a in range(0, len(frames), 20000):
        fr = frames[a:a + 20000]
        bpx, bpy = broadcast["px"][b_index[fr]], broadcast["py"][b_index[fr]]
        d = np.hypot(bpx[:, :, None] - tactical["px"][fr][:, None, :],
                     bpy[:, :, None] - tactical["py"][fr][:, None, :])
        seen_twice = np.where(np.isfinite(d), d, np.inf).min(axis=-1) < SAME_PLAYER
        keep = np.isfinite(bpx) & ~seen_twice
        ex[fr] = np.where(keep, bpx, np.nan)
        ey[fr] = np.where(keep, bpy, np.nan)
    return ex, ey


def fuse(tactical, broadcast, halves, cache_path=None, verbose=True):
    """A copy of the tactical data with the broadcast's observations added where it has some.

    The ball is the tactical ball where seen, else the verified broadcast ball ("ball_source":
    0 tactical, 1 broadcast, -1 neither). Players the tactical tracker missed but the broadcast
    shows are appended as extra slots, with the broadcast's team labels and no speed.

    Where both cameras see the ball the tactical one is kept: it is the reference camera, it has
    no cuts, and where the two disagree by metres it is usually a lofted ball that each camera
    projects onto the ground differently, and the wide, high tactical view is the less wrong.
    """
    if cache_path is not None:
        try:
            cached = np.load(cache_path)
            b_index = cached["b_index"]
        except (OSError, KeyError):
            b_index = None
    else:
        b_index = None
    if b_index is None:
        offsets = estimate_offsets(tactical, broadcast, halves, verbose=verbose)
        b_index, _ = verified_mapping(tactical, broadcast, halves, offsets)
        if cache_path is not None:
            np.savez(cache_path, b_index=b_index)
    bbx, bby = ball_track(tactical, broadcast, b_index)
    seen = np.isfinite(tactical["bx"])
    fused = dict(tactical)
    fused["bx"] = np.where(seen, tactical["bx"], bbx).astype(np.float32)
    fused["by"] = np.where(seen, tactical["by"], bby).astype(np.float32)
    fused["ball_source"] = np.where(seen, 0, np.where(np.isfinite(bbx), 1, -1)).astype(np.int8)
    ex, ey = extra_players(tactical, broadcast, b_index)
    fused["px"] = np.concatenate([tactical["px"], ex], axis=1)
    fused["py"] = np.concatenate([tactical["py"], ey], axis=1)
    fused["ps"] = np.concatenate([tactical["ps"], np.full_like(ex, np.nan)], axis=1)
    fused["team"] = np.concatenate([tactical["team"], broadcast["team"]])
    fused["gk"] = np.concatenate([tactical["gk"], broadcast["gk"]])
    if verbose:
        print("  broadcast verified on {:.1%} of frames; ball added on {} frames, {} player "
              "observations the tactical tracker missed".format(
                  float(np.mean(b_index >= 0)), int(np.sum(fused["ball_source"] == 1)),
                  int(np.isfinite(ex).sum())))
    return fused

