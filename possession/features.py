"""Per-frame features derived from the tracking and the Smart Tagging. No labels anywhere.

Every array is indexed by video frame (= submission row). Distances are metres, speeds m/s.

The features fall into two groups, matching the two questions the model has to answer:

  is the ball alive?   ball visibility, collective player speed and its rate of change, ball
                       position relative to the lines, ball speed, Smart Tagging set pieces
  who has it?          which team is in contact with the ball, and how clearly
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks  # noqa: E402

FPS = GAMES[GAME]["fps"]


def box_mean(x, half):
    """NaN-aware centred moving average over +/- half frames."""
    finite = np.isfinite(x)
    total = np.concatenate(([0.0], np.cumsum(np.where(finite, x, 0.0))))
    count = np.concatenate(([0], np.cumsum(finite)))
    index = np.arange(len(x))
    low = np.maximum(0, index - half)
    high = np.minimum(len(x), index + half + 1)
    n = count[high] - count[low]
    out = (total[high] - total[low]) / np.maximum(n, 1)
    out[n == 0] = np.nan
    return out


def graded_mask(n_frames, game=GAME):
    mask = np.zeros(n_frames, dtype=bool)
    for _, first, last in half_video_blocks(game):
        mask[first:last + 1] = True
    return mask


def run_lengths(mask):
    """[(start, stop)] of the True runs of a boolean array, half-open."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def ball_speed(bx, by, half=2):
    """Ball speed from a centred difference over +/- half frames. NaN where either end is."""
    n = len(bx)
    out = np.full(n, np.nan, dtype=np.float64)
    a, b = np.arange(n) - half, np.arange(n) + half
    ok = (a >= 0) & (b < n)
    ia, ib = a[ok], b[ok]
    out[ok] = np.hypot(bx[ib] - bx[ia], by[ib] - by[ia]) * FPS / (2.0 * half)
    return out


def _single_source_speed(bx, by, source=None, half=2):
    """ball_speed, but NaN wherever its window straddles a switch between cameras -- the two
    tracks disagree by up to a metre, which would otherwise read as a 25 m/s ball."""
    speed = ball_speed(bx.astype(np.float64), by.astype(np.float64), half)
    if source is None:
        return speed
    n = len(bx)
    a, b = np.clip(np.arange(n) - half, 0, n - 1), np.clip(np.arange(n) + half, 0, n - 1)
    return np.where(source[a] == source[b], speed, np.nan)


def contact_distances(px, py, bx, by, team):
    """(d_home, d_away): distance from the ball to the nearest tracked player of each team."""
    dist = np.hypot(px - bx[:, None], py - by[:, None])
    dist = np.where(np.isfinite(dist), dist, np.inf)
    return dist[:, team == 0].min(axis=1), dist[:, team == 1].min(axis=1)


def outside_margin(bx, by, pitch):
    """How far the ball is outside the pitch, in metres (<= 0 inside). NaN without a ball."""
    length, width = pitch
    return np.maximum.reduce([-bx, bx - length, -by, by - width])


def gap_length(vis):
    """For each frame without a ball, the length of the missing-ball run it sits in (else 0)."""
    out = np.zeros(len(vis), dtype=np.int32)
    for start, stop in run_lengths(~vis):
        out[start:stop] = stop - start
    return out


def set_piece_windows(events, n_frames):
    """Frames inside a Smart Tagging SET PIECES interval, and the tagged restart team.

    Returned as (inside, team) where team is 0/1 for a tagged side and -1 for N/A (every
    throw-in and kickoff). The interval is a window around the restart, not the restart itself.
    """
    inside = np.zeros(n_frames, dtype=bool)
    side = np.full(n_frames, -1, dtype=np.int8)
    team_ids = {"barcelona": 0, "atletic club": 1}
    for event in events:
        if event["code"].strip().lower() != "set pieces":
            continue
        start = max(0, event["start_frame"])
        stop = min(n_frames, event["end_frame"])
        inside[start:stop] = True
        team = team_ids.get(str(event["tags"].get("Team", "")).strip().lower(), -1)
        if team >= 0:
            side[start:stop] = team
    return inside, side


def build(data, events=None):
    """Every per-frame feature the model reads, as a dict of arrays."""
    px, py, ps, bx, by, team = (data[k] for k in ("px", "py", "ps", "bx", "by", "team"))
    n = len(bx)
    vis = np.isfinite(bx)

    d_home, d_away = contact_distances(px, py, bx, by, team)
    speed = np.nanmean(np.where(np.isfinite(ps), ps, np.nan), axis=1)
    speed = np.where(np.isfinite(px).sum(axis=1) > 0, speed, np.nan)

    out = {
        "n": n,
        "graded": graded_mask(n),
        "vis": vis,
        "bx": bx.astype(np.float64),
        "by": by.astype(np.float64),
        "ball_speed": _single_source_speed(bx, by, data.get("ball_source")),
        "d_home": d_home,
        "d_away": d_away,
        "outside": outside_margin(bx, by, data["pitch"]),
        "gap": gap_length(vis),
        "n_players": np.isfinite(px).sum(axis=1),
        "speed": speed,
        "speed_1s": box_mean(speed, FPS // 2),
        "speed_2s": box_mean(speed, FPS),
    }
    # Rate of change of collective speed over one second either side: players brake at a
    # whistle and accelerate at a restart, which is what times the edges of a dead-ball spell.
    s = out["speed_1s"]
    lag = FPS
    accel = np.full(n, np.nan)
    accel[lag:n - lag] = s[2 * lag:] - s[:n - 2 * lag]
    out["accel"] = accel

    if events is not None:
        out["set_piece"], out["set_piece_team"] = set_piece_windows(events, n)
    return out
