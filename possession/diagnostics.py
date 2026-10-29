"""Checks of a decoded match that need no ground truth.

The public five minutes are too small to choose between two versions of a model -- see
EVALUATION.md -- and the rules forbid tuning on them. These checks use the whole match instead,
and only things that are true by the laws of the game:

  restart geometry   every Smart Tagging corner, goal kick, throw-in and kickoff is a restart
                     from a known place. A decoded restart inside the tag window, with the ball
                     at that place, is a restart found and timed; anything else is a miss.
  plausibility       possession count and mean length, dead-ball share and spell count, home
                     share -- reported, never optimised, because a model can match them while
                     being wrong everywhere.

The Smart Tagging is itself automatic and imperfect, so neither check is a score. They exist to
catch a change that makes things clearly worse, on a sample of ~50 events rather than 17.
"""

import numpy as np

from features import FPS, run_lengths

# How close the ball must be to its restart spot, in metres. Generous on purpose: the ATD ball
# is projected to the ground, so a ball held over a thrower's head lands a metre or two inside.
TOL_CORNER = 4.0
TOL_TOUCHLINE = 4.0
TOL_GOAL_AREA_X = 8.0
TOL_GOAL_AREA_Y = 12.0
TOL_CENTRE = 4.0
SEARCH_BEFORE = 2 * FPS


def _ball_near(f, frame, window=10):
    """Ball position at `frame`, or at the nearest visible frame within +/- window."""
    for k in range(window + 1):
        for t in (frame + k, frame - k):
            if 0 <= t < f["n"] and f["vis"][t]:
                return f["bx"][t], f["by"][t]
    return None


def restart_in_place(kind, xy, pitch):
    if xy is None:
        return False
    x, y = xy
    length, width = pitch
    kind = kind.lower()
    if kind.startswith("corner"):
        return min(np.hypot(x - cx, y - cy) for cx in (0, length) for cy in (0, width)) \
            < TOL_CORNER
    if kind.startswith("throw"):
        return min(y, width - y) < TOL_TOUCHLINE
    if kind.startswith("goal kick"):
        return min(x, length - x) < TOL_GOAL_AREA_X and abs(y - width / 2) < TOL_GOAL_AREA_Y
    if kind.startswith("kick off"):
        return np.hypot(x - length / 2, y - width / 2) < TOL_CENTRE
    return True  # free kicks: anywhere


def restart_frames(dead):
    """First live frame after each dead spell."""
    return [stop for _, stop in run_lengths(dead)]


def restart_check(f, labels, events, pitch):
    """Per tagged set piece: was a restart decoded in its window, and from the right place?"""
    dead = labels == "D"
    restarts = np.array(restart_frames(dead))
    rows = []
    for event in events:
        if event["code"].strip().lower() != "set pieces":
            continue
        kind = event["tags"].get("Type", "")
        lo, hi = event["start_frame"] - SEARCH_BEFORE, event["end_frame"]
        inside = restarts[(restarts >= lo) & (restarts <= hi)] if len(restarts) else []
        found = len(inside) > 0
        placed = any(restart_in_place(kind, _ball_near(f, int(r)), pitch) for r in inside)
        rows.append({"type": kind, "start": event["start_frame"], "found": found,
                     "placed": placed})
    return rows


# Smart Tagging phases in which the tagged side is, by definition, the side with the ball. The
# defending codes are exact mirrors of these (same count, same intervals) and add nothing.
ATTACKING_PHASES = ("build up", "progression", "ball in final third", "creating chances",
                    "attacking transition", "ball in the box", "players in the box",
                    "long ball")
PHASE_TEAMS = {"barcelona": "H", "atletic club": "A"}


def phase_agreement(labels, events):
    """Share of decoded live frames inside tagged attacking phases that give the tagged side.

    A cross-check of the team call from an independent algorithm (Metrica's tagging). Phases are
    not possessions -- one can span a turnover -- so this is never 1.0 for a correct model; it
    is for catching a change that makes the team call clearly worse.
    """
    agree = total = 0
    for event in events:
        if event["code"].strip().lower() not in ATTACKING_PHASES:
            continue
        side = PHASE_TEAMS.get(str(event["tags"].get("Team", "")).strip().lower())
        if side is None:
            continue
        seg = labels[max(0, event["start_frame"]):event["end_frame"]]
        alive = (seg == "H") | (seg == "A")
        agree += int(np.sum(seg[alive] == side))
        total += int(alive.sum())
    return agree / float(total) if total else 0.0


def plausibility(labels, graded):
    lab = labels[graded]
    alive = lab != "D"
    spells, dead_spells = [], 0
    for start, stop in run_lengths(graded):
        seg = labels[start:stop]
        change = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        bounds = np.concatenate(([0], change, [len(seg)]))
        for a, b in zip(bounds[:-1], bounds[1:]):
            if seg[a] in ("H", "A"):
                spells.append(b - a)
            elif seg[a] == "D":
                dead_spells += 1
    spells = np.array(spells)
    return {
        "possessions": int(len(spells)),
        "mean_possession_s": float(spells.mean() / FPS) if len(spells) else 0.0,
        "short_possessions_lt2s": int(np.sum(spells < 2 * FPS)),
        "dead_spells": dead_spells,
        "dead_share": float(np.mean(lab == "D")),
        "home_share": float(np.mean(lab[alive] == "H")) if alive.any() else 0.0,
    }


def report(f, labels, events, pitch, name=""):
    rows = restart_check(f, labels, events, pitch)
    p = plausibility(labels, f["graded"])
    by_type = {}
    for row in rows:
        key = row["type"]
        found, placed, n = by_type.get(key, (0, 0, 0))
        by_type[key] = (found + row["found"], placed + row["placed"], n + 1)
    found = sum(r["found"] for r in rows)
    placed = sum(r["placed"] for r in rows)
    phases = phase_agreement(labels, events)
    print("{}  restarts found {}/{}  in place {}/{}  phase team {:.3f}  |  poss {}  mean {:.1f}s  "
          "<2s {}  dead spells {}  dead {:.3f}  home {:.3f}".format(
              name, found, len(rows), placed, len(rows), phases, p["possessions"],
              p["mean_possession_s"], p["short_possessions_lt2s"], p["dead_spells"],
              p["dead_share"], p["home_share"]))
    return {"restarts": rows, "by_type": by_type, "plausibility": p, "phase_agreement": phases,
            "found": found, "placed": placed, "n_tagged": len(rows)}
