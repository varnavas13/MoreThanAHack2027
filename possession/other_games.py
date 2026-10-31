"""Validate the model on two other matches: Metrica Sports' public sample games.

The provided match has no usable labels: the public five minutes are 17 possessions and the rules
forbid tuning on them. Labels on OTHER games are allowed in any amount (RULES.md), so this module
builds a possession ground truth for Metrica's two open sample matches from their event data, runs
the unchanged pipeline on their tracking, and scores it with the official metric.

    python possession/other_games.py                  # downloads ~125 MB once, ~5 min

Data: https://github.com/metrica-sports/sample-data (Sample_Game_1, Sample_Game_2): tracking at
25 fps and events annotated by Metrica Sports. Saved under data/external/, which is gitignored.

Ground truth from events, under the challenge's definition (CHALLENGE.md):

  * a side gains possession when it controls the ball: a pass, shot, set piece or recovery;
  * a challenge or duel is not control, so it changes nothing;
  * the ball dies at a ball out, a foul received, an offside, a handball, a goal or a shot out,
    and lives again at the next restart (a set piece, or a goal kick, which Metrica files as a
    pass);
  * a possession is a maximal run of one side with the ball alive.

Two honest differences from the competition: this tracking is corrected (no identity switches,
fewer gaps), so it is easier than the automatic tracking of BAR-ATH; and event times are annotated
by hand, a few frames from the instant a corrected tracking would mark.
"""

import argparse
import io
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for path in (HERE, ROOT, os.path.join(ROOT, "evaluation")):
    if path not in sys.path:
        sys.path.insert(0, path)

import deadball  # noqa: E402
import features  # noqa: E402
import model  # noqa: E402
import possession_metrics as pm  # noqa: E402
from games import DATA_DIR  # noqa: E402

URL = ("https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/"
       "Sample_Game_{g}/Sample_Game_{g}_{kind}.csv")
FILES = {"events": "RawEventsData", "home": "RawTrackingData_Home_Team",
         "away": "RawTrackingData_Away_Team"}
GAMES = (1, 2)
PITCH = (105.0, 68.0)
FPS = 25
CONTROL = {"PASS", "SHOT", "SET PIECE", "RECOVERY"}


def fetch(game, kind, folder):
    path = os.path.join(folder, "Sample_Game_{}_{}.csv".format(game, FILES[kind]))
    if not os.path.isfile(path):
        os.makedirs(folder, exist_ok=True)
        print("  downloading {}".format(os.path.basename(path)))
        urllib.request.urlretrieve(URL.format(g=game, kind=FILES[kind]), path)
    return path


def ground_truth(events, n_frames):
    """Per-frame H/A/D/X from the events (X before kick-off, between and after the periods)."""
    labels = np.full(n_frames, "X", dtype="<U1")
    for _, period in events.groupby("Period"):
        period = period.sort_values(["Start Frame", "End Frame"])
        start = int(period["Start Frame"].min())
        stop = min(n_frames, int(period[["Start Frame", "End Frame"]].max().max()) + 1)
        marks, dead = [], False
        for _, e in period.iterrows():
            kind, sub = e["Type"], str(e["Subtype"])
            side = "H" if e["Team"] == "Home" else "A"
            first, last = int(e["Start Frame"]), int(e["End Frame"])
            if kind == "SET PIECE" or "GOAL KICK" in sub:
                marks.append((first, side))
                dead = False
            elif dead:
                continue
            elif kind == "BALL OUT" or (kind == "SHOT" and (sub.endswith("GOAL")
                                                             or sub.endswith("-OUT"))):
                marks.append((max(first, last), "D"))
                dead = True
            elif kind == "FAULT RECEIVED" or (kind == "BALL LOST"
                                               and ("OFFSIDE" in sub or "HAND BALL" in sub)):
                marks.append((first, "D"))
                dead = True
            elif kind in CONTROL:
                marks.append((first, side))
        marks.sort()
        state, position = marks[0][1], start
        for frame, new in marks:
            frame = min(max(frame, start), stop)
            labels[position:frame] = state
            state, position = new, frame
        labels[position:stop] = state
    return labels


def _read_side(path):
    """Metrica's raw tracking CSV: three header lines, then x,y pairs per player, ball last."""
    with open(path, encoding="utf-8") as handle:
        handle.readline()
        handle.readline()
        names = handle.readline().strip().split(",")
        table = pd.read_csv(handle, header=None, dtype=np.float64, na_values=["NaN"]).values
    frame = table[:, 1].astype(np.int64) - 1
    columns = [i for i, name in enumerate(names) if name.startswith("Player")]
    xs = np.stack([table[:, c] for c in columns], axis=1)
    ys = np.stack([table[:, c + 1] for c in columns], axis=1)
    ball = names.index("Ball")
    return frame, xs, ys, table[:, ball], table[:, ball + 1]


def tracking(game, folder):
    """The pipeline's data dict for one sample game, in metres, video frame = row."""
    frame, hx, hy, bx, by = _read_side(fetch(game, "home", folder))
    frame_a, ax, ay, _, _ = _read_side(fetch(game, "away", folder))
    if not np.array_equal(frame, frame_a):
        raise ValueError("home and away tracking are not frame-aligned")
    n = int(frame.max()) + 1
    length, width = PITCH

    def place(values, scale):
        out = np.full((n,) + values.shape[1:], np.nan, np.float32)
        out[frame] = values * scale
        return out

    px = place(np.concatenate([hx, ax], axis=1), length)
    py = place(np.concatenate([hy, ay], axis=1), width)
    speed = np.full_like(px, np.nan)
    speed[1:-1] = np.hypot(px[2:] - px[:-2], py[2:] - py[:-2]) * FPS / 2.0
    team = np.array([0] * hx.shape[1] + [1] * ax.shape[1], dtype=np.int8)
    return {"px": px, "py": py, "ps": speed, "bx": place(bx, length), "by": place(by, width),
            "team": team, "gk": np.zeros(len(team), dtype=bool), "attack_first_half": {},
            "pitch": PITCH}


def run(data, truth, mean_possession_s=model.MEAN_POSSESSION_S, verbose=False):
    """The unchanged pipeline on another match; returns its official metrics."""
    data = dict(data, graded=truth != "X")
    f = features.build(data, events=None)
    dead = deadball.fit(f, verbose=verbose)
    priors = model.build_priors(f, [], data["pitch"], f)
    states, _, _ = model.fit(f, dead, verbose=verbose, priors=priors, target_s=mean_possession_s)
    labels = model.to_letters(states)
    return pm.evaluate(pm.label_codes(truth), pm.label_codes(labels)), labels


def baseline(data, truth, window=50):
    """The organisers' baseline on the same match: nearest player, majority vote over +/-50."""
    d_home, d_away = features.contact_distances(data["px"], data["py"], data["bx"], data["by"],
                                                data["team"])
    codes = np.where(np.isfinite(np.minimum(d_home, d_away)),
                     np.where(d_home < d_away, 0, 1), -1)
    home = np.concatenate(([0], np.cumsum(codes == 0)))
    away = np.concatenate(([0], np.cumsum(codes == 1)))
    i = np.arange(len(codes))
    lo, hi = np.maximum(0, i - window), np.minimum(len(codes), i + window + 1)
    h, a = home[hi] - home[lo], away[hi] - away[lo]
    labels = np.where(h + a > 0, np.where(h >= a, "H", "A"), "D")
    labels = np.where(truth == "X", "X", labels)
    return pm.evaluate(pm.label_codes(truth), pm.label_codes(labels))


def truth_stats(truth):
    graded = truth != "X"
    spells = pm.possession_spells(pm.label_codes(truth), pm.valid_runs(pm.label_codes(truth)))
    alive = sum(b - a for a, b, _ in spells)
    stops = len([s for s in pm.segments(pm.label_codes(truth), pm.valid_runs(pm.label_codes(truth)))
                 if s[2] == pm.D])
    return {"possessions": len(spells), "alive_per_possession_s": alive / len(spells) / FPS,
            "stoppages": stops, "dead_share": float(np.mean(truth[graded] == "D"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--folder", default=os.path.join(DATA_DIR, "external", "metrica"))
    parser.add_argument("--mean-possession", type=float, nargs="+",
                        default=[model.MEAN_POSSESSION_S])
    args = parser.parse_args()
    for game in GAMES:
        events = pd.read_csv(fetch(game, "events", args.folder))
        data = tracking(game, args.folder)
        truth = ground_truth(events, len(data["bx"]))
        t = truth_stats(truth)
        print("Sample game {}: {} possessions, {:.1f} s live per possession, {} stoppages, "
              "{:.0%} dead".format(game, t["possessions"], t["alive_per_possession_s"],
                                   t["stoppages"], t["dead_share"]))
        b = baseline(data, truth)
        print("  baseline (nearest player, +/-50)  spellF1 {:.3f}  macroF1 {:.3f}  possessions "
              "{}".format(b["spell_f1_mean"], b["macro_f1"], b["n_pred_spells"]))
        for target in args.mean_possession:
            r, _ = run(data, truth, target)
            print("  model, prior {:>4.1f} s             spellF1 {:.3f}  macroF1 {:.3f}  possessions "
                  "{}  stoppages {}  dead F1 {:.3f}".format(
                      target, r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"],
                      r["n_pred_segments"] - r["n_pred_spells"], r["per_class"]["D"]["f1"]))


if __name__ == "__main__":
    main()
