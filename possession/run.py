"""Regenerate the submission: provided data in, frame,label CSV out. One command, no labels.

    python possession/run.py                       # -> evaluation/results/<name>.csv
    python possession/run.py --name myteam --out somewhere/

Pipeline (each step is its own module, documented there):

  atd.py        parse the raw tracking into arrays, in metres, cached under data/cache/
  features.py   per-frame features: ball visibility and speed, contact distances per team,
                collective player speed and acceleration, line margins, set-piece windows
  deadball.py   alive vs dead: an HMM whose emission histograms and stop/restart event bonuses
                are learned by hard EM over the whole match
  model.py      home / away / dead: a joint HMM where possession changes only on sustained
                contact, with a proximity evidence table learned by self-training
  diagnostics   checks that need no ground truth: tagged restarts found and in place,
                possession counts and shares

Deterministic: the same inputs give the same file, byte for byte. Nothing is random and nothing
reads a ground truth -- the public five minutes are used only by --check, after the file is
written, to confirm the pipeline is wired correctly.
"""

import argparse
import hashlib
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for path in (HERE, ROOT, os.path.join(ROOT, "starter"), os.path.join(ROOT, "evaluation")):
    if path not in sys.path:
        sys.path.insert(0, path)

import atd  # noqa: E402
import broadcast  # noqa: E402
import deadball  # noqa: E402
import diagnostics  # noqa: E402
import features  # noqa: E402
from features import run_lengths  # noqa: E402
import load_smart_tagging  # noqa: E402
import model  # noqa: E402

DEFAULT_NAME = "hmm_possession"
DEFAULT_OUT = os.path.join(ROOT, "evaluation", "results")


def write_submission(path, labels):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))
    return path


def team_features(data, events, f, verbose=True):
    """Features for the possession model: the tactical ones, with the ball filled in from the
    broadcast camera wherever only it sees the ball (see broadcast.py). The alive/dead model
    keeps the tactical-only features -- for it, "the ball is visible" is a signal in itself, and
    the broadcast sees the ball through stoppages that the wide camera loses it in."""
    try:
        other = atd.load(source="broadcast")
    except IOError:
        if verbose:
            print("  no broadcast tracking found -- possession model uses the tactical ball only")
        return f
    cache = os.path.join(atd.DATA_DIR, "cache",
                         "broadcast_map_v{}.npz".format(broadcast.MAP_VERSION))
    fused = broadcast.fuse(data, other, run_lengths(f["graded"]), cache_path=cache,
                           verbose=verbose)
    return features.build(fused, events)


def build_labels(verbose=True, mean_possession_s=model.MEAN_POSSESSION_S, use_broadcast=True):
    """The whole pipeline. Returns (labels, features, events, data)."""
    clock = time.time()
    data = atd.load()
    events = load_smart_tagging.load()
    f = features.build(data, events)
    if verbose:
        print("aligning the broadcast tracking:" if use_broadcast else "broadcast not used")
    f_team = team_features(data, events, f, verbose) if use_broadcast else f
    if verbose:
        print("features ready in {:.1f}s".format(time.time() - clock))
        print("alive/dead model (hard EM over the full match):")
    dead = deadball.fit(f, verbose=verbose)
    if verbose:
        print("possession model (self-trained proximity evidence):")
    priors = model.build_priors(f, events, data["pitch"], f_team)
    states, _, _ = model.fit(f_team, dead, verbose=verbose, priors=priors,
                             target_s=mean_possession_s)
    labels = model.to_letters(states)
    if verbose:
        print("done in {:.1f}s".format(time.time() - clock))
    return labels, f, events, data


def public_check(path):
    """Score the written file against the public 5 minutes. A wiring check, not a target."""
    import possession_metrics as pm
    gt_path = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
    gt = pm.label_codes(pm.read_label_csv(gt_path))
    frame = pm.read_label_csv(path)
    reason = pm.validate_submission(frame, len(gt))
    if reason:
        print("INVALID: " + reason)
        return
    r = pm.evaluate(gt, pm.label_codes(frame))
    print("public 5-minute check (17 possessions -- NOT a ranking signal, see EVALUATION.md):")
    print("  macroF1 {:.4f}   spellF1 {:.4f}   possessions {} (truth 17)".format(
        r["macro_f1"], r["spell_f1_mean"], r["n_pred_spells"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default=DEFAULT_NAME, help="file name, without .csv")
    parser.add_argument("--out", default=DEFAULT_OUT, help="output folder")
    parser.add_argument("--mean-possession", type=float, default=model.MEAN_POSSESSION_S,
                        help="the possession-length prior, seconds (default %(default)s). A "
                             "football prior, not a fitted value -- see TECHNICAL.md")
    parser.add_argument("--no-broadcast", action="store_true",
                        help="ignore the broadcast tracking (tactical ball only)")
    parser.add_argument("--check", action="store_true",
                        help="after writing, score against the public 5 minutes")
    args = parser.parse_args()

    labels, f, events, data = build_labels(mean_possession_s=args.mean_possession,
                                           use_broadcast=not args.no_broadcast)
    path = write_submission(os.path.join(args.out, args.name + ".csv"), labels)
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
    print("\nwrote {}  (sha256 {}...)".format(os.path.relpath(path, ROOT), digest))
    counts = {k: int(np.sum(labels == k)) for k in "HADX"}
    print("  H {H}  A {A}  D {D}  X {X}".format(**counts))
    print("\nground-truth-free diagnostics, full match:")
    diagnostics.report(f, labels, events, data["pitch"], " ")
    if args.check:
        print()
        public_check(path)


if __name__ == "__main__":
    main()
