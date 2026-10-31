"""Check the tracking's team labels against the kits on the tactical video. Validation only.

The pipeline never reads the video. This tool exists to measure, on the video, a claim the
pipeline relies on: that goalkeeper slots must be re-assigned by which goal they stand in
(features.keeper_teams), while outfield slots can keep their metadata team.

It samples frames where a tracked player is on the ball, projects him into the image with the
provided homography, takes a torso patch, and classifies the kit by colour, with no labels and no
learning, two fixed colour rules read off the kits of this match:

  goalkeepers   Barcelona lime, Athletic teal: the share of teal pixels (hue 70-100) separates
                them, because grass never reaches that hue
  outfield      Barcelona dark blaugrana, Athletic white: the share of white pixels among the
                non-grass ones

    pip install opencv-python-headless==5.0.0.93      # only for this tool
    python possession/kit_check.py

Needs data/tactical/BAR-ATH_tactical.mp4 (or --video). Takes a few minutes: it seeks.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for path in (HERE, ROOT, os.path.join(ROOT, "starter")):
    if path not in sys.path:
        sys.path.insert(0, path)

import atd  # noqa: E402
import features  # noqa: E402
import load_homography  # noqa: E402
from games import video_path  # noqa: E402

SAMPLE = 120
CONTACT = {"keeper": 3.0, "outfield": 1.5}
TEAL_SHARE = 0.05       # goalkeeper: teal share above this is Athletic's keeper
WHITE_SHARE = 0.40      # outfield: white share above this is an Athletic shirt
SEED = 0


def torso_hsv(cv2, image, matrix, x_norm, y_norm, length):
    """HSV pixels of the torso of a player standing at a pitch position, or None if off-frame."""
    def project(x, y):
        p = matrix @ np.array([x, y, 1.0])
        return p[0] / p[2], p[1] / p[2]
    u, v = project(x_norm, y_norm)
    u2, v2 = project(x_norm + 1.0 / length, y_norm)
    height = 1.8 * np.hypot(u2 - u, v2 - v)            # a player is ~1.8 m tall
    y0, y1 = int(v - 0.75 * height), int(v - 0.40 * height)
    x0, x1 = int(u - 0.18 * height), int(u + 0.18 * height)
    if y0 < 0 or x0 < 0 or y1 > image.shape[0] or x1 > image.shape[1] or y1 <= y0 or x1 <= x0:
        return None
    return cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)


def kit_team(hsv, keeper):
    """0 Barcelona, 1 Athletic, None if the patch is too small to say."""
    if keeper:
        teal = (hsv[:, 0] >= 70) & (hsv[:, 0] <= 100) & (hsv[:, 1] > 60) & (hsv[:, 2] > 70)
        return int(np.mean(teal) >= TEAL_SHARE)
    grass = (hsv[:, 0] >= 30) & (hsv[:, 0] <= 65) & (hsv[:, 1] > 50)
    body = hsv[~grass]
    if len(body) < 8:
        return None
    white = (body[:, 1] < 60) & (body[:, 2] > 150)
    return int(np.mean(white) > WHITE_SHARE)


def check(kind, data, matrices, capture, cv2, rng):
    length, width = data["pitch"]
    graded = features.graded_mask(len(data["bx"]))
    slots = np.flatnonzero(data["gk"] if kind == "keeper" else ~data["gk"])
    dist = np.hypot(data["px"][:, slots] - data["bx"][:, None],
                    data["py"][:, slots] - data["by"][:, None])
    nearest = np.nan_to_num(dist, nan=99.0)
    frames = np.flatnonzero(graded & (nearest.min(axis=1) < CONTACT[kind]))
    frames = np.sort(rng.choice(frames, min(SAMPLE, len(frames)), replace=False))
    corrected = features.keeper_teams(data["px"], data["team"], data["gk"], data["pitch"],
                                      data["attack_first_half"])
    rows = []
    for frame in frames:
        slot = slots[int(np.argmin(nearest[frame]))]
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
        ok, image = capture.read()
        if not ok:
            continue
        hsv = torso_hsv(cv2, image, matrices[frame], data["px"][frame, slot] / length,
                        data["py"][frame, slot] / width, length)
        team = None if hsv is None else kit_team(hsv, kind == "keeper")
        if team is not None:
            rows.append((data["team"][slot], corrected[frame, slot], team))
    rows = np.array(rows)
    meta = np.mean(rows[:, 0] == rows[:, 2])
    fixed = np.mean(rows[:, 1] == rows[:, 2])
    print("{:<9} {:3d} on-ball frames: metadata team matches the kit {:.3f}, keeper-corrected "
          "team {:.3f}".format(kind, len(rows), meta, fixed))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", default=video_path())
    args = parser.parse_args()
    try:
        import cv2
    except ImportError:
        sys.exit("needs OpenCV:  pip install opencv-python-headless==5.0.0.93")
    if not os.path.isfile(args.video):
        sys.exit("tactical video not found at {}; pass --video".format(args.video))
    data = atd.load()
    matrices, _, _ = load_homography.load()
    capture = cv2.VideoCapture(args.video)
    rng = np.random.RandomState(SEED)
    for kind in ("keeper", "outfield"):
        check(kind, data, matrices, capture, cv2, rng)


if __name__ == "__main__":
    main()
