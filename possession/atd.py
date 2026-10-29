"""Fast loader for the raw FIFA/EPTS tracking: the whole file into numpy arrays, cached.

The raw line is "<frame_id>:<x,y,s>;<x,y,s>;...;<x,y,s>:<bx,by,bs>", 46 player groups and one
ball group. Turning every separator into a comma makes each line a flat row of 1 + 46*3 + 3
numbers, which pandas' C parser reads in seconds -- an order of magnitude faster than splitting
199,319 lines in Python.

Everything comes back on the video-frame index (row i = submission row i), in METRES, with the
origin at the corner of the pitch: x in [0, length], y in [0, width]. NaN marks a missing
observation, for players and ball alike.

The cache is written under data/ (gitignored) because it is derived from the gated tracking and
must not be committed or redistributed.
"""

import io
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, DATA_DIR, require, source_frames  # noqa: E402

N_SLOTS = 46
CACHE_VERSION = 1


def team_of_slots(meta_path):
    """0 for each home slot, 1 for each away slot. First <Team> in the file is home (EPTS)."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    slots = [player.get("teamId") for player in root.iter("Player")]
    return np.array([0 if t == team_ids[0] else 1 for t in slots], dtype=np.int8)


def goalkeeper_slots(meta_path):
    """Boolean per slot, True where the metadata calls the slot a goalkeeper."""
    root = ET.parse(meta_path).getroot()
    out = []
    for player in root.iter("Player"):
        kind = None
        for param in player.iter("ProviderParameter"):
            if param.findtext("Name") == "position_type":
                kind = param.findtext("Value")
        out.append(kind == "Goalkeeper")
    return np.array(out, dtype=bool)


def attack_directions(meta_path):
    """{team_index: 'left_to_right' | 'right_to_left'} for the first half, from the metadata."""
    root = ET.parse(meta_path).getroot()
    out = {}
    for index, team in enumerate(root.iter("Team")):
        for param in team.iter("ProviderParameter"):
            if param.findtext("Name") == "attack_direction_first_half":
                out[index] = param.findtext("Value")
    return out


def _parse(raw_path, n_frames, pitch):
    with open(raw_path, "rb") as handle:
        blob = handle.read().replace(b":", b",").replace(b";", b",")
    table = pd.read_csv(io.BytesIO(blob), header=None, dtype=np.float64,
                        na_values=["NaN"], engine="c").values
    del blob
    if table.shape[1] != 1 + N_SLOTS * 3 + 3:
        raise ValueError("unexpected column count {}".format(table.shape[1]))

    frame = table[:, 0].astype(np.int64) - 1          # frame_id counts from 1
    length, width = pitch
    players = table[:, 1:1 + N_SLOTS * 3].reshape(-1, N_SLOTS, 3)
    ball = table[:, 1 + N_SLOTS * 3:]

    px = np.full((n_frames, N_SLOTS), np.nan, np.float32)
    py = np.full((n_frames, N_SLOTS), np.nan, np.float32)
    ps = np.full((n_frames, N_SLOTS), np.nan, np.float32)
    bx = np.full(n_frames, np.nan, np.float32)
    by = np.full(n_frames, np.nan, np.float32)
    px[frame] = players[:, :, 0] * length
    py[frame] = players[:, :, 1] * width
    ps[frame] = players[:, :, 2]                     # already m/s, whatever the XML unit says
    bx[frame] = ball[:, 0] * length
    by[frame] = ball[:, 1] * width
    return px, py, ps, bx, by


def load(game=GAME, data_dir=None, cache=True):
    """dict with px, py, ps (frames x slots), bx, by (frames), team, gk, pitch."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    pitch = GAMES[game]["pitch"]
    cache_path = os.path.join(data_dir or DATA_DIR, "cache",
                              "{}_tactical_atd_v{}.npz".format(game, CACHE_VERSION))

    if cache and os.path.isfile(cache_path) \
            and os.path.getmtime(cache_path) > os.path.getmtime(raw_path):
        arrays = dict(np.load(cache_path))
    else:
        px, py, ps, bx, by = _parse(raw_path, n_frames, pitch)
        arrays = dict(px=px, py=py, ps=ps, bx=bx, by=by)
        if cache:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez(cache_path, **arrays)

    arrays["team"] = team_of_slots(meta_path)
    arrays["gk"] = goalkeeper_slots(meta_path)
    arrays["attack_first_half"] = attack_directions(meta_path)
    arrays["pitch"] = pitch
    return arrays
