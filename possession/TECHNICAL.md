# Ball possession as a hidden state — technical document

**More than a Hack 2027, Phase 1.** One label per frame of BAR-ATH: `H`, `A`, `D` (or `X`).

> Rename this file `<team_name>.md` (or export to `<team_name>.pdf`) before submitting, and the
> CSV to `<team_name>.csv` — see SUBMISSION.md.

---

## 1. The approach in one paragraph

Possession is treated as what the challenge defines it to be: a **state** that persists through
passes, loose balls and missing tracking until the other team gains *control*, and that ends at
every stoppage. That is a hidden Markov model with three states — home, away, dead — decoded
with Viterbi over each half. Every frame contributes evidence; every change of state costs a
penalty; a stoppage lasts at least 2 s. The two kinds of evidence (is the ball alive? who is
near it?) are each **learned from the match itself without any labels**: the alive/dead model
by hard EM over the full match, the proximity model by self-training on the model's own decode.
Two rules from the Laws of the Game are imposed as hard constraints: no half is in play before
its kick-off (Law 8), and a ball that has left the pitch is out of play (Law 9).

No hand labels, no external labelled data, no ground truth was used to fit or choose anything.

## 2. Why this rather than something else

The baseline's failure is structural, not a tuning problem: a per-frame vote cannot represent a
possession, so it flickers (too many possessions) or, smoothed hard, blurs every boundary and
swallows the short ones. Two observations about the data shaped the design:

* **The missing ball is mostly dead time.** Graded play has only 150 missing-ball gaps, almost
  all longer than 2 s, and the ball is invisible 79–94 % of the time when players move under
  1 m/s but ~10 % when they run. So "ball missing" is largely a symptom of a stoppage, and
  dead-ball detection — which splits possessions and therefore dominates `spellF1` — is worth
  modelling properly rather than leaving as "no ball → D".
* **Possession changes on contact, and is held between contacts.** Reading the public window
  for *semantics* (which CHALLENGE.md invites), the ground truth switches roughly when the new
  side gets the ball close and keeps the old side through flight and loose balls — exactly a
  switching cost against accumulated contact evidence.

An HMM is the smallest model that says both things, it is exactly solvable, and its parameters
are interpretable — which matters when they cannot be tuned against labels.

## 3. Assets used, and not used

| asset | used? | how |
|---|---|---|
| tactical ATD — ball | **yes** | visibility, speed, position vs lines, distance to each side |
| tactical ATD — players | **yes** | collective speed and acceleration; nearest player per side |
| ATD metadata | **yes** | slot → team; no player identities are needed or used |
| Smart Tagging `SET PIECES` | **yes**, as evidence | inside-a-set-piece-window feature; kick-off windows for Law 8 |
| Smart Tagging phase codes | no | the defending codes mirror the attacking ones exactly (same counts and durations), and phases are not possessions; the team evidence from tracking is far denser |
| homography | no | only needed for video work, which this does not do |
| videos | no | — |
| broadcast ATD | no | unsynchronised; see §8 |

Speeds in the ATD are already in m/s despite the XML saying kph (verified: the speed derived
from positions matches the raw value 1 : 1).

## 4. The model

### 4.1 Alive vs dead — `deadball.py`

Per-frame categorical features, binned for physical meaning and grouped into **views** so that
naive Bayes counts each source of evidence once (ball visibility would otherwise be counted by
three features and the model concludes a dead ball is never visible — wrong for throw-ins):

* **ball view** — missing: how long the gap is; visible: speed × margin to the nearest line ×
  distance to the nearest player, jointly;
* **players view** — collective speed (±0.5 s) × its change over ±1 s, jointly;
* **tagging view** — inside a Smart Tagging set-piece window or not.

Fitted by **hard EM over the whole match**: a crude seed split (players walking, or a long gap
while slow, or the ball over a line) → class histograms with Laplace smoothing → tempered
log-likelihood ratio (×1/3, for correlated views) → two-state Viterbi with minimum durations
(dead ≥ 2 s, alive ≥ 1 s) → refit, until the split stops changing (8 iterations).

**Event-gated transitions.** A stoppage starts at an event and ends at one, so the switching
penalty is made time-varying. For each frame, a *stop code* (ball crossed a line nearby; ball
vanishes into a ≥1 s gap within 2 s; collective braking) and a *restart code* (ball struck from
rest; ball reappears; collective acceleration; inside a set-piece window) are computed, and EM
learns `log P(code | transition here) − log P(code)` from the decoded transitions. These average
to zero, so the prior on how many stoppages a match has stays calibrated. What EM learned,
without labels, is the Laws of the Game: a stop is likeliest when the ball crosses a line and
vanishes while players brake (+3.6 nats), a restart when the ball is struck from rest (+3.2 to
+4.3), and a restart with no kick, no reappearance and no acceleration is all but ruled out
(−3.8).

### 4.2 Home vs away — `model.py`

Joint three-state Viterbi with minimum durations (possession ≥ 10 frames, dead ≥ 2 s), the
dead-ball log-ratio and event bonuses from 4.1, and **zero-sum team evidence**: +w/2 to the side
nearest the ball, −w/2 to the other, so team evidence never moves the alive/dead decision.

* **What w is worth** is learned by self-training: the per-bin agreement between "nearest side"
  and the decoded possession, over (distance to ball × margin to the nearest opponent × ball
  speed), converted to log-odds and floored at zero. The result is monotone and physical —
  strongest at the feet with a clear margin, decaying with distance, and **zero for a ball over
  15 m/s** (a struck ball says nothing about who has it).
* **How hot the evidence is** — one scalar temperature — is set by **self-consistency with the
  one prior the model holds**: possessions average `MEAN_POSSESSION_S = 12 s` of live play.
  Bisection finds the temperature at which the decoded full match agrees with that assumption.
  The target is a football prior, not a statistic of this game's labels.

### 4.3 Laws of the Game as constraints

* **Law 8** — every graded frame of a half before its tagged kick-off window is dead (warm-ups
  look like play: before this, the 2,934 pre-kick-off frames of the second half were only 40 %
  dead). The decoder still chooses the kick-off frame inside the window.
* **Law 9** — a ball that leaves the pitch *continuously* (no position jump > 1.6 m/frame, which
  is how a spare ball by the touchline appears), stays > 1 m out for ≥ 5 frames, is dead until
  1 s after it is back in view; over a goal line, until it has been seen at a corner or in a goal
  area (the only places the restart can be taken), capped at 60 s. 33 exits qualify.

## 5. Possession rules implemented

| situation | rule |
|---|---|
| pass between team-mates, ball in flight | no evidence above 15 m/s; possession persists |
| touch / deflection / lost duel | a few frames of contact cannot pay the switching penalty — no change |
| controlled reception by the other side | sustained contact pays it — change, placed where the evidence changes |
| loose or contested ball | small margin → little evidence; possession stays with the last side |
| ball missing in play | no evidence; possession persists (HMM) |
| stoppage | dead state; event-gated entry/exit; ≥ 2 s |
| restart | dead → whichever side's contact evidence follows the restart |
| dead-ball ownership | not modelled — the answer is `D`, as the challenge asks |
| outside the graded halves | `X` |

## 6. Training data

**None.** No labelled data of any kind, from this game or another. Every fitted quantity —
dead-ball histograms, event bonuses, proximity table, temperature — is estimated from the
provided game's tracking by EM or self-training, i.e. from algorithmic pseudo-labels with no
person deciding any of them (RULES.md: allowed).

**How the public 5 minutes were used.** Only as CHALLENGE.md and RULES.md permit: to read what
the labels mean, and as a wiring check after a version was built. No parameter was tuned on it
and no version was chosen by its score. In the interest of full disclosure, two design changes
were prompted by *inspecting failures* inside it — late dead-ball onsets (→ event-gated
transitions, §4.1) and a goal-line exit called live (→ the Law 9 constraint, §4.3). Both are
general mechanisms applied over the whole match, and both were judged on the ground-truth-free
checks of §7, not on the public score.

## 7. How it was evaluated without ground truth

The public window has 17 possessions; the scorer's own bootstrap puts its `spellF1` noise band
at ±0.36. It cannot choose between versions. Instead, on the **full match**:

* **Restart check.** Every Smart Tagging corner, goal kick, throw-in and kick-off is a restart.
  Final model: a decoded restart inside **54 / 59** tag windows (throw-ins 26 / 26, corners
  12 / 12). "In place" (ball seen at the right spot at the restart) is 30 / 59, limited mostly by
  the ball being untracked at the kick — corners in particular.
* **Plausibility.** 287 possessions, mean 12.0 s, 12 shorter than 2 s; 144 stoppages, 45.9 %
  dead; home share 0.605.
* **Sensitivity to the one prior** (`--mean-possession`): 10 s → 350 possessions, 12 s → 287
  (shipped), 14 s → 248. The dead-ball model is unaffected; nothing falls off a cliff.

Public window, for wiring only — not a ranking signal:

| | macroF1 | spellF1 | F1 H / A / D | alive/dead F1 | possessions (truth 17) |
|---|---|---|---|---|---|
| this model | 0.847 | 0.476 | 0.852 / 0.857 / 0.834 | 0.963 | 20 |
| baseline_smooth50 | 0.621 | 0.326 | 0.778 / 0.762 / 0.323 | 0.899 | 26 |

## 8. Limitations, and what to do with more time

* **Short possessions** (< ~2 s) are mostly missed — the price of not inventing spurious ones.
* **Identity switches across teams** in the tracking produce confident wrong contact evidence;
  the slots carry a fixed team, so a tracklet that jumps to an opponent is undetectable without
  the video. *Fix:* jersey-colour check of each slot on the tactical video.
* **The ball is untracked at many restarts** (corners, kick-offs), so restart timing falls back
  on collective acceleration. *Fix:* the broadcast ATD is a second, independent ball track; it
  can be synchronised to the tactical timeline by matching player configurations (both are in
  pitch coordinates), and would fill gaps where the tactical camera loses the ball.
* **Restart team** comes only from post-restart contact. *Fix:* last-touch rule at the exit
  (throw-ins, corners, goal kicks go to the opponent of the last toucher) and the tagged team of
  corners, goal kicks and free kicks.
* **Kick-offs after goals** are slow and often untracked; one of the four decoded stoppages
  overruns its kick-off by ~14 s.
* The possession-length prior (12 s) is the single assumption everything else is calibrated
  to; it comes from football, not from this match, and §7 shows its effect.

## 9. Reproduction

**numpy and pandas only**, pinned in `possession/requirements.txt`; produced and verified on
Python 3.11. With the data bundle in `data/`:

```bash
pip install -r possession/requirements.txt
python possession/run.py --name <team_name>            # ~80 s; writes evaluation/results/<team_name>.csv
python evaluation/validate_submission.py evaluation/results/<team_name>.csv
python possession/run.py --name <team_name> --check    # optional: public 5-minute wiring check
```

Deterministic: no randomness anywhere, and two runs from a cold cache give byte-identical files.
The first run caches the parsed tracking under `data/cache/` (gitignored — it is derived from
the gated data and must not be committed).

| file | role |
|---|---|
| `possession/atd.py` | raw tracking → arrays in metres, cached |
| `possession/features.py` | per-frame features |
| `possession/hmm.py` | Viterbi with minimum durations and time-varying penalties (numpy) |
| `possession/deadball.py` | alive/dead model, hard EM, event bonuses |
| `possession/model.py` | three-state model, self-trained proximity evidence, calibration, Laws 8–9 |
| `possession/diagnostics.py` | ground-truth-free checks |
| `possession/run.py` | the one command |
