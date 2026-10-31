# Ball possession as a hidden state: technical document

**More than a Hack 2027, Phase 1.** One label per frame of BAR-ATH: `H`, `A`, `D` (or `X`).

> Rename this file `<team_name>.md` (or export to `<team_name>.pdf`) before submitting, and the
> CSV to `<team_name>.csv`. See SUBMISSION.md.

---

## 1. The approach in one paragraph

Possession is treated as what the challenge defines it to be: a **state** that persists through
passes, loose balls and missing tracking until the other team gains *control*, and that ends at
every stoppage. That is a hidden Markov model with three states (home, away, dead) decoded
with Viterbi over each half. Every frame contributes evidence; every change of state costs a
penalty; a stoppage lasts at least 3 s. The two kinds of evidence (is the ball alive? who is
near it?) are each **learned from the match itself without any labels**: the alive/dead model
by hard EM over the full match, the proximity model by self-training on the model's own decode.
Two rules from the Laws of the Game are imposed as hard constraints: no half is in play before
its kick-off, which is taken from the centre mark (Law 8), and a ball that has left the pitch is
out of play (Law 9). The tagged side of a set piece takes the restart (soft evidence). The
broadcast recording, aligned to the tactical timeline from the tracking alone, supplies a second
ball track for the possession model. Goalkeeper slots, which the tracker fills by position rather
than by team, are credited to the side whose goal they stand in, a fix found by overlaying the
tracking on the video and verified there against the keepers' kits.

On two other professional matches, with possessions rebuilt from their event data, the same
pipeline scores **0.60 spellF1**, against 0.44 for the organisers' baseline method (§7).

No hand labels, no external labelled data, no ground truth was used to fit or choose anything.

## 2. Why this rather than something else

The baseline's failure is structural, not a tuning problem: a per-frame vote cannot represent a
possession, so it flickers (too many possessions) or, smoothed hard, blurs every boundary and
swallows the short ones. Two observations about the data shaped the design:

* **The missing ball is mostly dead time.** Graded play has only 150 missing-ball gaps, almost
  all longer than 2 s, and the ball is invisible 79–94 % of the time when players move under
  1 m/s but ~10 % when they run. So "ball missing" is largely a symptom of a stoppage, and
  dead-ball detection, which splits possessions and therefore dominates `spellF1`, is worth
  modelling properly rather than leaving as "no ball → D".
* **Possession changes on contact, and is held between contacts.** Reading the public window
  for *semantics* (which CHALLENGE.md invites), the ground truth switches roughly when the new
  side gets the ball close and keeps the old side through flight and loose balls: exactly a
  switching cost against accumulated contact evidence.

An HMM is the smallest model that says both things, it is exactly solvable, and its parameters
are interpretable, which matters when they cannot be tuned against labels.

## 3. Assets used, and not used

| asset | used? | how |
|---|---|---|
| tactical ATD: ball | **yes** | visibility, speed, position vs lines, distance to each side |
| tactical ATD: players | **yes** | collective speed and acceleration; nearest player per side |
| ATD metadata | **yes** | slot → team; no player identities are needed or used |
| Smart Tagging `SET PIECES` | **yes**, as evidence | inside-a-set-piece-window feature; kick-off windows for Law 8 |
| Smart Tagging phase codes | no | the defending codes mirror the attacking ones exactly (same counts and durations), and phases are not possessions; the team evidence from tracking is far denser |
| homography | for validation | projects tracked players into the video for the kit check (§4.6) |
| tactical video | for validation only | the kit check (§4.6, `kit_check.py`); the pipeline never reads it |
| broadcast video | no | not needed |
| other games: Metrica's two open sample matches | for validation only | labelled possessions to test the method and choose two settings (§6, §7.1) |
| broadcast ATD | **yes**, for the team model | aligned to the tactical timeline from the tracking alone (§4.5); its ball fills frames the tactical camera misses |

Speeds in the ATD are already in m/s despite the XML saying kph (verified: the speed derived
from positions matches the raw value 1 : 1).

## 4. The model

### 4.1 Alive vs dead (`deadball.py`)

Per-frame categorical features, binned for physical meaning and grouped into **views** so that
naive Bayes counts each source of evidence once (ball visibility would otherwise be counted by
three features and the model concludes a dead ball is never visible, which is wrong for throw-ins):

* **ball view.** If missing, how long the gap is; if visible, speed × margin to the nearest line ×
  distance to the nearest player, jointly;
* **players view.** Collective speed (±0.5 s) × its change over ±1 s, jointly;
* **tagging view.** Inside a Smart Tagging set-piece window or not.

Fitted by **hard EM over the whole match**: a crude seed split (players walking, or a long gap
while slow, or the ball over a line) → class histograms with Laplace smoothing → tempered
log-likelihood ratio (×1/3, for correlated views) → two-state Viterbi with minimum durations
(dead ≥ 3 s, alive ≥ 1 s) → refit, until the split stops changing (8 iterations).

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

### 4.2 Home vs away (`model.py`)

Joint three-state Viterbi with minimum durations (possession ≥ 10 frames, dead ≥ 3 s), the
dead-ball log-ratio and event bonuses from 4.1, and **zero-sum team evidence**: +w/2 to the side
nearest the ball, −w/2 to the other, so team evidence never moves the alive/dead decision.

* **What w is worth** is learned by self-training: the per-bin agreement between "nearest side"
  and the decoded possession, over (distance to ball × margin to the nearest opponent × ball
  speed), converted to log-odds and floored at zero. The result is monotone and physical:
  strongest at the feet with a clear margin, decaying with distance, and **zero for a ball over
  15 m/s** (a struck ball says nothing about who has it).
* **How hot the evidence is** (one scalar temperature) is set by **self-consistency with the
  one prior the model holds**: possessions average `MEAN_POSSESSION_S = 12 s` of live play.
  Bisection finds the temperature at which the decoded full match agrees with that assumption.
  The target is a football prior, not a statistic of this game's labels. Checked afterwards
  on two other matches (§7): 12 s and 14 s score the same, 16 s is clearly worse, and those
  matches' own measured mean (10.8 s) scores slightly lower, so 12 s stays.

### 4.3 Laws of the Game as constraints

* **Law 8.** Every graded frame of a half before its tagged kick-off window is dead (warm-ups
  look like play: before this, the 2,934 pre-kick-off frames of the second half were only 40 %
  dead). The decoder still chooses the kick-off frame inside the window.
* **Law 9.** A ball that leaves the pitch *continuously* (no position jump > 1.6 m/frame, which
  is how a spare ball by the touchline appears), stays > 1 m out for ≥ 5 frames, is dead until
  1 s after it is back in view; over a goal line, until it has been seen at a corner or in a goal
  area (the only places the restart can be taken), capped at 60 s. 33 exits qualify.
* **Law 8, the centre mark.** Inside a kick-off window the restart may only happen with the
  ball at the centre mark (within 4 m) or unseen, so a ball being rolled back to the spot no
  longer reads as the kick-off; and a tagged kick-off has happened by the end of its window
  (dead is ruled out for the 5 s after it). The second-half kick-off now decodes 5 frames from
  the published one, where it was 55 frames early.

### 4.4 Who restarts

CHALLENGE.md: "whenever play restarts after a stoppage, there is a clear possessor". Two pieces
of evidence name it, and both enter the same way.

* **The tag.** The tag of a corner, goal kick or free kick names the side taking it, confirmed
  from geometry alone: every tagged corner sits at the tagged side's attacking end, every goal
  kick at its own. Log-odds `log(0.95 / 0.05)`, from 2 s before the tag window opens to 18 s
  after (where 90 % of first restarts fall).
* **The taker.** Whoever takes a set piece stands on the ball just before it, so the side with
  clear contact in the last 2 s before a restart restarts. Checked against the tags it was right
  13 times in 14 (the other 10 had nobody tracked near the ball); log-odds are the
  Laplace-smoothed record, `log(14 / 2)`. It covers throw-ins, which are never tagged.

The summed log-odds `r` enter the transition as a factorised probability:
`P(dead → side at t) = P(restart at t) · P(side | t)`, i.e. bonuses `log 2σ(r)` for home and
`log 2σ(−r)` for away relative to the 50/50 split already in the transition matrix. The two
sides' probabilities still sum to one, so this evidence decides **who** restarts, never
**whether**. A first version that simply added `±r` also made restarting cheaper wherever a
player stood on a dead ball, and invented sub-second "restarts" inside stoppages.

It acts on the restart only: if the other side then gains sustained control, possession still
changes. Restarts whose decoded side contradicts the taker fell from 21 to 14 of 98, and are 8 of 92 in
the final model. The decoded side agrees with the tag on 20 of 24; most of the others
are corners after which the defending side then holds the ball for 5–17 s, i.e. a clearance
kept, where skipping a ~1 s attacking spell costs little.

The last-touch rule (Law 9: the restart goes to the opponents of the last player to touch the
ball) was tried as further evidence and **dropped**: the nearest player at the moment the ball
leaves the frame agreed with the tag on only 3 of 5 checkable exits.

### 4.5 A second camera (`broadcast.py`)

The broadcast recording has its own automatic tracking, in the same pitch coordinates, on its
own timeline, and no mapping is provided. Matching the broadcast's visible players against every
tactical frame (chamfer distance over a one-second clip, coarse grid then refined to the frame)
shows that live TV sits at **one constant offset per half**: +4014 frames in the first,
+26920 in the second. A TV feed plays replays over live time rather than pausing it, so the
offset never drifts; replay and close-up frames simply fail to match, and every frame is
verified individually (mean player-to-nearest-player distance under 1.2 m; true matches measure
0.2–0.9 m, the best wrong frame anywhere ≥ 1.8 m). 47 % of frames verify.

The offsets are confirmed by an independent signal: agreement between the two ball tracks is
sharpest at exactly these offsets (median 0.74 m), and worsens by ~0.1 m per frame either side.

Use, for the **possession model's** contact evidence:

* **the ball**, wherever only the broadcast sees it: 5,658 frames, a third of the live-play
  frames the tactical camera loses the ball in;
* **the players the tactical tracker missed**: a broadcast player with no tactical player
  within 1 m is someone the wide camera's tracker lost (it sees ~20.5 of 22); 132,915 such
  observations are appended as extra slots with the broadcast's team labels.

Phase-team agreement (§7) went 0.699 → 0.703 with the ball and → 0.706 with the players.

The alive/dead model keeps the tactical ball only: for it "the ball is visible" is a
signal in itself, and the broadcast sees the ball through stoppages the wide camera loses it in.
Using it there too (tested) changed twice as many frames with no measurable gain.

### 4.6 Goalkeeper slots, and what the video says

Overlaying the tracking on the tactical video (homography projection) showed the tracker filing
a goalkeeper under a goalkeeper *slot* by where he stands rather than which side he plays for:
the slot labelled Athletic spends 10–13 % of its time at Barcelona's goal, in runs of up to
230 s, and the Barcelona slot does the mirror error less often: 953 ball contacts credited to
the wrong side at full weight, goal kicks included.

The fix needs no video: **a goalkeeper defends his own goal**, so each goalkeeper slot is
credited, frame by frame, to the side whose goal is nearer, using each side's attacking direction
from the metadata (swapped at halftime), in `features.keeper_teams`, applied to both trackers.

Measured on the video with `kit_check.py`: the two keepers wear different greens (lime and
teal; grass never reaches the teal hue), and the outfield kits are dark and white. On 120 random
frames with a keeper on the ball, the metadata team matches his kit **78 %** of the time, the
corrected team **97.5 %** (on 60 frames where the two labellings disagree: corrected right 58,
metadata right 2). Outfield slots match their kit 98.3 % of the time, so there is no comparable
error to fix there and the pipeline does not read the video at all.

Effect: restarts whose decoded side contradicts the taker fell 14 → 10 of 96; restart side vs
tag 19 → 20 of 24. Phase-team agreement *fell* 0.704 → 0.682; see §7 for why that check
cannot judge this change.

### 4.7 Tried and dropped

Each was measured on the ground-truth-free checks of §7 at the same possession prior, and
dropped for making no difference or making things worse:

* **Abstaining where the two trackers disagree on the team** of the player nearest the ball
  (4.7 % of such frames): inside tagged phases the two trackers are each right about half the
  time (53 % / 47 %), so neither can arbitrate; agreement 0.703 → 0.700.
* **Last-touch rule for the restart side**: the nearest player at the exit agreed with the tag
  on 3 of 5 checkable exits.
* **The broadcast ball in the alive/dead model**: see 4.5.
* **The TV director's cuts as dead-ball evidence.** Frames where the broadcast fails to verify
  are replays and close-ups, which the director cuts to during stoppages: the feed shows the
  live wide shot on 85 % of decoded live frames but 27 % of dead ones. But the signal lags:
  it cuts away 2–4 s after a stop and returns a few seconds before the restart. As a per-frame
  view in the alive/dead model it improved restart geometry (found 55 → 56, in place 33 → 35)
  but created spurious 1–2 s possessions between stoppages (possessions under 2 s 16 → 22);
  as stop/restart event evidence EM gave it little weight (log-ratios within ±0.7) and it
  changed almost nothing. Neither passed "better with no regression".
* **Touch evidence**: whether the ball just changed velocity next to the nearest player, as
  a fourth dimension of the proximity table. On the held-out matches the nearest side 1–2 m
  away is right 94–97 % of the time at a gentle touch (a 3–8 m/s change) against 67–71 %
  otherwise, and spellF1 rose 0.602 → 0.608 and 0.589 → 0.590; on BAR-ATH one fewer tagged
  restart was found. Inside the noise both ways, so not shipped.
* **The decoder's structural settings**, tuned on the held-out matches (§7.1): minimum
  possession 5 or 20 frames instead of 10, switching prior halved or doubled, confidence cap
  0.999 instead of 0.99. Every one landed within ±0.001 of the shipped model: the temperature
  calibration absorbs them, so the model is not balanced on any of these choices.
* **A per-frame jersey-colour check of every outfield slot on the video**: outfield slots
  already match their kit 98.3 % of the time (§4.6); the residual is mostly isolated frames the
  switching penalty absorbs, not worth decoding the video on every run.

## 5. Possession rules implemented

| situation | rule |
|---|---|
| pass between team-mates, ball in flight | no evidence above 15 m/s; possession persists |
| touch / deflection / lost duel | a few frames of contact cannot pay the switching penalty, so no change |
| controlled reception by the other side | sustained contact pays it: change, placed where the evidence changes |
| loose or contested ball | small margin → little evidence; possession stays with the last side |
| ball missing in play | no evidence; possession persists (HMM) |
| stoppage | dead state; event-gated entry/exit; ≥ 3 s |
| restart | the side named by the tag (corners, goal kicks, free kicks) and by who stands on the ball before it (all restarts), as soft evidence on who restarts |
| kick-off | not before the half's tagged window; only from the centre mark; done by the window's end |
| ball over a line | dead (Law 9), until it is back in view (for corners and goal kicks, back at the spot) |
| goalkeeper on the ball | his touches count for the side whose goal he is in, whatever slot the tracker filed him under |
| dead-ball ownership | not modelled; the answer is `D`, as the challenge asks |
| outside the graded halves | `X` |

## 6. Training data

**Nothing from the provided game.** Every fitted quantity (dead-ball histograms, event bonuses,
proximity table, temperature) is estimated from the provided game's tracking by EM or
self-training, i.e. from algorithmic pseudo-labels with no person deciding any of them
(RULES.md: allowed).

**Labels from two other games, for validation.** Metrica Sports' two open sample matches
(github.com/metrica-sports/sample-data, `Sample_Game_1` and `Sample_Game_2`): tracking at 25 fps
and 3,680 events annotated by Metrica Sports. `other_games.py` turns the events into a
possession ground truth under the challenge's definition (control by a pass, shot, set piece or
recovery; duels change nothing; the ball dies at a ball out, foul, offside, handball, goal or
shot out and lives again at the next restart) and scores the unchanged pipeline on them with
the official metric. They were used to validate the method and to choose two settings: the
minimum stoppage (3 s) and the possession prior (12 s), §7. Nothing is trained on them: every
weight is still fitted, without labels, on the match being labelled.

**How the public 5 minutes were used.** Only as CHALLENGE.md and RULES.md permit: to read what
the labels mean, and as a wiring check after a version was built. No parameter was tuned on it
and no version was chosen by its score. In the interest of full disclosure, two design changes
were prompted by *inspecting failures* inside it: late dead-ball onsets (→ event-gated
transitions, §4.1) and a goal-line exit called live (→ the Law 9 constraint, §4.3). Both are
general mechanisms applied over the whole match, and both were judged on the ground-truth-free
checks of §7.2, not on the public score.

**How the video was used.** Watched and measured to inspect the tracking, as RULES.md allows,
never labelled. The goalkeeper rule of §4.6 came from overlaying the tracking on frames; it is a
general rule (a keeper defends his own goal), not tied to any frame, and its check
(`kit_check.py`) is an algorithmic colour measurement anyone can rerun.

## 7. How it was evaluated

The public window has 17 possessions; the scorer's own bootstrap puts its `spellF1` noise band
at ±0.36. It cannot choose between versions. Two other sources can.

### 7.1 On two other matches, with labels

Metrica's two open sample matches (§6), full length, each with about 300 possessions, so a
noise band of roughly ±0.04 per match. The official scorer, the unchanged pipeline:

| | sample game 1 | sample game 2 |
|---|---|---|
| truth: possessions, live s per possession, stoppages | 340, 9.9 s, 86 | 302, 11.8 s, 86 |
| organisers' baseline method (nearest player, ±50 majority) | 0.448 | 0.433 |
| **this model** | **0.602** | **0.589** |

Two settings were chosen here, on mean spellF1 over the two matches:

| minimum stoppage | prior 12 s | prior 14 s | prior 16 s |
|---|---|---|---|
| 2 s (first version) | 0.590 | 0.594 | 0.544 |
| **3 s (shipped)** | **0.595** | 0.599 | 0.561 |

* **3 s minimum stoppage**: better in all six paired comparisons, closer to the true number of
  stoppages (game 2: 112 → 98, truth 86), and consistent with the events themselves, where
  only 5 % of stoppages last under 3.5–3.9 s. It had earlier been rejected on BAR-ATH by the
  phase-team check, which §7.2 shows cannot judge it.
* **Possession prior**: flat from 12 to 14 s, a cliff at 16 s. The 14 s edge is far inside the
  noise and sits nearer the cliff, so 12 s stays. Calibrating to the matches' own measured mean
  (10.8 s) scored lower (0.590 and 0.559 at 2 s): the decoder misses short possessions anyway,
  so it should not be asked for as many as the truth has.

Two honest caveats: this tracking is corrected (no identity switches, fewer gaps), so it is
easier than BAR-ATH's automatic tracking and the absolute scores will not transfer; and the
event-derived truth is timed by hand, a few frames from what a corrected tracking would mark.
What transfers is the comparison: the method beats the baseline method by ~0.15 on two matches
it never saw.

**Where the remaining errors are.** About 80 % of the true possessions the model misses are
shorter than 5 s (found at IoU 0.5: 5–20 % of those under 2 s, 42–51 % at 2–5 s, 73–87 % above
5 s). The missed ones have a median of 3–4 frames with their own side clearly on the ball,
against 25–30 for the short ones it does find: they are single touches, an interception passed
or cleared at once, which the annotators counted as a recovery. CHALLENGE.md says a touch
without control is no change of possession, so chasing these would fit the annotators'
convention rather than the competition's definition. Stoppage timing is already exact on this
tracking (median offset 0 s at both stops and restarts, interquartile range within ±0.3 s); the
extra decoded stoppages (about 15 per match) look like real ones, ball unseen and players
walking for 7–16 s, and are most likely stoppages the events do not mark.

**Robustness to BAR-ATH's ball loss** (`--ball-dropout`). This tracking almost never loses the
ball in live play; BAR-ATH's tactical camera does on 10 % of live frames, 6.4 % after adding the
broadcast (§4.5). Removing the ball at random from that share of live frames, in 1–5 s gaps:

| ball removed from live frames | sample game 1 | sample game 2 | cost |
|---|---|---|---|
| 0 % | 0.602 | 0.589 | |
| 6.4 % (BAR-ATH with the broadcast) | 0.586 | 0.576 | 0.014 |
| 10 % (BAR-ATH's tactical camera alone) | 0.582 | 0.564 | 0.022 |

So a perfect way of filling ball gaps could win back about 0.014 at most, inside the noise, and
the broadcast fusion already recovered roughly a third of the cost.

### 7.2 On the provided match, without ground truth

* **Restart check.** Every Smart Tagging corner, goal kick, throw-in and kick-off is a restart.
  Final model: a decoded restart inside **55 / 59** tag windows (throw-ins 26 / 26, corners
  12 / 12). "In place" (ball seen at the right spot at the restart) is 34 / 59, limited mostly by
  the ball being untracked at the kick, corners in particular. This is geometry, not tags, so
  it is not circular even though the tagged side is used as evidence (§4.4).
* **Phase-team agreement.** The share of decoded live frames inside Smart Tagging attacking
  phases (Build Up, Progression, Ball in Final Third, …) that give the tagged side: **0.682**,
  against 0.62–0.64 for the baselines. Phases are not possessions, so this is never 1.0, and it
  rises with smoother decodes (0.661 at a 10 s prior, 0.702 at 14 s), so it is only used to
  compare variants at the *same* prior, never to choose the prior. (Submissions that use the
  phase tags as input, like a tag-filled vote, score high on it by construction.) Final: 0.681.
  **It also inherits the tracking's team errors**: Smart Tagging is generated from the same
  tracking, so a change that *corrects* team labels can lower it. The goalkeeper fix took it from
  0.704 to 0.682 while the video shows that fix right 58 times in 60 (§4.6). It is therefore used
  only for changes that add observations without relabelling anyone (§4.5), and the kit check
  judges relabelling.
* **Plausibility.** 289 possessions, mean 11.9 s, 21 shorter than 2 s; 139 stoppages, 45.7 %
  dead; home share 0.602. (Real matches have about 60 possessions under 2 s each: 61 and 62 in
  the sample games. A few short possessions are not a warning sign.)
* **Sensitivity to the one prior** (`--mean-possession`): 10 s → 350 possessions, 12 s → 289
  (shipped), 14 s → 246. The dead-ball model is unaffected; nothing falls off a cliff.

Public window, for wiring only (not a ranking signal):

| | macroF1 | spellF1 | F1 H / A / D | alive/dead F1 | possessions (truth 17) |
|---|---|---|---|---|---|
| this model | 0.846 | 0.473 | 0.856 / 0.836 / 0.846 | 0.967 | 16 |
| baseline_smooth50 | 0.621 | 0.326 | 0.778 / 0.762 / 0.323 | 0.899 | 26 |

The public `spellF1` of successive versions moved between 0.40 and 0.48 for changes that left
the full-match checks flat or better: the 17-possession noise the scorer warns about, seen
first-hand. One intermediate version merged a 3 s Athletic possession between two long Barcelona
ones (spellF1 0.400); it was kept anyway because the full-match checks favoured it, and the
next change happened to recover it. Neither move was a reason for anything.

## 8. Limitations, and what to do with more time

* **Short possessions** (< ~2 s) are mostly missed, the price of not inventing spurious ones:
  real matches have about 60 per match, and this model decodes 21 on BAR-ATH. On the labelled
  matches the missed ones are mostly single touches with 3–4 frames of contact (§7.1), the grey
  zone between a touch and control.
* **Identity switches across teams** in the tracking produce confident wrong contact evidence.
  The large, systematic case (goalkeeper slots) is fixed (§4.6); outfield slots still carry
  a ~1.7 % residual that a second tracker cannot arbitrate (§4.7).
* **The ball is untracked at many restarts** (corners, kick-offs), so restart timing falls back
  on collective acceleration; the broadcast adds a ball around 811 restart frames but, being a
  TV feed, often cuts away exactly then.
* **Restart side** is unknown when nobody is tracked near the ball before the restart and the
  set piece is untagged (33 of 137 restarts), and then comes only from post-restart contact.
* **Kick-offs after goals** are slow and often untracked. Law 8 now forces the restart inside
  the tag window, but in two of the four the decoder then calls a second stoppage ~10 s later,
  with the ball still unseen and play slow.
* The possession-length prior (12 s) is the single assumption everything else is calibrated
  to. It comes from football, not from this match; on two other matches it scores as well as
  14 s and better than 10.8 s or 16 s (§7.1).
* **Validation is on corrected tracking.** The two labelled matches are cleaner than BAR-ATH;
  a labelled match with automatic tracking would be the better test.

## 9. Reproduction

**numpy and pandas only**, pinned in `possession/requirements.txt`; produced and verified on
Python 3.11. With the data bundle in `data/`:

```bash
pip install -r possession/requirements.txt
python possession/run.py --name <team_name>            # ~2.5 min cold; writes evaluation/results/<team_name>.csv
python possession/run.py --name <team_name> --out submission --with-doc   # both deliverables: submission/<team_name>.csv and .md
python evaluation/validate_submission.py evaluation/results/<team_name>.csv
python possession/run.py --name <team_name> --check    # optional: public 5-minute wiring check
python possession/run.py --name <team_name> --no-broadcast   # tactical tracking only

# optional, validation only: the kit check of §4.6 (needs the tactical video and OpenCV)
pip install opencv-python-headless==5.0.0.93
python possession/kit_check.py                     # or --video path/to/BAR-ATH_tactical.mp4

# optional: the validation of §7.1 on two other matches (downloads ~125 MB once, ~5 min)
python possession/other_games.py --mean-possession 12 14
python possession/other_games.py --ball-dropout 0.064   # robustness to BAR-ATH's ball loss
```

Deterministic: no randomness anywhere, and two runs from a cold cache give byte-identical files.
The first run caches the parsed tracking and the broadcast mapping under `data/cache/`
(gitignored, since it is derived from the gated data and must not be committed); deleting the cache
reproduces the same file.

| file | role |
|---|---|
| `possession/atd.py` | raw tracking → arrays in metres, cached |
| `possession/features.py` | per-frame features |
| `possession/hmm.py` | Viterbi with minimum durations and time-varying penalties (numpy) |
| `possession/deadball.py` | alive/dead model, hard EM, event bonuses |
| `possession/model.py` | three-state model, self-trained proximity evidence, calibration, Laws 8–9, set-piece priors |
| `possession/broadcast.py` | broadcast ↔ tactical alignment from the tracking; second ball track |
| `possession/other_games.py` | validation on Metrica's two open sample matches: events → possession truth, official scorer |
| `possession/kit_check.py` | optional: team labels vs jersey colours on the video (validation only) |
| `possession/diagnostics.py` | ground-truth-free checks |
| `possession/run.py` | the one command |
