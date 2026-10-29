"""Viterbi decoding over a small state space, numpy only.

Possession is a state, not a per-frame vote: it persists through passes, loose balls and missing
tracking until someone else gains control. A hidden Markov model says exactly that. Each frame
contributes log-evidence for each state, every change of state costs a fixed log-penalty, and the
decoder returns the single most probable state sequence. A deflection -- a few frames of contact
by the other side -- cannot pay the switching cost; a controlled reception can.

Durations are enforced by expanding a state into a chain of sub-states (see expand_min_duration):
a state with minimum duration k becomes k copies that must be visited in order, and only the last
one may stay or leave. That turns "a stoppage lasts at least three seconds" into a hard constraint
rather than a post-hoc filter, so the decoder can trade it off against the evidence.
"""

import numpy as np


def viterbi(log_emission, log_transition, log_initial=None):
    """Most probable path. log_emission is (T, S), log_transition (S, S) from-row to-column."""
    n_frames, n_states = log_emission.shape
    if log_initial is None:
        log_initial = np.zeros(n_states)
    back = np.empty((n_frames, n_states), dtype=np.int32)
    score = log_initial + log_emission[0]
    back[0] = -1
    trans = log_transition
    for t in range(1, n_frames):
        candidates = score[:, None] + trans           # (from, to)
        best_from = candidates.argmax(axis=0)
        back[t] = best_from
        score = candidates[best_from, np.arange(n_states)] + log_emission[t]
    path = np.empty(n_frames, dtype=np.int32)
    path[-1] = int(score.argmax())
    for t in range(n_frames - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def expand_min_duration(log_emission, log_transition, min_duration):
    """Chain-expand each state s into min_duration[s] sub-states.

    Returns (expanded_emission, expanded_transition, owner) where owner maps each expanded state
    back to its original state. Entering state s always lands on its first sub-state; sub-state
    i < k-1 must move to i+1; only the last sub-state may self-loop or leave, with the original
    transition penalties.
    """
    n_states = log_emission.shape[1]
    owner, first, last = [], [], []
    for s in range(n_states):
        k = max(1, int(min_duration[s]))
        first.append(len(owner))
        owner.extend([s] * k)
        last.append(len(owner) - 1)
    owner = np.array(owner)
    n_expanded = len(owner)

    trans = np.full((n_expanded, n_expanded), -np.inf)
    for s in range(n_states):
        for i in range(first[s], last[s]):
            trans[i, i + 1] = 0.0                      # forced march through the chain
        trans[last[s], last[s]] = log_transition[s, s]
        for r in range(n_states):
            if r != s:
                trans[last[s], first[r]] = log_transition[s, r]
    return log_emission[:, owner], trans, owner


def viterbi_chain(log_emission, log_transition, min_duration, switch_bonus=None):
    """Viterbi with minimum durations, without materialising the dense expanded matrix.

    Same result as viterbi(*expand_min_duration(...)) but O(T * total_substates) rather than
    O(T * total_substates^2): inside a chain the only move is a shift by one.

    switch_bonus, if given, is (T, S, S): a log-bonus added to the transition from state s to
    state r when it happens AT frame t (i.e. t is the first frame of r). It makes switching
    penalties time-varying, so that a change can be cheap where the data shows an event and
    expensive where it does not.
    """
    n_frames, n_states = log_emission.shape
    ks = [max(1, int(k)) for k in min_duration]
    offsets = np.concatenate(([0], np.cumsum(ks)))
    n_expanded = int(offsets[-1])
    owner = np.repeat(np.arange(n_states), ks)
    last = offsets[1:] - 1
    first = offsets[:-1]

    # back pointers: for a chain-interior sub-state the predecessor is fixed (i-1), so only the
    # first sub-state of each chain and the last (self-loop vs arrival) need storing.
    back_first = np.empty((n_frames, n_states), dtype=np.int16)   # which state we came from
    back_last = np.empty((n_frames, n_states), dtype=np.bool_)    # True: self-loop
    score = np.full(n_expanded, -np.inf)
    score[first] = log_emission[0]
    back_first[0] = -1
    back_last[0] = False

    stay = np.diag(log_transition).copy()
    switch = log_transition.copy()
    np.fill_diagonal(switch, -np.inf)
    for t in range(1, n_frames):
        end = score[last]                                          # (S,)
        enter = end[:, None] + switch                              # (from, to)
        if switch_bonus is not None:
            enter = enter + switch_bonus[t]
        from_state = enter.argmax(axis=0)
        enter_best = enter[from_state, np.arange(n_states)]

        new = np.empty(n_expanded)
        new[1:] = score[:-1]                                       # shift within chains
        loop = end + stay
        arrive = new[last]                                         # from sub-state k-2
        single = np.array(ks) == 1
        arrive = np.where(single, -np.inf, arrive)
        take_loop = loop >= arrive
        new[last] = np.where(take_loop, loop, arrive)
        new[first] = np.where(single, np.maximum(enter_best, loop), enter_best)
        take_loop = np.where(single, loop >= enter_best, take_loop)
        back_first[t] = from_state
        back_last[t] = take_loop
        new += log_emission[t][owner]
        score = new

    # backtrack
    path = np.empty(n_frames, dtype=np.int32)
    j = int(score[last].argmax())
    state, sub = j, ks[j] - 1
    for t in range(n_frames - 1, -1, -1):
        path[t] = state
        if t == 0:
            break
        if sub == ks[state] - 1 and back_last[t, state]:
            continue                                               # self-loop, same sub-state
        if sub == 0:
            state = int(back_first[t, state])
            sub = ks[state] - 1
        else:
            sub -= 1
    return path


def viterbi_duration(log_emission, log_transition, min_duration, switch_bonus=None):
    """Exactly viterbi_chain's optimum, in duration-explicit form -- several times faster.

    A state s with minimum duration k is either *entered* at t (its first frame is t) or
    *settled* at t (it has lasted at least k frames). With C_s the cumulative emission of s:

        enter_s(t)  = max_r settled_r(t-1) + trans(r, s) + bonus_t(r, s)  +  e_s(t)
        settled_s(t) = max( settled_s(t-1) + trans(s, s) + e_s(t),
                            enter_s(t-k+1) + C_s(t) - C_s(t-k+1) )

    so each frame costs O(S^2) scalar work regardless of k, instead of a pass over every
    sub-state of every chain. The path starts in an entered state and must end settled, the
    same boundary conditions as viterbi_chain.
    """
    n_frames, n_states = log_emission.shape
    ks = [max(1, int(k)) for k in min_duration]
    cum = np.vstack([np.zeros((1, n_states)), np.cumsum(log_emission, axis=0)])
    emit = log_emission.tolist()
    stay = [float(log_transition[s, s]) for s in range(n_states)]
    trans = log_transition.tolist()
    bonus = switch_bonus
    neg = -np.inf

    enter = np.full((n_frames, n_states), neg)
    enter_from = np.full((n_frames, n_states), -1, dtype=np.int16)
    settled_prev = [neg] * n_states
    loop_flag = np.zeros((n_frames, n_states), dtype=np.bool_)
    settled_hist = np.full((n_frames, n_states), neg)

    for t in range(n_frames):
        e = emit[t]
        row_enter = [neg] * n_states
        if t == 0:
            row_enter = list(e)
        else:
            b = bonus[t] if bonus is not None else None
            for r in range(n_states):
                best, arg = neg, -1
                for q in range(n_states):
                    if q == r:
                        continue
                    v = settled_prev[q] + trans[q][r]
                    if b is not None:
                        v += b[q, r]
                    if v > best:
                        best, arg = v, q
                row_enter[r] = best + e[r]
                enter_from[t, r] = arg
        enter[t] = row_enter

        settled = [neg] * n_states
        for r in range(n_states):
            k = ks[r]
            loop = settled_prev[r] + stay[r] + e[r]
            start = t - k + 1
            if start >= 0:
                fresh = enter[start, r] + (cum[t + 1, r] - cum[start + 1, r])
            else:
                fresh = neg
            if loop >= fresh:
                settled[r] = loop
                loop_flag[t, r] = True
            else:
                settled[r] = fresh
        settled_hist[t] = settled
        settled_prev = settled

    path = np.empty(n_frames, dtype=np.int32)
    state = int(np.argmax(settled_prev))
    t = n_frames - 1
    while t >= 0:
        if loop_flag[t, state] and t > 0 and np.isfinite(settled_hist[t - 1, state]):
            path[t] = state
            t -= 1
            continue
        start = t - ks[state] + 1
        path[start:t + 1] = state
        if start == 0:
            break
        state, t = int(enter_from[start, state]), start - 1
    return path
