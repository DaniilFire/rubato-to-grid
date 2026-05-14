#!/usr/bin/env python3
"""
rubato_to_grid_v3.py  -  Beat-aware rubato-to-grid MIDI converter

Improvements over v2 (auto search):
  * No --bpm needed: beat period estimated from weighted IOI histogram
  * Viterbi DP for globally consistent beat assignment
  * Events weighted by bass register, velocity, chord density
  * Gaussian-smoothed BPM curve (not raw per-anchor jumps)
  * Auto meter detection: 4/4 vs 3/4
  * Rolled chord detection and preservation (1/32 arpeggio grid)
  * Single pass, one output  (no 72-candidate search)

Algorithm:
  1. Parse MIDI -> absolute-time notes
  2. Group note-ons within 40 ms into chord events
  3. Weight each event: bass bonus + velocity + chord density
  4. Estimate tactus via weighted IOI histogram (handles half/double-time)
  5. Detect meter: 4/4 vs 3/4 via sub-beat phase analysis
  6. Filter strong events (beat candidates: high weight, spaced >= 0.3 beats)
  7. Viterbi DP: assign integer beat counts to inter-event intervals
     State = beats-in-previous-interval (k in 1..8)
     Penalises timing deviation AND large tempo jumps
  8. Build full beat_times array: linear interp between anchors
  9. Gaussian-smooth BPM curve (sigma = 2 beats default), reintegrate
 10. Quantize every note to (beat_num + subdivision) on the grid
 11. Detect rolls (ascending/descending >=3 notes within 150 ms) -> 1/32 spacing
 12. Write grid MIDI: per-beat tempo events + quantised notes
 13. Report timing deviation vs original

Usage:
    py rubato_to_grid_v3.py input.mid [output.mid] [options]

Options:
    --subdiv  4/8/12/16/32   Grid resolution (default 16 = 16th notes)
    --smooth  FLOAT          Gaussian sigma in beats for BPM smoothing (default 2.0)
    --meter   N/D            Force time signature e.g. 3/4 (default: auto)
    --min-bpm FLOAT          Minimum expected BPM (default 50)
    --max-bpm FLOAT          Maximum expected BPM (default 220)
    --verbose                Print step-by-step details

Install:
    py -m pip install mido numpy scipy
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import mido
except ImportError:
    sys.exit("Install mido:  py -m pip install mido")

try:
    import numpy as np
    from scipy.ndimage import gaussian_filter1d
    _HAS_NP = True
except ImportError:
    _HAS_NP = False
    print("[warn] numpy/scipy not found - using fallback smoothing (less accurate).")
    print("       Install:  py -m pip install numpy scipy")

# -- Constants -----------------------------------------------------------------

OUTPUT_PPQ     = 960     # ticks per quarter note (FL Studio handles up to 960 fine)
CHORD_WINDOW_S = 0.040   # note-ons within this window -> one chord event
ROLL_WINDOW_S  = 0.150   # max span of a rolled chord / arpeggio
ROLL_MIN_NOTES = 3       # minimum notes in a row to call it a roll
DEFAULT_SUBDIV = 16
DEFAULT_SMOOTH = 2.0
MIN_BPM        = 50
MAX_BPM        = 220

# Valid subdivisions as fractions of one beat (quarter note), keyed by subdiv param
_SUBDIV_FRACS: Dict[int, List[float]] = {
    4:  [k / 4  for k in range(5)],
    8:  [k / 8  for k in range(9)],
    12: [k / 12 for k in range(13)],
    16: [k / 16 for k in range(17)],
    32: [k / 32 for k in range(33)],
}
_TRIPLET_FRACS = [k / 3 for k in range(4)]  # 0, 1/3, 2/3, 1


# -- Data classes --------------------------------------------------------------

@dataclass
class Note:
    pitch:    int
    velocity: int
    start_s:  float
    end_s:    float
    channel:  int = 0


@dataclass
class Event:
    """One chord onset: all note-ons that start within CHORD_WINDOW_S of each other."""
    time_s:  float
    notes:   List[Note]
    weight:  float = 0.0
    is_roll: bool  = False   # marked by detect_arpeggios()


@dataclass
class GridNote:
    pitch:        int
    velocity:     int
    channel:      int
    tick_start:   int    # quantised start (output ticks)
    tick_len:     int    # quantised duration (output ticks)
    orig_start_s: float  # for error reporting
    orig_end_s:   float


# -- 1. MIDI parsing -----------------------------------------------------------

def _build_tempo_map(mid: "mido.MidiFile") -> List[Tuple[int, int]]:
    """Collect all set_tempo messages across all tracks, return sorted by abs_tick."""
    changes: List[Tuple[int, int]] = []
    for track in mid.tracks:
        t = 0
        for msg in track:
            t += msg.time
            if msg.type == "set_tempo":
                changes.append((t, msg.tempo))
    changes.sort(key=lambda x: x[0])
    # Ensure tick-0 entry exists
    if not changes or changes[0][0] != 0:
        changes.insert(0, (0, 500_000))  # default 120 BPM
    return changes


def _tick_to_sec(tick: int, tempo_map: List[Tuple[int, int]], ppq: int) -> float:
    """Convert absolute MIDI ticks to seconds using a tempo map."""
    t = 0.0
    prev_tick, prev_tempo = 0, tempo_map[0][1]
    for change_tick, tempo in tempo_map:
        if change_tick >= tick:
            break
        t += (change_tick - prev_tick) * prev_tempo / (1_000_000.0 * ppq)
        prev_tick, prev_tempo = change_tick, tempo
    t += (tick - prev_tick) * prev_tempo / (1_000_000.0 * ppq)
    return t


def parse_midi(path: str) -> Tuple[List[Note], int]:
    """Return (notes sorted by start_s, original PPQ)."""
    mid = mido.MidiFile(path)
    ppq = mid.ticks_per_beat
    tmap = _build_tempo_map(mid)
    active: Dict[Tuple[int, int], Tuple[int, int]] = {}  # (ch, pitch) -> (abs_tick, vel)
    notes: List[Note] = []

    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                active[(msg.channel, msg.note)] = (abs_tick, msg.velocity)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                key = (msg.channel, msg.note)
                if key in active:
                    s_tick, vel = active.pop(key)
                    notes.append(Note(
                        pitch=msg.note, velocity=vel, channel=msg.channel,
                        start_s=_tick_to_sec(s_tick, tmap, ppq),
                        end_s=  _tick_to_sec(abs_tick, tmap, ppq),
                    ))

    notes.sort(key=lambda n: n.start_s)
    return notes, ppq


# -- 2. Event grouping + weighting ---------------------------------------------

def group_events(notes: List[Note]) -> List[Event]:
    """Group note-ons within CHORD_WINDOW_S of each other into Events."""
    if not notes:
        return []
    events: List[Event] = []
    grp = [notes[0]]
    t0 = notes[0].start_s
    for n in notes[1:]:
        if n.start_s - t0 <= CHORD_WINDOW_S:
            grp.append(n)
        else:
            events.append(Event(time_s=t0, notes=list(grp)))
            grp, t0 = [n], n.start_s
    events.append(Event(time_s=t0, notes=list(grp)))
    return events


def _compute_weight(ev: Event) -> float:
    """
    Importance score for beat tracking.
    Bass notes and chord hits strongly mark beats; loud notes matter too.
    """
    max_vel = max(n.velocity for n in ev.notes)
    min_pit = min(n.pitch    for n in ev.notes)
    n_notes = len(ev.notes)

    vel_score   = max_vel / 127.0
    # pitch=30 (A1) -> 1.0, pitch=60 (C4) -> 0.0, higher -> negative (clamped to 0)
    bass_score  = max(0.0, min(1.0, (60 - min_pit) / 30.0))
    chord_score = min(n_notes / 4.0, 1.0)

    return 0.40 * vel_score + 0.35 * bass_score + 0.25 * chord_score


def weight_events(events: List[Event]) -> None:
    for ev in events:
        ev.weight = _compute_weight(ev)


# -- 3. Tactus estimation ------------------------------------------------------

def estimate_tactus(
    events: List[Event],
    min_bpm: float = MIN_BPM,
    max_bpm: float = MAX_BPM,
) -> float:
    """
    Estimate beat period in seconds from a weighted IOI histogram.

    Each inter-onset interval is treated as potentially 1, 2, or 3 beats,
    producing Gaussian bumps in a period histogram. The dominant peak is
    the tactus. Returns beat_period_s.
    """
    if len(events) < 4:
        return 60.0 / 120.0  # fallback 120 BPM

    min_p = 60.0 / max_bpm
    max_p = 60.0 / min_bpm
    N_BINS = 400
    bin_edges = [min_p + (max_p - min_p) * i / N_BINS for i in range(N_BINS + 1)]
    hist = [0.0] * N_BINS

    for i in range(len(events) - 1):
        dt = events[i + 1].time_s - events[i].time_s
        w  = events[i].weight * events[i + 1].weight
        if w < 1e-6 or dt <= 0:
            continue
        for mult in (1.0, 2.0, 3.0, 0.5):
            p = dt / mult
            if not (min_p <= p <= max_p):
                continue
            w_scaled = w / max(mult, 1.0)
            sigma = 0.02 * p  # 2% relative bandwidth
            for b in range(N_BINS):
                centre = (bin_edges[b] + bin_edges[b + 1]) * 0.5
                hist[b] += w_scaled * math.exp(-0.5 * ((centre - p) / sigma) ** 2)

    best = max(range(N_BINS), key=lambda b: hist[b])
    period = (bin_edges[best] + bin_edges[best + 1]) * 0.5
    return period


# -- 4. Meter detection --------------------------------------------------------

def detect_meter(events: List[Event], beat_period_s: float) -> Tuple[int, int]:
    """
    Return (num, denom): (4,4) or (3,4).

    Computes each event's phase within the estimated beat period and scores
    alignment with binary (0, 1/4, 1/2, 3/4) vs ternary (0, 1/3, 2/3) positions.
    """
    binary  = [0.0, 0.25, 0.5, 0.75]
    ternary = [0.0, 1/3, 2/3]
    tol = 0.10  # tolerance in beat fractions

    b_score = t_score = 0.0
    for ev in events:
        phase = (ev.time_s / beat_period_s) % 1.0
        b_score += ev.weight * max(0.0, 1.0 - min(abs(phase - p) for p in binary)  / tol)
        t_score += ev.weight * max(0.0, 1.0 - min(abs(phase - p) for p in ternary) / tol)

    return (3, 4) if t_score > 1.5 * b_score else (4, 4)


# -- 5. Viterbi beat assignment ------------------------------------------------

def _select_strong(events: List[Event], beat_period_s: float) -> List[Event]:
    """
    Keep only events likely to mark beats.

    The gap threshold of 0.75 * beat_period ensures consecutive selected
    events are at least ~3/4 of a beat apart, so the Viterbi can assign
    integer k values with valid BPM (8th-note spacing would give a BPM
    too far from the reference to be accepted by the DP).

    Threshold: top 35% by weight keeps bass chord hits and loud accents
    while discarding passing tones and ornaments.
    """
    if not events:
        return []
    sorted_w = sorted(ev.weight for ev in events)
    thresh   = sorted_w[int(0.65 * len(sorted_w))]   # keep top 35%
    min_gap  = 0.75 * beat_period_s                   # ~75% of a beat

    selected: List[Event] = []
    for ev in events:
        if ev.weight < thresh:
            continue
        if selected and ev.time_s - selected[-1].time_s < min_gap:
            if ev.weight > selected[-1].weight:
                selected[-1] = ev
        else:
            selected.append(ev)
    return selected


def assign_beats_dp(
    strong_events: List[Event],
    beat_period_s: float,
    max_k: int = 8,
    lambda_smooth: float = 3.0,
) -> List[int]:
    """
    Viterbi DP: find the integer beat count K[i] for each interval between
    consecutive strong events.

    State  = K[i] (beats in the current interval, 1..max_k)
    Cost   = (timing_deviation)^2  +  lambda_smooth * (BPM_change / BPM_ref)^2
    Global optimum via forward DP + traceback.

    Returns beat_numbers[0..M-1] starting from 0, monotonically increasing.
    """
    M = len(strong_events)
    if M == 0:
        return []
    if M == 1:
        return [0]

    times   = [ev.time_s for ev in strong_events]
    dts     = [times[i + 1] - times[i] for i in range(M - 1)]
    bpm_ref = 60.0 / beat_period_s
    BPM_LO  = bpm_ref * 0.40
    BPM_HI  = bpm_ref * 2.50
    INF     = float("inf")

    def t_cost(k: int, ii: int) -> float:
        """Cost of assigning k beats to interval ii."""
        dt = dts[ii]
        if dt <= 1e-6:
            return INF
        bpm = 60.0 * k / dt
        if not (BPM_LO <= bpm <= BPM_HI):
            return INF
        dev = (dt - k * beat_period_s) / beat_period_s
        return dev * dev

    # dp[ii][ki]   = min cost when interval ii has (ki+1) beats
    # prev[ii][ki] = ki index of previous interval at optimum
    dp   = [[INF] * max_k for _ in range(M - 1)]
    prev = [[-1]  * max_k for _ in range(M - 1)]

    # Initialise first interval
    for ki in range(max_k):
        dp[0][ki] = t_cost(ki + 1, 0)

    # Fill forward
    for ii in range(1, M - 1):
        dt_c = dts[ii]
        dt_p = dts[ii - 1]
        for ki in range(max_k):
            k  = ki + 1
            tc = t_cost(k, ii)
            if tc == INF:
                continue
            bpm_c     = 60.0 * k / max(dt_c, 1e-6)
            best_cost = INF
            best_pki  = -1
            for pki in range(max_k):
                if dp[ii - 1][pki] == INF:
                    continue
                pk    = pki + 1
                bpm_p = 60.0 * pk / max(dt_p, 1e-6)
                sc    = lambda_smooth * ((bpm_c - bpm_p) / bpm_ref) ** 2
                total = dp[ii - 1][pki] + sc + tc
                if total < best_cost:
                    best_cost, best_pki = total, pki
            dp[ii][ki]   = best_cost
            prev[ii][ki] = best_pki

    # Guard: if the entire DP is unreachable, fall back to uniform 1-beat steps
    if all(dp[M - 2][ki] == INF for ki in range(max_k)):
        print("[warn] Viterbi DP produced no valid solution — falling back to k=1 everywhere.")
        print("       Try --min-bpm / --max-bpm to widen the tempo search range.")
        return list(range(M))

    # Traceback
    last_ki = min(range(max_k), key=lambda ki: dp[M - 2][ki])
    k_seq        = [1] * (M - 1)   # default k=1 (safest fallback)
    k_seq[M - 2] = last_ki + 1
    ki = last_ki
    for ii in range(M - 2, 0, -1):
        pk = prev[ii][ki]
        if pk < 0:                  # -1 means this state was never reached
            pk = 0                  # fall back to ki=0 (k=1)
        ki             = pk
        k_seq[ii - 1]  = ki + 1

    beat_nums = [0] * M
    for i in range(M - 1):
        beat_nums[i + 1] = beat_nums[i] + k_seq[i]
    return beat_nums


# -- 6. Beat time array --------------------------------------------------------

def build_beat_times(
    strong_times: List[float],
    strong_beat_nums: List[int],
    beat_period_s: float,
    extra_beats: int = 8,
) -> List[float]:
    """
    Build beat_times[b] = real seconds at which beat b starts, for b = 0..B+extra.
    Between anchor beats: linear interpolation.
    Before/after anchors: linear extrapolation.
    """
    if not strong_beat_nums:
        return []

    B = strong_beat_nums[-1] + extra_beats
    beat_times = [0.0] * (B + 1)

    anchors = sorted(zip(strong_beat_nums, strong_times))

    # Set anchor points
    for bn, t in anchors:
        if 0 <= bn <= B:
            beat_times[bn] = t

    # Interpolate between consecutive anchors
    for seg in range(len(anchors) - 1):
        b0, t0 = anchors[seg]
        b1, t1 = anchors[seg + 1]
        n = b1 - b0
        if n <= 0:
            continue
        dt_per_beat = (t1 - t0) / n
        for b in range(b0, b1):
            if 0 <= b <= B:
                beat_times[b] = t0 + (b - b0) * dt_per_beat

    # Extrapolate before first anchor
    first_b, first_t = anchors[0]
    if first_b > 0:
        dt_back = (anchors[1][1] - first_t) / max(1, anchors[1][0] - first_b) if len(anchors) > 1 else beat_period_s
        for b in range(0, first_b):
            beat_times[b] = first_t + (b - first_b) * dt_back

    # Extrapolate after last anchor
    last_b, last_t = anchors[-1]
    dt_fwd = (last_t - anchors[-2][1]) / max(1, last_b - anchors[-2][0]) if len(anchors) > 1 else beat_period_s
    for b in range(last_b + 1, B + 1):
        beat_times[b] = last_t + (b - last_b) * dt_fwd

    return beat_times


# -- 7. BPM smoothing ----------------------------------------------------------

def _gaussian_smooth_py(values: List[float], sigma: float) -> List[float]:
    """Pure-Python fallback Gaussian smoothing (no numpy)."""
    if sigma <= 0 or not values:
        return list(values)
    radius = max(1, int(3.0 * sigma))
    out = []
    n = len(values)
    for i in range(n):
        tw = tv = 0.0
        for j in range(max(0, i - radius), min(n, i + radius + 1)):
            w  = math.exp(-0.5 * ((j - i) / sigma) ** 2)
            tw += w
            tv += w * values[j]
        out.append(tv / tw if tw > 0 else values[i])
    return out


def smooth_beat_times(
    beat_times: List[float],
    sigma_beats: float,
) -> Tuple[List[float], List[float]]:
    """
    Smooth the raw per-beat BPM sequence with a Gaussian, then reintegrate
    to produce smoothed beat timestamps.

    Returns (smoothed_beat_times, smoothed_bpm_per_beat).
    """
    B = len(beat_times)
    if B < 2:
        return list(beat_times), [120.0] * B

    # Raw BPM
    raw_bpm: List[float] = []
    for i in range(B - 1):
        dt = beat_times[i + 1] - beat_times[i]
        raw_bpm.append(60.0 / max(dt, 0.005))
    raw_bpm.append(raw_bpm[-1])  # pad last

    # Smooth
    if _HAS_NP:
        smoothed = list(gaussian_filter1d(np.array(raw_bpm, dtype=float), sigma=sigma_beats))
    else:
        smoothed = _gaussian_smooth_py(raw_bpm, sigma_beats)

    # Clamp
    smoothed = [max(MIN_BPM, min(MAX_BPM, b)) for b in smoothed]

    # Reintegrate: fix first beat, derive rest from smoothed BPM
    new_times = [beat_times[0]] * B
    for i in range(B - 1):
        new_times[i + 1] = new_times[i] + 60.0 / smoothed[i]

    return new_times, smoothed


# -- 8. Arpeggio detection -----------------------------------------------------

def detect_arpeggios(events: List[Event]) -> None:
    """
    Mark events as is_roll=True when they form a 3+-note ascending or descending
    run within ROLL_WINDOW_S. Modifies events in-place.
    """
    n = len(events)
    for i in range(n):
        if events[i].is_roll:
            continue
        span = [events[i]]
        for j in range(i + 1, n):
            if events[j].time_s - events[i].time_s > ROLL_WINDOW_S:
                break
            span.append(events[j])

        if len(span) < ROLL_MIN_NOTES:
            continue

        pitches = [min(ev.notes, key=lambda note: note.pitch).pitch for ev in span]
        asc  = all(pitches[k] <= pitches[k + 1] for k in range(len(pitches) - 1))
        desc = all(pitches[k] >= pitches[k + 1] for k in range(len(pitches) - 1))
        if asc or desc:
            for ev in span:
                ev.is_roll = True


# -- 9. Note quantization ------------------------------------------------------

def _nearest_subdiv(frac: float, subdiv: int, allow_triplets: bool = True) -> float:
    """Snap a beat fraction to the nearest valid subdivision."""
    candidates = list(_SUBDIV_FRACS.get(subdiv, _SUBDIV_FRACS[16]))
    if allow_triplets:
        candidates = candidates + _TRIPLET_FRACS
    return min(candidates, key=lambda s: abs(frac - s))


def _find_beat_frac(t: float, beat_times: List[float]) -> Tuple[int, float]:
    """
    Binary-search beat_times for t, return (beat_num, frac_within_beat).
    frac is 0.0..1.0; handles times before the first beat or after the last.
    """
    B = len(beat_times)
    if B < 2:
        return 0, 0.0

    # Before first beat
    if t <= beat_times[0]:
        dt = beat_times[1] - beat_times[0]
        frac = (t - beat_times[0]) / max(dt, 1e-6)
        return 0, max(0.0, frac)

    # After last beat
    if t >= beat_times[-1]:
        dt = beat_times[-1] - beat_times[-2]
        frac = (t - beat_times[-1]) / max(dt, 1e-6)
        return B - 1, min(0.9999, frac)

    # Binary search
    lo, hi = 0, B - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if beat_times[mid] <= t:
            lo = mid
        else:
            hi = mid

    b  = lo
    t0 = beat_times[b]
    t1 = beat_times[b + 1]
    frac = (t - t0) / max(t1 - t0, 1e-6)
    return b, max(0.0, min(0.9999, frac))


def quantize_notes(
    notes: List[Note],
    events: List[Event],
    beat_times: List[float],
    subdiv: int,
    ppq: int,
) -> List[GridNote]:
    """
    Snap all note starts to the grid.

    Roll notes: first note of the run snaps to nearest subdivision,
    subsequent notes are spaced at 1/32 intervals.

    Note durations are preserved proportionally (not snapped), since
    piano sustain pedal causes durations to overlap naturally.
    """
    # Build a mapping: note -> event
    note_to_event: Dict[int, Event] = {}
    for ev in events:
        for n in ev.notes:
            note_to_event[id(n)] = ev

    # Identify roll groups: consecutive events that are rolls
    roll_groups: List[List[Event]] = []
    i = 0
    while i < len(events):
        if events[i].is_roll:
            grp = [events[i]]
            j = i + 1
            while j < len(events) and events[j].is_roll and \
                  events[j].time_s - events[i].time_s <= ROLL_WINDOW_S:
                grp.append(events[j])
                j += 1
            roll_groups.append(grp)
            i = j
        else:
            i += 1

    # Map event -> roll_offset_ticks
    roll_tick_offset: Dict[int, int] = {}  # id(event) -> extra ticks
    roll_spacing = ppq // 32  # 1/32 note spacing
    for grp in roll_groups:
        t_base = grp[0].time_s
        for k, ev in enumerate(grp):
            roll_tick_offset[id(ev)] = k * roll_spacing

    grid_notes: List[GridNote] = []
    for n in notes:
        ev = note_to_event.get(id(n))

        # Quantise start
        b, frac = _find_beat_frac(n.start_s, beat_times)
        snapped_frac = _nearest_subdiv(frac, subdiv)
        tick_start = int((b + snapped_frac) * ppq)

        # Apply roll offset if applicable
        if ev and id(ev) in roll_tick_offset:
            tick_start += roll_tick_offset[id(ev)]

        # Duration: preserve proportionally from tempo map
        b_end, frac_end = _find_beat_frac(n.end_s, beat_times)
        raw_tick_end = int((b_end + frac_end) * ppq)
        tick_len = max(ppq // subdiv, raw_tick_end - tick_start)

        grid_notes.append(GridNote(
            pitch=n.pitch, velocity=max(1, min(127, n.velocity)), channel=n.channel,
            tick_start=max(0, tick_start), tick_len=max(1, tick_len),
            orig_start_s=n.start_s, orig_end_s=n.end_s,
        ))

    return grid_notes


# -- 10. Tempo events ----------------------------------------------------------

def build_tempo_events(
    bpm_per_beat: List[float],
    ppq: int,
) -> List[Tuple[int, int]]:
    """One MIDI set_tempo event per beat. FL Studio reads these as a tempo map."""
    events: List[Tuple[int, int]] = []
    seen: set = set()
    for i, bpm in enumerate(bpm_per_beat):
        tick = i * ppq
        if tick in seen:
            continue
        seen.add(tick)
        tempo_us = int(round(60_000_000.0 / max(bpm, 1.0)))
        events.append((tick, tempo_us))
    return sorted(events, key=lambda x: x[0])


# -- 11. MIDI output -----------------------------------------------------------

def write_midi(
    path: str,
    grid_notes: List[GridNote],
    tempo_events: List[Tuple[int, int]],
    ppq: int,
) -> None:
    mid   = mido.MidiFile(ticks_per_beat=ppq)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # Collect all messages with absolute ticks + sort key
    msgs: List[Tuple[int, int, object]] = []
    for tick, tempo_us in tempo_events:
        msgs.append((max(0, tick), 0,
                     mido.MetaMessage("set_tempo", tempo=tempo_us, time=0)))
    for gn in grid_notes:
        s = max(0, gn.tick_start)
        e = max(s + 1, s + gn.tick_len)
        msgs.append((s, 2, mido.Message("note_on",  note=gn.pitch, velocity=gn.velocity,
                                        channel=gn.channel, time=0)))
        msgs.append((e, 1, mido.Message("note_off", note=gn.pitch, velocity=0,
                                        channel=gn.channel, time=0)))

    msgs.sort(key=lambda x: (x[0], x[1]))

    last_tick = 0
    for abs_tick, _, msg in msgs:
        msg.time = max(0, abs_tick - last_tick)
        track.append(msg)
        last_tick = abs_tick

    track.append(mido.MetaMessage("end_of_track", time=0))
    mid.save(path)


# -- 12. Timing report ---------------------------------------------------------

def _sec_at_tick(tick: int, tempo_events: List[Tuple[int, int]], ppq: int) -> float:
    """Integrate tempo events to convert output ticks back to seconds."""
    t = 0.0
    prev_tick, prev_tempo = 0, 500_000
    for ev_tick, tempo in sorted(tempo_events, key=lambda x: x[0]):
        if ev_tick >= tick:
            break
        t += (ev_tick - prev_tick) * prev_tempo / (1_000_000.0 * ppq)
        prev_tick, prev_tempo = ev_tick, tempo
    t += (tick - prev_tick) * prev_tempo / (1_000_000.0 * ppq)
    return t


def report_timing(
    grid_notes: List[GridNote],
    tempo_events: List[Tuple[int, int]],
    ppq: int,
) -> None:
    """
    Report timing accuracy of the output MIDI vs original.

    Two numbers are shown:
    - Constant offset: how much the entire output is shifted relative to the
      original (usually the lead-in silence; irrelevant for FL Studio usage).
    - Residual error: per-note timing deviation AFTER removing the constant
      offset.  This is the true measure of rubato fidelity.
    """
    raw_errors = []
    for gn in grid_notes:
        out_s = _sec_at_tick(gn.tick_start, tempo_events, ppq)
        raw_errors.append(out_s - gn.orig_start_s)

    if not raw_errors:
        print("  No notes to compare.")
        return

    raw_sorted = sorted(raw_errors)
    offset_s = raw_sorted[len(raw_sorted) // 2]   # median = constant shift

    residuals = [abs(e - offset_s) * 1000.0 for e in raw_errors]
    res_sorted = sorted(residuals)
    n = len(residuals)
    rms_e = math.sqrt(sum(r * r for r in residuals) / n)
    p95   = res_sorted[int(0.95 * n)]
    p99   = res_sorted[int(0.99 * n)]
    max_e = max(residuals)
    g20   = sum(1 for r in residuals if r < 20.0)
    g50   = sum(1 for r in residuals if r < 50.0)

    print(f"  Constant offset vs original: {offset_s * 1000:.0f} ms")
    print(f"  (This is the lead-in silence; FL Studio ignores clip start time.)")
    print(f"  Residual timing error (rubato fidelity):")
    print(f"    RMS  = {rms_e:.1f} ms")
    print(f"    P95  = {p95:.1f} ms")
    print(f"    P99  = {p99:.1f} ms")
    print(f"    Max  = {max_e:.1f} ms")
    print(f"    Within 20 ms: {g20}/{n}  ({100 * g20 // n}%)")
    print(f"    Within 50 ms: {g50}/{n}  ({100 * g50 // n}%)")


# -- 13. Pipeline --------------------------------------------------------------

def pipeline(
    in_path: str,
    out_path: str,
    subdiv: int           = DEFAULT_SUBDIV,
    smooth: float         = DEFAULT_SMOOTH,
    force_meter: Optional[Tuple[int, int]] = None,
    min_bpm: float        = MIN_BPM,
    max_bpm: float        = MAX_BPM,
    verbose: bool         = False,
) -> None:
    def log(msg: str) -> None:
        if verbose:
            print(f"      {msg}")

    print(f"[1] Parsing  {in_path}")
    notes, orig_ppq = parse_midi(in_path)
    print(f"    {len(notes)} notes  (original PPQ={orig_ppq})")

    print("[2] Grouping note-ons into chord events (window=40 ms)")
    events = group_events(notes)
    weight_events(events)
    log(f"{len(events)} events  weight range: "
        f"{min(e.weight for e in events):.2f}-{max(e.weight for e in events):.2f}")

    print("[3] Estimating tactus from IOI histogram")
    beat_period = estimate_tactus(events, min_bpm=min_bpm, max_bpm=max_bpm)
    est_bpm = 60.0 / beat_period
    print(f"    Estimated BPM ~{est_bpm:.1f}  (beat period = {beat_period*1000:.0f} ms)")

    print("[4] Detecting meter")
    meter = force_meter if force_meter else detect_meter(events, beat_period)
    print(f"    Meter: {meter[0]}/{meter[1]}")

    print("[5] Selecting beat candidates (high weight + spaced >= 0.3 beats)")
    strong = _select_strong(events, beat_period)
    log(f"Weights of first 10: {[f'{e.weight:.2f}' for e in strong[:10]]}")
    print(f"    {len(strong)} strong events selected")

    if len(strong) < 2:
        print("[ERR] Too few strong events for beat tracking.")
        print("      Try --min-bpm / --max-bpm or check the MIDI file.")
        sys.exit(1)

    print("[6] Viterbi DP: assigning beat numbers")
    beat_nums = assign_beats_dp(strong, beat_period)

    # Add lead-in so notes before the first strong event have valid beat positions.
    # Without this, pre-beat-0 notes collapse to tick 0.
    t_first_note   = notes[0].start_s if notes else 0.0
    t_first_strong = strong[0].time_s
    lead_s         = max(0.0, t_first_strong - t_first_note)
    lead_beats     = max(0, math.ceil(lead_s / beat_period) + 1)
    beat_nums      = [b + lead_beats for b in beat_nums]

    total_beats = beat_nums[-1]
    log(f"Beat sequence: {beat_nums[:20]}{'...' if len(beat_nums)>20 else ''}")
    log(f"Lead-in: {lead_s*1000:.0f} ms = {lead_beats} beats")
    print(f"    {total_beats} beats found  "
          f"({total_beats / meter[0]:.1f} bars of {meter[0]}/{meter[1]})")

    print("[7] Building full beat time array")
    strong_times = [ev.time_s for ev in strong]
    raw_beat_times = build_beat_times(strong_times, beat_nums, beat_period)
    print(f"    {len(raw_beat_times)} beat slots")

    print(f"[8] Smoothing BPM curve  (sigma = {smooth} beats)")
    beat_times, bpm_per_beat = smooth_beat_times(raw_beat_times, sigma_beats=smooth)
    print(f"    BPM range: {min(bpm_per_beat):.1f} - {max(bpm_per_beat):.1f}")

    print("[9] Detecting rolled chords")
    detect_arpeggios(events)
    n_rolls = sum(1 for ev in events if ev.is_roll)
    print(f"    {n_rolls} roll events marked")

    print(f"[10] Quantizing notes to 1/{subdiv} grid")
    grid_notes = quantize_notes(notes, events, beat_times, subdiv, OUTPUT_PPQ)

    print("[11] Building tempo map")
    tempo_events = build_tempo_events(bpm_per_beat, OUTPUT_PPQ)
    print(f"     {len(tempo_events)} tempo change events")

    print(f"[12] Writing  {out_path}")
    write_midi(out_path, grid_notes, tempo_events, OUTPUT_PPQ)
    print(f"     Done. Output PPQ={OUTPUT_PPQ}, grid=1/{subdiv}")

    print("[13] Timing accuracy report:")
    report_timing(grid_notes, tempo_events, OUTPUT_PPQ)


# -- CLI -----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Beat-aware rubato -> grid MIDI converter (v3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  py rubato_to_grid_v3.py piece.mid
  py rubato_to_grid_v3.py piece.mid out.mid --subdiv 8
  py rubato_to_grid_v3.py piece.mid out.mid --meter 3/4 --smooth 1.5
  py rubato_to_grid_v3.py piece.mid out.mid --min-bpm 80 --max-bpm 160 --verbose

FL Studio:
  Drag the output .mid into FL Studio.
  When prompted, enable "Import tempo map / tempo changes".
""",
    )
    parser.add_argument("input",           help="Rubato MIDI input file")
    parser.add_argument("output", nargs="?",
                        help="Grid MIDI output (default: <input>_grid.mid)")
    parser.add_argument("--subdiv",  type=int,   default=DEFAULT_SUBDIV,
                        metavar="N",
                        help="Grid subdivision: 4 8 12 16 32  (default %(default)s)")
    parser.add_argument("--smooth",  type=float, default=DEFAULT_SMOOTH,
                        metavar="SIGMA",
                        help="Gaussian BPM smoothing in beats (default %(default)s)")
    parser.add_argument("--meter",   default=None,
                        metavar="N/D",
                        help="Force time signature e.g. 3/4  (default: auto)")
    parser.add_argument("--min-bpm", type=float, default=MIN_BPM,
                        metavar="BPM",
                        help="Min BPM for beat detection (default %(default)s)")
    parser.add_argument("--max-bpm", type=float, default=MAX_BPM,
                        metavar="BPM",
                        help="Max BPM for beat detection (default %(default)s)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed debug info")
    args = parser.parse_args()

    if args.subdiv not in _SUBDIV_FRACS:
        sys.exit(f"--subdiv must be one of {sorted(_SUBDIV_FRACS.keys())}")

    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"File not found: {in_path}")

    out_path = str(args.output or in_path.with_stem(in_path.stem + "_grid"))

    meter = None
    if args.meter:
        parts = args.meter.split("/")
        if len(parts) != 2 or not all(p.isdigit() for p in parts):
            sys.exit("--meter must be N/D, e.g. 3/4 or 4/4")
        meter = (int(parts[0]), int(parts[1]))

    pipeline(
        in_path=str(in_path),
        out_path=out_path,
        subdiv=args.subdiv,
        smooth=args.smooth,
        force_meter=meter,
        min_bpm=args.min_bpm,
        max_bpm=args.max_bpm,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
