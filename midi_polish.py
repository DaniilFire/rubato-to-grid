#!/usr/bin/env python3
"""
midi_polish.py  --  Post-process a grid-aligned MIDI for professional piano roll aesthetics.

Input:  1_grid.mid  (output of rubato_to_grid_v3.py)
Output: 1_grid_polished.mid  (or specify with second argument)

Transforms applied
  1. Snap note starts to 8th-note grid using FLOOR snap (never jumps a note forward to
     the next beat -- always snaps back, preserving the note's musical beat position)
  2. Trim note durations to max 0.5 beats, capped at gap to next note (8th-note look)
  3. Split notes into Right Hand (pitch >= split) and Left Hand (pitch < split) tracks
  4. Normalize velocity to [VEL_MIN, VEL_MAX] range
  5. Preserve the tempo map from the input unchanged

Usage:
  python midi_polish.py 1_grid.mid
  python midi_polish.py 1_grid.mid output.mid
  python midi_polish.py 1_grid.mid --split 55 --max-dur 0.5 --vel-min 10 --vel-max 90
"""

import argparse
import math
import os

import mido

# ---- defaults (all overridable via CLI) ----
PITCH_SPLIT = 60    # pitch >= this -> Right Hand, else Left Hand
SNAP_DIV    = 0.5   # 8th-note grid (never change below 0.0625)
MAX_DUR     = 0.5   # maximum note duration in beats (8th note = like example.mid)
MIN_DUR     = 0.0625  # minimum note duration (1/16 beat floor)
VEL_MIN     = 15
VEL_MAX     = 90


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def snap_floor(beat: float) -> float:
    """Floor-snap to SNAP_DIV grid: always moves notes backward (never forward).
    Prevents notes from jumping to the NEXT beat when they are in the late portion
    of the current beat -- which would break the melody."""
    return math.floor(beat / SNAP_DIV) * SNAP_DIV


def beat_to_tick(beat: float, ppq: int) -> int:
    return int(round(beat * ppq))


def tick_to_beat(tick: int, ppq: int) -> float:
    return tick / ppq


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def collect_notes(mid: mido.MidiFile):
    """Return list of note dicts and the PPQ of the file."""
    ppq = mid.ticks_per_beat
    notes = []
    for track in mid.tracks:
        active = {}   # (pitch, channel) -> (abs_tick, velocity)
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                active[(msg.note, msg.channel)] = (abs_tick, msg.velocity)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                key = (msg.note, msg.channel)
                if key in active:
                    start_tick, vel = active.pop(key)
                    notes.append({
                        "pitch":       msg.note,
                        "vel":         vel,
                        "start_beat":  tick_to_beat(start_tick, ppq),
                        "end_beat":    tick_to_beat(abs_tick,   ppq),
                    })
    notes.sort(key=lambda n: n["start_beat"])
    return notes, ppq


def collect_tempo_events(mid: mido.MidiFile):
    """Return sorted list of (abs_tick, tempo_us) covering all tracks."""
    seen = {}
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "set_tempo":
                seen[abs_tick] = msg.tempo
    if not seen:
        return [(0, 500000)]
    return sorted(seen.items())


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def normalize_velocity(notes):
    if not notes:
        return
    vmin = min(n["vel"] for n in notes)
    vmax = max(n["vel"] for n in notes)
    if vmax == vmin:
        mid_vel = (VEL_MIN + VEL_MAX) // 2
        for n in notes:
            n["vel"] = mid_vel
        return
    for n in notes:
        t = (n["vel"] - vmin) / (vmax - vmin)
        n["vel"] = int(round(VEL_MIN + t * (VEL_MAX - VEL_MIN)))


def process_notes(notes):
    """
    Duration trim + velocity normalisation. Note starts are kept as-is.
    Returns (right_hand_notes, left_hand_notes).
    """
    # 1. Floor-snap to 8th-note grid (never moves notes forward to the next beat)
    for n in notes:
        n["snapped_start"] = max(0.0, snap_floor(n["start_beat"]))

    # 2. Split into hands
    right = [n for n in notes if n["pitch"] >= PITCH_SPLIT]
    left  = [n for n in notes if n["pitch"] <  PITCH_SPLIT]

    # 3. Trim durations per hand: cap at MAX_DUR and at gap to the next note
    for hand in (right, left):
        hand.sort(key=lambda n: n["snapped_start"])
        for i, n in enumerate(hand):
            cur = n["snapped_start"]
            # Find next note with a strictly later start (skip chord-mates at same tick)
            next_start = None
            for j in range(i + 1, len(hand)):
                if hand[j]["snapped_start"] > cur + 1e-9:
                    next_start = hand[j]["snapped_start"]
                    break
            if next_start is not None:
                dur = min(MAX_DUR, max(MIN_DUR, next_start - cur))
            else:
                dur = MAX_DUR
            n["polished_dur"] = dur

    # 4. Normalize velocity across all notes
    normalize_velocity(notes)

    return right, left


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def build_tempo_track(tempo_events):
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name="Tempo Map", time=0))
    prev_tick = 0
    for abs_tick, tempo in tempo_events:
        delta = abs_tick - prev_tick
        prev_tick = abs_tick
        track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=delta))
    track.append(mido.MetaMessage("end_of_track", time=0))
    return track


def build_note_track(notes, ppq, track_name, channel):
    # Build flat event list: (abs_tick, priority, pitch, velocity)
    # priority 0 = note_off, 1 = note_on  (offs fire before ons at same tick)
    raw = []
    for n in notes:
        start_tick = beat_to_tick(n["snapped_start"], ppq)
        end_tick   = beat_to_tick(n["snapped_start"] + n["polished_dur"], ppq)
        end_tick   = max(start_tick + 1, end_tick)
        vel        = max(1, min(127, n["vel"]))
        raw.append((start_tick, 1, n["pitch"], vel))
        raw.append((end_tick,   0, n["pitch"], 0))

    raw.sort(key=lambda x: (x[0], x[1]))  # offs before ons

    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name=track_name, time=0))
    prev_tick = 0
    for abs_tick, priority, pitch, vel in raw:
        delta = abs_tick - prev_tick
        prev_tick = abs_tick
        if priority == 1:
            track.append(mido.Message("note_on",  note=pitch, velocity=vel, channel=channel, time=delta))
        else:
            track.append(mido.Message("note_off", note=pitch, velocity=0,   channel=channel, time=delta))
    track.append(mido.MetaMessage("end_of_track", time=0))
    return track


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def count_overlaps(hand):
    """Count beat-position groups that overlap the next group."""
    if len(hand) < 2:
        return 0
    hand = sorted(hand, key=lambda n: n["snapped_start"])
    overlaps = 0
    i = 0
    while i < len(hand):
        cur_start = hand[i]["snapped_start"]
        # find end of current beat-position group
        j = i
        while j < len(hand) and hand[j]["snapped_start"] == cur_start:
            j += 1
        cur_end = cur_start + hand[i]["polished_dur"]
        if j < len(hand):
            next_start = hand[j]["snapped_start"]
            if cur_end > next_start + 1e-9:
                overlaps += 1
        i = j
    return overlaps


def slot_stats(hand, snap_div):
    """Count notes at each fractional beat slot within a measure."""
    from collections import Counter
    slots = Counter()
    for n in hand:
        slot = round((n["snapped_start"] % 1.0) / snap_div)
        slots[slot] += 1
    return slots


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def polish(input_path, output_path):
    print(f"Reading: {input_path}")
    mid = mido.MidiFile(input_path)

    notes, ppq       = collect_notes(mid)
    tempo_events     = collect_tempo_events(mid)

    print(f"  {len(notes)} notes, PPQ={ppq}, {len(tempo_events)} tempo events")

    right, left = process_notes(notes)

    print(f"  Right hand (pitch>={PITCH_SPLIT}): {len(right)} notes")
    print(f"  Left  hand (pitch< {PITCH_SPLIT}): {len(left)} notes")
    print(f"  Overlaps after polish -- Right: {count_overlaps(right)}, Left: {count_overlaps(left)}")

    # Duration distribution summary
    all_durs = [n["polished_dur"] for n in notes]
    if all_durs:
        trimmed = sum(1 for n in notes if n["polished_dur"] < n["end_beat"] - n["start_beat"] - 1e-9)
        print(f"  Trimmed {trimmed}/{len(all_durs)} notes  (max_dur={MAX_DUR} beats)")

    # Velocity summary
    vels = [n["vel"] for n in notes]
    if vels:
        print(f"  Velocity: min={min(vels)}, max={max(vels)}, mean={sum(vels)//len(vels)}")

    out = mido.MidiFile(type=1, ticks_per_beat=ppq)
    out.tracks.append(build_tempo_track(tempo_events))
    out.tracks.append(build_note_track(right, ppq, "Right Hand", channel=0))
    out.tracks.append(build_note_track(left,  ppq, "Left Hand",  channel=1))
    out.save(output_path)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Post-process grid MIDI for professional piano roll appearance"
    )
    ap.add_argument("input",  help="Input grid MIDI (e.g. 1_grid.mid)")
    ap.add_argument("output", nargs="?", help="Output file (default: <input>_polished.mid)")
    ap.add_argument("--split",   type=int,   default=60,  help="Pitch split point (default 60 = C4)")
    ap.add_argument("--max-dur", type=float, default=0.5, help="Max note duration in beats (default 0.5 = 8th note, like example.mid)")
    ap.add_argument("--vel-min", type=int,   default=15,  help="Min output velocity (default 15)")
    ap.add_argument("--vel-max", type=int,   default=90,  help="Max output velocity (default 90)")
    args = ap.parse_args()

    global PITCH_SPLIT, MAX_DUR, VEL_MIN, VEL_MAX
    PITCH_SPLIT = args.split
    MAX_DUR     = args.max_dur
    VEL_MIN     = args.vel_min
    VEL_MAX     = args.vel_max

    if args.output:
        out_path = args.output
    else:
        base     = os.path.splitext(args.input)[0]
        out_path = base + "_polished.mid"

    polish(args.input, out_path)


if __name__ == "__main__":
    main()
