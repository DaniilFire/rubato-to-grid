#!/usr/bin/env python3
"""
midi_harmonize.py  --  Add chord tones and bass notes to a polished MIDI.

Input:  1_grid_polished.mid  (2-track, 8th-note grid, from midi_polish.py)
Output: 1_grid_harmonized.mid

What it adds
  Right hand: diatonic 3rd and 5th BELOW each melody note -> full triads
  Left hand:  bass root note at every beat position where left hand is empty

Detects the musical key automatically (Krumhansl-Schmuckler).
Added notes are 15 velocity units softer than the note they harmonize.

Usage:
  python midi_harmonize.py 1_grid_polished.mid
  python midi_harmonize.py 1_grid_polished.mid output.mid
  python midi_harmonize.py 1_grid_polished.mid --key C --mode minor
  python midi_harmonize.py 1_grid_polished.mid --no-chords   # bass only
  python midi_harmonize.py 1_grid_polished.mid --no-bass     # chords only
  python midi_harmonize.py 1_grid_polished.mid --max-chord 3 --bass-oct 2
"""

import argparse
import math
import os
from collections import defaultdict

import mido

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

MAJOR_STEPS = [0, 2, 4, 5, 7, 9, 11]
MINOR_STEPS = [0, 2, 3, 5, 7, 8, 10]

# Krumhansl-Schmuckler profiles
KS_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KS_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

# ---------------------------------------------------------------------------
# Key detection
# ---------------------------------------------------------------------------

def detect_key(notes):
    """Return (root 0-11, mode 'major'|'minor') using K-S profile correlation."""
    histogram = [0.0] * 12
    for n in notes:
        histogram[n["pitch"] % 12] += 1
    total = sum(histogram) or 1
    histogram = [h / total for h in histogram]

    sum_major = sum(KS_MAJOR)
    sum_minor = sum(KS_MINOR)
    norm_major = [v / sum_major for v in KS_MAJOR]
    norm_minor = [v / sum_minor for v in KS_MINOR]

    best_score = -1
    best_root = 0
    best_mode = "major"

    for root in range(12):
        for mode, profile in [("major", norm_major), ("minor", norm_minor)]:
            score = sum(histogram[(root + i) % 12] * profile[i] for i in range(12))
            if score > best_score:
                best_score = score
                best_root = root
                best_mode = mode

    return best_root, best_mode


# ---------------------------------------------------------------------------
# Scale utilities
# ---------------------------------------------------------------------------

def build_scale_pitches(key_root, mode):
    """Return sorted list of all scale pitches in MIDI range 0-127."""
    steps = MAJOR_STEPS if mode == "major" else MINOR_STEPS
    scale = []
    for octave in range(-1, 11):
        for step in steps:
            p = key_root + step + octave * 12
            if 0 <= p <= 127:
                scale.append(p)
    return sorted(scale)


def nearest_scale_index(pitch, scale):
    """Return index in scale of the note closest to pitch."""
    return min(range(len(scale)), key=lambda i: abs(scale[i] - pitch))


def diatonic_below(melody_pitch, steps_below, scale):
    """Return a pitch 'steps_below' diatonic steps below melody_pitch."""
    idx = nearest_scale_index(melody_pitch, scale)
    target = idx - steps_below
    if 0 <= target < len(scale):
        return scale[target]
    return None


# ---------------------------------------------------------------------------
# Chord root from a group of notes
# ---------------------------------------------------------------------------

CHORD_TEMPLATES = [
    ("major",  [0, 4, 7]),
    ("minor",  [0, 3, 7]),
    ("dom7",   [0, 4, 7, 10]),
    ("min7",   [0, 3, 7, 10]),
    ("dim",    [0, 3, 6]),
    ("sus4",   [0, 5, 7]),
    ("power",  [0, 7]),
]

def detect_root(pitch_list):
    """Return the best-fit chord root (0-11) for a list of pitches."""
    if not pitch_list:
        return None
    pcs = set(p % 12 for p in pitch_list)

    best_root = pitch_list[0] % 12
    best_score = -999

    for root in range(12):
        for _, intervals in CHORD_TEMPLATES:
            chord_pcs = set((root + i) % 12 for i in intervals)
            hits   = len(pcs & chord_pcs)
            extras = len(pcs - chord_pcs)
            score  = hits * 3 - extras * 2
            if score > best_score:
                best_score = score
                best_root  = root

    return best_root


# ---------------------------------------------------------------------------
# MIDI I/O (same helpers as midi_polish.py)
# ---------------------------------------------------------------------------

def tick_to_beat(tick, ppq):
    return tick / ppq

def beat_to_tick(beat, ppq):
    return int(round(beat * ppq))


def collect_notes(mid):
    ppq = mid.ticks_per_beat
    notes = []
    for track in mid.tracks:
        abs_tick = 0
        active = {}
        for msg in track:
            abs_tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                active[(msg.note, msg.channel)] = (abs_tick, msg.velocity)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                key = (msg.note, msg.channel)
                if key in active:
                    start_tick, vel = active.pop(key)
                    notes.append({
                        "pitch":    msg.note,
                        "vel":      vel,
                        "channel":  msg.channel,
                        "start":    tick_to_beat(start_tick, ppq),
                        "dur":      tick_to_beat(abs_tick - start_tick, ppq),
                    })
    notes.sort(key=lambda n: n["start"])
    return notes, ppq


def collect_tempo_events(mid):
    seen = {}
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "set_tempo":
                seen[abs_tick] = msg.tempo
    return sorted(seen.items()) if seen else [(0, 500000)]


def build_tempo_track(tempo_events):
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name="Tempo Map", time=0))
    prev = 0
    for tick, tempo in tempo_events:
        track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=tick - prev))
        prev = tick
    track.append(mido.MetaMessage("end_of_track", time=0))
    return track


def build_note_track(notes, ppq, name, channel):
    events = []
    for n in notes:
        st = beat_to_tick(n["start"], ppq)
        et = beat_to_tick(n["start"] + n["dur"], ppq)
        et = max(st + 1, et)
        vel = max(1, min(127, n["vel"]))
        events.append((st, 1, n["pitch"], vel))
        events.append((et, 0, n["pitch"], 0))
    events.sort(key=lambda x: (x[0], x[1]))

    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name=name, time=0))
    prev = 0
    for tick, kind, pitch, vel in events:
        delta = tick - prev
        prev = tick
        if kind == 1:
            track.append(mido.Message("note_on",  note=pitch, velocity=vel, channel=channel, time=delta))
        else:
            track.append(mido.Message("note_off", note=pitch, velocity=0,   channel=channel, time=delta))
    track.append(mido.MetaMessage("end_of_track", time=0))
    return track


# ---------------------------------------------------------------------------
# Harmonization
# ---------------------------------------------------------------------------

def harmonize(notes, key_root, key_mode,
              add_chords=True, add_bass=True,
              max_chord=3, bass_octave=2,
              vel_offset=-15):
    """
    Add chord tones and bass notes.

    Parameters
    ----------
    notes        : list of note dicts with 'pitch','vel','channel','start','dur'
    key_root     : detected key root (0-11)
    key_mode     : 'major' or 'minor'
    add_chords   : add diatonic 3rd+5th below right-hand melody notes
    add_bass     : add bass root note to empty left-hand beats
    max_chord    : max notes per beat position in right hand (default 3)
    bass_octave  : octave for bass root notes (2 = C2=36..B2=47, 3 = C3..B3)
    vel_offset   : velocity of added notes relative to reference note

    Returns augmented list of notes.
    """
    scale = build_scale_pitches(key_root, key_mode)
    RH, LH = 0, 1   # channel assignments from midi_polish.py

    # Separate by channel
    rh_notes = [n for n in notes if n["channel"] == RH]
    lh_notes = [n for n in notes if n["channel"] == LH]

    # Index right-hand notes by beat position
    rh_by_beat = defaultdict(list)
    for n in rh_notes:
        rh_by_beat[n["start"]].append(n)

    lh_by_beat = defaultdict(list)
    for n in lh_notes:
        lh_by_beat[n["start"]].append(n)

    added_rh = []
    added_lh = []

    # ---- Right hand: add chord tones ----
    if add_chords:
        for beat, group in rh_by_beat.items():
            slots_used = len(group)
            if slots_used >= max_chord:
                continue
            # Sort by pitch descending: top note = melody
            group_sorted = sorted(group, key=lambda n: n["pitch"], reverse=True)
            melody_note = group_sorted[0]
            ref_vel = melody_note["vel"]
            added_vel = max(1, ref_vel + vel_offset)
            existing_pcs = {n["pitch"] % 12 for n in group}

            # Add diatonic 3rd then 5th below melody
            for steps in (2, 4):
                if slots_used >= max_chord:
                    break
                harmony_pitch = diatonic_below(melody_note["pitch"], steps, scale)
                if harmony_pitch is None:
                    continue
                # Clamp to right-hand range
                harmony_pitch = max(36, min(96, harmony_pitch))
                if harmony_pitch % 12 in existing_pcs:
                    continue  # already present, skip octave duplicates
                added_rh.append({
                    "pitch":   harmony_pitch,
                    "vel":     added_vel,
                    "channel": RH,
                    "start":   beat,
                    "dur":     melody_note["dur"],
                })
                existing_pcs.add(harmony_pitch % 12)
                slots_used += 1

    # ---- Left hand: add bass notes ----
    if add_bass:
        # Collect all beat positions that have ANY note (right or left hand)
        all_beats = sorted(set(n["start"] for n in notes))

        for beat in all_beats:
            if lh_by_beat[beat]:
                continue  # left hand already has a note here

            rh_here = rh_by_beat[beat]
            if not rh_here:
                continue  # no context to derive bass from

            # Detect chord root from right-hand pitches at this beat
            root_pc = detect_root([n["pitch"] for n in rh_here])
            if root_pc is None:
                continue

            # Place root in target octave
            bass_pitch = root_pc + bass_octave * 12
            bass_pitch = max(0, min(127, bass_pitch))

            ref_vel = sum(n["vel"] for n in rh_here) // len(rh_here)
            bass_vel = max(1, ref_vel + vel_offset)

            # Duration: same as the shortest note in right hand at this beat
            dur = min(n["dur"] for n in rh_here)

            added_lh.append({
                "pitch":   bass_pitch,
                "vel":     bass_vel,
                "channel": LH,
                "start":   beat,
                "dur":     dur,
            })

    return notes + added_rh + added_lh, len(added_rh), len(added_lh)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(input_path, output_path, args):
    print(f"Reading: {input_path}")
    mid = mido.MidiFile(input_path)
    notes, ppq = collect_notes(mid)
    tempo_events = collect_tempo_events(mid)

    print(f"  {len(notes)} notes, PPQ={ppq}")

    # --- Key detection ---
    if args.key is not None:
        key_root = NOTE_NAMES.index(args.key.upper().replace("B", "A#")
                                    .replace("DB","C#").replace("EB","D#")
                                    .replace("GB","F#").replace("AB","G#")) if "#" not in args.key else \
                   NOTE_NAMES.index(args.key.upper())
        try:
            key_root = NOTE_NAMES.index(args.key)
        except ValueError:
            # handle flat/sharp aliases
            key_root = 0
        key_mode = args.mode
    else:
        key_root, key_mode = detect_key(notes)
    print(f"  Key: {NOTE_NAMES[key_root]} {key_mode}")

    # --- Harmonize ---
    augmented, n_rh, n_lh = harmonize(
        notes, key_root, key_mode,
        add_chords = not args.no_chords,
        add_bass   = not args.no_bass,
        max_chord  = args.max_chord,
        bass_octave= args.bass_oct,
        vel_offset = -15,
    )
    print(f"  Added: {n_rh} chord tones (right), {n_lh} bass notes (left)")
    print(f"  Total: {len(augmented)} notes")

    # --- Write ---
    rh_notes = [n for n in augmented if n["channel"] == 0]
    lh_notes = [n for n in augmented if n["channel"] == 1]

    out = mido.MidiFile(type=1, ticks_per_beat=ppq)
    out.tracks.append(build_tempo_track(tempo_events))
    out.tracks.append(build_note_track(rh_notes, ppq, "Right Hand", channel=0))
    out.tracks.append(build_note_track(lh_notes, ppq, "Left Hand",  channel=1))
    out.save(output_path)
    print(f"Saved: {output_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Add chord tones + bass to a polished MIDI"
    )
    ap.add_argument("input",  help="Input polished MIDI (e.g. 1_grid_polished.mid)")
    ap.add_argument("output", nargs="?", help="Output file (default: <input>_harmonized.mid)")
    ap.add_argument("--key",       default=None, help="Override key root, e.g. C, F#, Bb")
    ap.add_argument("--mode",      default="major", choices=["major", "minor"])
    ap.add_argument("--no-chords", action="store_true", help="Skip right-hand chord tones")
    ap.add_argument("--no-bass",   action="store_true", help="Skip left-hand bass notes")
    ap.add_argument("--max-chord", type=int,   default=3,
                    help="Max notes per beat in right hand (default 3)")
    ap.add_argument("--bass-oct",  type=int,   default=2,
                    help="Bass octave: 2=C2..B2, 3=C3..B3 (default 2)")
    args = ap.parse_args()

    if args.output:
        out_path = args.output
    else:
        base     = os.path.splitext(args.input)[0]
        out_path = base + "_harmonized.mid"

    run(args.input, out_path, args)


if __name__ == "__main__":
    main()
