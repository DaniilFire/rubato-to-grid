#!/usr/bin/env python3
"""
midi_split_voices.py — Split a MIDI file into 3 voice layers.

Input:  any.mid
Output: any_melody.mid  — top voice  (highest notes at each moment)
        any_middle.mid  — inner voices / chords
        any_bass.mid    — bass voice  (lowest notes at each moment)

Usage:
  python midi_split_voices.py input.mid
  python midi_split_voices.py input.mid --method voice       # time-based (default, most accurate)
  python midi_split_voices.py input.mid --method range       # fixed pitch ranges
  python midi_split_voices.py input.mid --method percentile  # auto percentile split
  python midi_split_voices.py input.mid --bass-max 47 --melody-min 60  # custom range thresholds
  python midi_split_voices.py input.mid --out-dir ./split    # save to a specific folder

Requires:
  pip install mido
"""

import argparse
import os
import sys
from collections import defaultdict

import mido


# ---------------------------------------------------------------------------
# MIDI I/O
# ---------------------------------------------------------------------------

def collect_notes(mid):
    ppq = mid.ticks_per_beat
    notes = []
    for track in mid.tracks:
        abs_tick = 0
        active = {}
        for msg in track:
            abs_tick += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                active[(msg.note, msg.channel)] = (abs_tick, msg.velocity)
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                key = (msg.note, msg.channel)
                if key in active:
                    start_tick, vel = active.pop(key)
                    notes.append({
                        'pitch':   msg.note,
                        'vel':     vel,
                        'channel': msg.channel,
                        'start':   start_tick / ppq,
                        'dur':     (abs_tick - start_tick) / ppq,
                    })
    notes.sort(key=lambda n: n['start'])
    return notes, ppq


def collect_tempo_events(mid):
    seen = {}
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == 'set_tempo':
                seen[abs_tick] = msg.tempo
    return sorted(seen.items()) if seen else [(0, 500_000)]


def build_tempo_track(tempo_events):
    track = mido.MidiTrack()
    track.append(mido.MetaMessage('track_name', name='Tempo Map', time=0))
    prev = 0
    for tick, tempo in tempo_events:
        track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=tick - prev))
        prev = tick
    track.append(mido.MetaMessage('end_of_track', time=0))
    return track


def build_note_track(notes, ppq, name, channel=0):
    events = []
    for n in notes:
        st = int(round(n['start'] * ppq))
        et = int(round((n['start'] + n['dur']) * ppq))
        et = max(st + 1, et)
        vel = max(1, min(127, n['vel']))
        events.append((st, 1, n['pitch'], vel))
        events.append((et, 0, n['pitch'], 0))
    events.sort(key=lambda x: (x[0], x[1]))  # off (0) before on (1) at same tick
    track = mido.MidiTrack()
    track.append(mido.MetaMessage('track_name', name=name, time=0))
    prev = 0
    for tick, kind, pitch, vel in events:
        delta = tick - prev
        prev = tick
        if kind == 1:
            track.append(mido.Message('note_on',  note=pitch, velocity=vel, channel=channel, time=delta))
        else:
            track.append(mido.Message('note_off', note=pitch, velocity=0,   channel=channel, time=delta))
    track.append(mido.MetaMessage('end_of_track', time=0))
    return track


def save_midi(notes, ppq, tempo_events, path, track_name, channel=0):
    out = mido.MidiFile(type=1, ticks_per_beat=ppq)
    out.tracks.append(build_tempo_track(tempo_events))
    out.tracks.append(build_note_track(notes, ppq, track_name, channel))
    out.save(path)
    print(f"  {len(notes):5d} notes  ->  {path}")


# ---------------------------------------------------------------------------
# Voice separation: time-based (default, most accurate)
# ---------------------------------------------------------------------------

def split_by_voice(notes):
    """
    Duration-weighted polyphonic voice separation.

    Algorithm:
      Build a timeline of all note-on / note-off events.
      Between consecutive events, vote for each active note:
        - highest pitch → melody
        - lowest pitch  → bass
        - others        → middle
      (Only when ≥2 notes are simultaneously active; solo notes fall back to pitch.)
    Each note is assigned to the role it accumulated most time in.
    """
    # Events: (time, priority, pitch, note_index)
    # At the same tick: off (0) processed before on (1) so two notes that swap
    # don't appear simultaneous.
    events = []
    for i, note in enumerate(notes):
        events.append((note['start'],                1, note['pitch'], i))   # note_on
        events.append((note['start'] + note['dur'],  0, note['pitch'], i))   # note_off

    events.sort(key=lambda e: (e[0], e[1]))

    active    = {}                                    # note_idx → pitch
    votes     = defaultdict(lambda: [0.0, 0.0, 0.0]) # [melody, middle, bass]
    prev_time = None

    for time, priority, pitch, idx in events:
        # Accumulate votes for the elapsed interval (only when >1 note active)
        if active and prev_time is not None and prev_time != time and len(active) > 1:
            duration       = time - prev_time
            pitches_sorted = sorted(active.values())
            hi, lo         = pitches_sorted[-1], pitches_sorted[0]
            for nidx, npitch in active.items():
                if npitch >= hi:
                    votes[nidx][0] += duration   # melody
                elif npitch <= lo:
                    votes[nidx][2] += duration   # bass
                else:
                    votes[nidx][1] += duration   # middle

        prev_time = time

        if priority == 1:        # note_on
            active[idx] = pitch
        else:                    # note_off
            active.pop(idx, None)

    # Final interval (notes that never ended within the file — shouldn't happen but guard it)
    if active and len(active) > 1:
        pitches_sorted = sorted(active.values())
        hi, lo         = pitches_sorted[-1], pitches_sorted[0]
        for nidx, npitch in active.items():
            if npitch >= hi:
                votes[nidx][0] += 1
            elif npitch <= lo:
                votes[nidx][2] += 1
            else:
                votes[nidx][1] += 1

    melody, middle, bass = [], [], []
    for i, note in enumerate(notes):
        v = votes[i]
        if sum(v) == 0:
            # Always played solo — classify by pitch
            (bass if note['pitch'] < 48 else melody).append(note)
        else:
            dominant = v.index(max(v))
            [melody, middle, bass][dominant].append(note)

    return melody, middle, bass


# ---------------------------------------------------------------------------
# Voice separation: fixed pitch ranges
# ---------------------------------------------------------------------------

def split_by_range(notes, bass_max=47, melody_min=60):
    """
    Split by static MIDI pitch thresholds.
      bass_max:    highest bass note (default 47 = B2)
      melody_min:  lowest melody note (default 60 = C4)
    """
    melody = [n for n in notes if n['pitch'] >= melody_min]
    bass   = [n for n in notes if n['pitch'] <= bass_max]
    middle = [n for n in notes if bass_max < n['pitch'] < melody_min]
    return melody, middle, bass


# ---------------------------------------------------------------------------
# Voice separation: percentile auto-split
# ---------------------------------------------------------------------------

def split_by_percentile(notes):
    """
    Automatically finds split points from the pitch distribution.
    Bottom 33% → bass, top 33% → melody, middle 33% → middle.
    """
    pitches = sorted(n['pitch'] for n in notes)
    n       = len(pitches)
    lo_cut  = pitches[n // 3]
    hi_cut  = pitches[(n * 2) // 3]
    melody  = [n for n in notes if n['pitch'] > hi_cut]
    bass    = [n for n in notes if n['pitch'] <= lo_cut]
    middle  = [n for n in notes if lo_cut < n['pitch'] <= hi_cut]
    return melody, middle, bass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description='Split a MIDI file into melody / middle / bass voices'
    )
    ap.add_argument('input',
                    help='Input MIDI file (e.g. song.mid)')
    ap.add_argument('--method', choices=['voice', 'range', 'percentile'],
                    default='voice',
                    help='Split method: voice (default) | range | percentile')
    ap.add_argument('--bass-max', type=int, default=47,
                    help='Highest bass pitch for range method (default 47 = B2)')
    ap.add_argument('--melody-min', type=int, default=60,
                    help='Lowest melody pitch for range method (default 60 = C4)')
    ap.add_argument('--out-dir', default=None,
                    help='Output directory (default: same folder as input)')
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: file not found: {args.input}")
        sys.exit(1)

    print(f"Reading: {args.input}")
    mid   = mido.MidiFile(args.input)
    notes, ppq = collect_notes(mid)
    tempo_events = collect_tempo_events(mid)

    pitches = [n['pitch'] for n in notes]
    print(f"  {len(notes)} notes, PPQ={ppq}, "
          f"pitch range {min(pitches)}–{max(pitches)}")

    if args.method == 'voice':
        print("  Method: time-based voice separation (default)")
        melody, middle, bass = split_by_voice(notes)
    elif args.method == 'range':
        print(f"  Method: range split  (bass <= {args.bass_max}, melody >= {args.melody_min})")
        melody, middle, bass = split_by_range(notes, args.bass_max, args.melody_min)
    else:
        print("  Method: percentile auto-split (33/33/33)")
        melody, middle, bass = split_by_percentile(notes)

    # Build output paths
    base_dir  = args.out_dir or os.path.dirname(args.input) or '.'
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    os.makedirs(base_dir, exist_ok=True)

    melody_path = os.path.join(base_dir, base_name + '_melody.mid')
    middle_path = os.path.join(base_dir, base_name + '_middle.mid')
    bass_path   = os.path.join(base_dir, base_name + '_bass.mid')

    print()
    save_midi(melody, ppq, tempo_events, melody_path, 'Melody',  channel=0)
    save_midi(middle, ppq, tempo_events, middle_path, 'Middle',  channel=1)
    save_midi(bass,   ppq, tempo_events, bass_path,   'Bass',    channel=2)

    total = len(melody) + len(middle) + len(bass)
    print(f"\n  Total split: {len(melody)} melody + {len(middle)} middle + "
          f"{len(bass)} bass = {total}  (input: {len(notes)})")
    print("Done.")


if __name__ == '__main__':
    main()
