#!/usr/bin/env python3
"""
midi_arrange.py  --  Arrange a MIDI file using Claude AI.

Input:  1_grid.mid  (output of rubato_to_grid_v3.py)
Output: 1_grid_arranged.mid

What it does
  Sends melody chunks to Claude API with an Einaudi-style arrangement prompt.
  Claude adds left-hand bass/arpeggios and right-hand chord voices.
  Original melody notes are preserved exactly.

Usage:
  python midi_arrange.py 1_grid.mid
  python midi_arrange.py 1_grid.mid output.mid
  python midi_arrange.py 1_grid.mid --bars 8   # test on first 8 bars only
  python midi_arrange.py 1_grid.mid --model claude-sonnet-4-6  # better quality
  python midi_arrange.py 1_grid.mid --style "Bach baroque counterpoint"

Requires:
  pip install anthropic mido
  Set environment variable: ANTHROPIC_API_KEY=sk-ant-...
"""

import argparse
import os
import re
import sys
from collections import defaultdict

import mido

try:
    import anthropic
except ImportError:
    print("ERROR: anthropic not installed. Run: pip install anthropic")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Note name utilities
# ---------------------------------------------------------------------------

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

FLAT_TO_SHARP = {
    'Cb': 'B', 'Db': 'C#', 'Eb': 'D#', 'Fb': 'E',
    'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#',
}

# Krumhansl-Schmuckler profiles
KS_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KS_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


def pitch_to_name(pitch):
    return f"{NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"


def name_to_pitch(name):
    name = name.strip()
    # Normalize flats to sharps
    for flat, sharp in FLAT_TO_SHARP.items():
        if name.upper().startswith(flat.upper()):
            name = sharp + name[len(flat):]
            break
    name = name[0].upper() + name[1:]
    # Try 2-char note name first (e.g. C#), then 1-char
    for prefix_len in (2, 1):
        prefix = name[:prefix_len]
        if prefix in NOTE_NAMES:
            try:
                octave = int(name[prefix_len:])
                return NOTE_NAMES.index(prefix) + (octave + 1) * 12
            except ValueError:
                continue
    raise ValueError(f"Cannot parse note name: {name!r}")


def detect_key(notes):
    histogram = [0.0] * 12
    for n in notes:
        histogram[n['pitch'] % 12] += 1
    total = sum(histogram) or 1
    histogram = [h / total for h in histogram]
    sum_maj = sum(KS_MAJOR)
    sum_min = sum(KS_MINOR)
    norm_maj = [v / sum_maj for v in KS_MAJOR]
    norm_min = [v / sum_min for v in KS_MINOR]
    best_score, best_root, best_mode = -1, 0, 'major'
    for root in range(12):
        for mode, profile in [('major', norm_maj), ('minor', norm_min)]:
            score = sum(histogram[(root + i) % 12] * profile[i] for i in range(12))
            if score > best_score:
                best_score, best_root, best_mode = score, root, mode
    return best_root, best_mode


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
    return sorted(seen.items()) if seen else [(0, 500000)]


def build_tempo_track(tempo_events):
    track = mido.MidiTrack()
    track.append(mido.MetaMessage('track_name', name='Tempo Map', time=0))
    prev = 0
    for tick, tempo in tempo_events:
        track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=tick - prev))
        prev = tick
    track.append(mido.MetaMessage('end_of_track', time=0))
    return track


def build_note_track(notes, ppq, name, channel):
    events = []
    for n in notes:
        st = int(round(n['start'] * ppq))
        et = int(round((n['start'] + n['dur']) * ppq))
        et = max(st + 1, et)
        vel = max(1, min(127, n['vel']))
        events.append((st, 1, n['pitch'], vel))
        events.append((et, 0, n['pitch'], 0))
    events.sort(key=lambda x: (x[0], x[1]))
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


# ---------------------------------------------------------------------------
# Text conversion
# ---------------------------------------------------------------------------

def notes_to_text(notes, offset=0.0):
    """Convert note list to text: 'PitchName Beat Duration Velocity'"""
    lines = []
    for n in notes:
        beat = n['start'] - offset
        lines.append(f"{pitch_to_name(n['pitch'])} {beat:.3f} {n['dur']:.3f} {n['vel']}")
    return '\n'.join(lines)


def text_to_notes(text, offset=0.0, channel=0):
    """Parse 'PitchName Beat Duration Velocity' lines back to note dicts."""
    notes = []
    pattern = re.compile(
        r'^([A-Ga-g][b#]?\d+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)',
        re.MULTILINE
    )
    for m in pattern.finditer(text):
        try:
            pitch = name_to_pitch(m.group(1))
            beat  = float(m.group(2)) + offset
            dur   = max(0.0625, float(m.group(3)))
            vel   = min(127, max(1, int(m.group(4))))
            notes.append({'pitch': pitch, 'vel': vel, 'channel': channel,
                          'start': beat, 'dur': dur})
        except (ValueError, KeyError):
            pass
    return notes


def split_llm_response(response):
    """Extract LEFT_HAND and RIGHT_HAND sections from LLM response."""
    lh_text = ''
    rh_text = ''
    # Normalize section headers
    response = re.sub(r'(?i)left[_ ]hand\s*:', 'LEFT_HAND:', response)
    response = re.sub(r'(?i)right[_ ]hand\s*:', 'RIGHT_HAND:', response)

    if 'LEFT_HAND:' in response:
        after_lh = response.split('LEFT_HAND:', 1)[1]
        if 'RIGHT_HAND:' in after_lh:
            lh_text = after_lh.split('RIGHT_HAND:', 1)[0]
            rh_text = after_lh.split('RIGHT_HAND:', 1)[1]
        else:
            lh_text = after_lh
    if 'RIGHT_HAND:' in response and not rh_text:
        rh_text = response.split('RIGHT_HAND:', 1)[1]
        if 'LEFT_HAND:' in rh_text:
            rh_text = rh_text.split('LEFT_HAND:', 1)[0]

    return lh_text, rh_text


# ---------------------------------------------------------------------------
# Claude API
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert solo piano arranger specializing in the style of Ludovico Einaudi.
Given a melody, add accompaniment: left-hand bass/arpeggios and right-hand chord voices.

RULES:
1. Preserve the melody EXACTLY — do not change, add, or remove melody notes.
2. Style: sparse, lyrical, meditative. Gentle arpeggios, flowing patterns.
3. Left hand: root notes + broken chord arpeggios (quarter/eighth note patterns). Octave 2-3.
4. Right hand additions: soft inner voices or chords below the melody, when appropriate. Octave 4-5.
5. Stay strictly in key. Velocity 35-65 for accompaniment (softer than melody).
6. Duration: bass notes 1-2 beats, arpeggios 0.5 beats.

OUTPUT FORMAT — output ONLY these two sections, nothing else:

LEFT_HAND:
NoteName Beat Duration Velocity
...

RIGHT_HAND:
NoteName Beat Duration Velocity
...

Example format (not real notes):
LEFT_HAND:
Bb2 0.000 2.000 55
F3 0.500 0.500 48
D3 1.000 0.500 45
F3 1.500 0.500 48
RIGHT_HAND:
D4 0.000 0.500 42
"""


def call_claude(melody_text, key_name, bpm, chunk_start, chunk_end,
                style, model):
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set.\n"
            "Get a key at console.anthropic.com, then run:\n"
            "  set ANTHROPIC_API_KEY=sk-ant-..."
        )

    client = anthropic.Anthropic(api_key=api_key)

    user_msg = (
        f"Arrange this piano melody. Key: {key_name}. Tempo: ~{bpm:.0f} BPM. "
        f"Style: {style}.\n"
        f"Beat range: {chunk_start:.1f} – {chunk_end:.1f} (beats are relative, start at 0.0)\n\n"
        f"MELODY:\n{melody_text}\n\n"
        f"Output LEFT_HAND: and RIGHT_HAND: sections."
    )

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    return response.content[0].text, response.usage.input_tokens, response.usage.output_tokens


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(input_path, output_path, args):
    print(f"Reading: {input_path}")
    mid = mido.MidiFile(input_path)
    notes, ppq = collect_notes(mid)
    tempo_events = collect_tempo_events(mid)

    avg_tempo = sum(t for _, t in tempo_events) / len(tempo_events)
    bpm = 60_000_000 / avg_tempo

    key_root, key_mode = detect_key(notes)
    key_name = f"{NOTE_NAMES[key_root]} {key_mode}"
    print(f"  {len(notes)} notes, PPQ={ppq}, BPM~{bpm:.0f}, Key: {key_name}")

    # Split by pitch
    SPLIT = args.split
    rh_notes = [n for n in notes if n['pitch'] >= SPLIT]
    lh_notes = [n for n in notes if n['pitch'] <  SPLIT]
    print(f"  RH (>={SPLIT}): {len(rh_notes)} notes, LH (<{SPLIT}): {len(lh_notes)} notes")

    total_beats = max(n['start'] + n['dur'] for n in notes)
    if args.bars:
        total_beats = min(total_beats, args.bars * 4.0)
        print(f"  Processing first {args.bars} bars ({total_beats:.0f} beats) only")

    chunk_size = args.chunk
    chunks_needed = int((total_beats + chunk_size - 1) // chunk_size)

    # Cost estimate (haiku: $0.80/MTok in, $4/MTok out; sonnet ~10x)
    approx_input_tokens  = len(rh_notes) * 8
    approx_output_tokens = approx_input_tokens * 3
    if 'haiku' in args.model:
        cost = (approx_input_tokens * 0.80 + approx_output_tokens * 4.0) / 1_000_000
    else:
        cost = (approx_input_tokens * 3.0 + approx_output_tokens * 15.0) / 1_000_000
    print(f"  Estimated cost: ~${cost:.3f} ({chunks_needed} chunks x {chunk_size} beats)")

    added_rh = []
    added_lh = []
    total_in_tok = 0
    total_out_tok = 0

    beat = 0.0
    chunk_num = 0
    while beat < total_beats:
        chunk_end = min(beat + chunk_size, total_beats)
        chunk_rh  = [n for n in rh_notes if beat <= n['start'] < chunk_end]

        if not chunk_rh:
            beat = chunk_end
            continue

        chunk_num += 1
        melody_text = notes_to_text(chunk_rh, offset=beat)
        print(f"  Chunk {chunk_num}/{chunks_needed}: "
              f"beats {beat:.1f}-{chunk_end:.1f} ({len(chunk_rh)} notes)...",
              end='', flush=True)

        try:
            response_text, in_tok, out_tok = call_claude(
                melody_text, key_name, bpm, beat, chunk_end,
                args.style, args.model
            )
            total_in_tok  += in_tok
            total_out_tok += out_tok

            lh_text, rh_text = split_llm_response(response_text)
            new_lh = text_to_notes(lh_text, offset=beat, channel=1)
            new_rh = text_to_notes(rh_text, offset=beat, channel=0)

            # Clamp to reasonable ranges
            for n in new_lh:
                n['pitch'] = max(24, min(59, n['pitch']))
            for n in new_rh:
                n['pitch'] = max(SPLIT, min(96, n['pitch']))

            # Remove added RH notes that duplicate melody pitches (same beat + pitch class)
            melody_pcs = defaultdict(set)
            for n in chunk_rh:
                melody_pcs[round(n['start'] * 8)].add(n['pitch'] % 12)
            new_rh = [
                n for n in new_rh
                if n['pitch'] % 12 not in melody_pcs.get(round(n['start'] * 8), set())
            ]

            added_lh.extend(new_lh)
            added_rh.extend(new_rh)
            print(f" +{len(new_lh)} LH, +{len(new_rh)} RH  "
                  f"[{in_tok}+{out_tok} tok]")

        except RuntimeError as e:
            print(f"\n\nFATAL: {e}")
            sys.exit(1)
        except Exception as e:
            print(f" ERROR: {e} (skipping chunk)")

        beat = chunk_end

    # Summary
    actual_cost = (total_in_tok * (0.80 if 'haiku' in args.model else 3.0) +
                   total_out_tok * (4.0  if 'haiku' in args.model else 15.0)) / 1_000_000
    print(f"\n  Tokens used: {total_in_tok} in / {total_out_tok} out  (actual cost ~${actual_cost:.4f})")

    final_rh = rh_notes + added_rh
    final_lh = lh_notes + added_lh
    print(f"  Original: {len(rh_notes)} RH + {len(lh_notes)} LH = {len(notes)}")
    print(f"  Added:    {len(added_rh)} RH + {len(added_lh)} LH")
    print(f"  Final:    {len(final_rh)} RH + {len(final_lh)} LH = {len(final_rh)+len(final_lh)}")

    out = mido.MidiFile(type=1, ticks_per_beat=ppq)
    out.tracks.append(build_tempo_track(tempo_events))
    out.tracks.append(build_note_track(final_rh, ppq, 'Right Hand', channel=0))
    out.tracks.append(build_note_track(final_lh, ppq, 'Left Hand',  channel=1))
    out.save(output_path)
    print(f"Saved: {output_path}")


def main():
    ap = argparse.ArgumentParser(
        description='Arrange MIDI using Claude AI (Einaudi style)'
    )
    ap.add_argument('input',         help='Input MIDI (e.g. 1_grid.mid)')
    ap.add_argument('output', nargs='?', help='Output file (default: input_arranged.mid)')
    ap.add_argument('--split',  type=int,   default=60,
                    help='Pitch split RH/LH (default 60 = C4)')
    ap.add_argument('--chunk',  type=float, default=32.0,
                    help='Beats per API call (default 32 = ~8 bars)')
    ap.add_argument('--bars',   type=int,   default=None,
                    help='Process only first N bars (for testing)')
    ap.add_argument('--style',  default='Ludovico Einaudi minimalist, lyrical, meditative',
                    help='Arrangement style description')
    ap.add_argument('--model',  default='claude-haiku-4-5-20251001',
                    help='Claude model: claude-haiku-4-5-20251001 (fast/cheap) '
                         'or claude-sonnet-4-6 (better quality)')
    args = ap.parse_args()

    if not args.output:
        base = os.path.splitext(args.input)[0]
        args.output = base + '_arranged.mid'

    run(args.input, args.output, args)


if __name__ == '__main__':
    main()
