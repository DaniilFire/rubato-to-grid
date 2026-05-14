#!/usr/bin/env python3
# rubato_to_grid_auto.py
#
# Auto-search version.
#
# Goal:
#   Input:  off-grid/rubato MIDI that already sounds good.
#   Output: grid-looking MIDI + tempo map that sounds close to the input.
#
# Why this exists:
#   A single setting like grid-div/group-ms can easily look ugly in FL.
#   This script tries many parameter combinations, compares each output to the
#   original MIDI in real time, scores visual readability + timing similarity,
#   and exports the best variants.
#
# Install:
#   py -m pip install mido
#
# Basic:
#   py .\rubato_to_grid_auto.py "1.mid" --bpm 118
#
# FL Studio:
#   Drag the output .mid into FL and enable tempo changes / tempo map import.

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import mido
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: mido\n"
        "Install it with:\n"
        "  py -m pip install mido\n"
    ) from exc


@dataclass
class Note:
    idx: int
    note: int
    channel: int
    velocity: int
    start_tick: int
    end_tick: int
    start_sec: float
    end_sec: float


@dataclass
class OutNote:
    idx: int
    note: int
    channel: int
    velocity: int
    start_tick: int
    length_ticks: int
    orig_start_sec: float
    orig_end_sec: float


@dataclass
class Candidate:
    name: str
    grid_div: int
    group_ms: float
    preserve_roll: bool
    max_roll_ms: float
    max_note_beats: float
    out_notes: List[OutNote]
    tempo_events: List[Tuple[int, int]]
    timing_rms_ms: float
    timing_p95_ms: float
    duration_rms_ms: float
    tempo_event_count: int
    tempo_jump_rms: float
    tempo_range: float
    visual_penalty: float
    score_balanced: float
    score_readable: float
    score_accurate: float


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def rms(values: List[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(v * v for v in values) / len(values))


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    k = int(round((len(vals) - 1) * p))
    return vals[max(0, min(len(vals) - 1, k))]


def bpm_from_tempo(tempo: int) -> float:
    return 60000000.0 / float(tempo)


def extract_notes(mid: "mido.MidiFile") -> List[Note]:
    ppq = mid.ticks_per_beat
    merged = mido.merge_tracks(mid.tracks)

    current_tick = 0
    current_sec = 0.0
    tempo = 500000  # default 120 BPM

    active: Dict[Tuple[int, int], List[Tuple[int, float, int]]] = defaultdict(list)
    notes: List[Note] = []
    idx = 0

    for msg in merged:
        current_sec += mido.tick2second(msg.time, ppq, tempo)
        current_tick += msg.time

        if msg.type == "set_tempo":
            tempo = msg.tempo
            continue

        if msg.type == "note_on" and msg.velocity > 0:
            active[(msg.channel, msg.note)].append((current_tick, current_sec, msg.velocity))
            continue

        is_note_off = msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0)
        if is_note_off:
            key = (msg.channel, msg.note)
            if active[key]:
                start_tick, start_sec, velocity = active[key].pop(0)
                if current_tick > start_tick:
                    notes.append(
                        Note(
                            idx=idx,
                            note=msg.note,
                            channel=msg.channel,
                            velocity=velocity,
                            start_tick=start_tick,
                            end_tick=current_tick,
                            start_sec=start_sec,
                            end_sec=current_sec,
                        )
                    )
                    idx += 1

    return notes


def group_notes_by_start(notes: List[Note], group_ms: float) -> List[List[Note]]:
    if not notes:
        return []

    notes_sorted = sorted(notes, key=lambda n: (n.start_sec, n.note))
    groups: List[List[Note]] = []

    current = [notes_sorted[0]]
    anchor_sec = notes_sorted[0].start_sec

    for note in notes_sorted[1:]:
        if (note.start_sec - anchor_sec) * 1000.0 <= group_ms:
            current.append(note)
        else:
            groups.append(current)
            current = [note]
            anchor_sec = note.start_sec

    groups.append(current)
    return groups


def convert_to_grid(
    notes: List[Note],
    ppq: int,
    base_bpm: float,
    grid_div: int,
    group_ms: float,
    preserve_roll: bool,
    max_roll_ms: float,
    max_note_beats: float,
    min_note_ms: float,
    tiny_gap_ticks: int,
) -> Tuple[List[OutNote], List[Tuple[int, float]]]:
    groups = group_notes_by_start(notes, group_ms)
    if not groups:
        return [], []

    first_sec = min(n.start_sec for n in notes)
    beat_sec = 60.0 / base_bpm

    # 1=beat, 2=eighth, 3=triplet-ish, 4=sixteenth, 6=sextuplet-ish
    step_ticks = max(1, int(round(ppq / float(grid_div))))

    out_notes: List[OutNote] = []
    anchor_pairs: List[Tuple[int, float]] = []

    max_roll_ticks = int(round((max_roll_ms / 1000.0) / beat_sec * ppq))
    min_len_ticks = max(1, int(round((min_note_ms / 1000.0) / beat_sec * ppq)))

    for group in groups:
        group_anchor_sec = min(n.start_sec for n in group)
        rel_sec = group_anchor_sec - first_sec

        nominal_ticks = rel_sec / beat_sec * ppq
        grid_anchor_tick = int(round(nominal_ticks / step_ticks) * step_ticks)

        anchor_pairs.append((grid_anchor_tick, rel_sec))

        for n in group:
            roll_ticks = 0
            if preserve_roll:
                roll_sec = max(0.0, n.start_sec - group_anchor_sec)
                roll_ticks = int(round(roll_sec / beat_sec * ppq))
                roll_ticks = min(roll_ticks, max_roll_ticks)

            new_start = max(0, grid_anchor_tick + roll_ticks)

            duration_sec = max(0.001, n.end_sec - n.start_sec)
            new_len = int(round(duration_sec / beat_sec * ppq))
            new_len = max(min_len_ticks, new_len)

            if max_note_beats > 0:
                max_len = int(round(max_note_beats * ppq))
                new_len = min(new_len, max(1, max_len - tiny_gap_ticks))

            out_notes.append(
                OutNote(
                    idx=n.idx,
                    note=n.note,
                    channel=n.channel,
                    velocity=max(1, min(127, n.velocity)),
                    start_tick=new_start,
                    length_ticks=max(1, new_len),
                    orig_start_sec=n.start_sec,
                    orig_end_sec=n.end_sec,
                )
            )

    return out_notes, anchor_pairs


def build_tempo_events(
    anchor_pairs: List[Tuple[int, float]],
    ppq: int,
    base_bpm: float,
    min_bpm: float,
    max_bpm: float,
    smooth_window: int = 1,
) -> List[Tuple[int, int]]:
    if not anchor_pairs:
        return [(0, mido.bpm2tempo(base_bpm))]

    sorted_pairs = sorted(anchor_pairs, key=lambda x: (x[0], x[1]))

    # Collapse duplicate grid ticks.
    unique: List[Tuple[int, float]] = []
    seen = set()
    for tick, sec in sorted_pairs:
        if tick not in seen:
            unique.append((tick, sec))
            seen.add(tick)

    if len(unique) < 2:
        return [(0, mido.bpm2tempo(base_bpm))]

    raw_segments: List[Tuple[int, float]] = []

    for i in range(len(unique) - 1):
        tick0, sec0 = unique[i]
        tick1, sec1 = unique[i + 1]

        dticks = tick1 - tick0
        dsec = sec1 - sec0

        if dticks <= 0 or dsec <= 0:
            continue

        beats = dticks / float(ppq)
        bpm = 60.0 * beats / dsec
        bpm = clamp(bpm, min_bpm, max_bpm)
        raw_segments.append((tick0, bpm))

    if not raw_segments:
        return [(0, mido.bpm2tempo(base_bpm))]

    # Optional tiny moving average. Default 1 = no smoothing.
    smoothed: List[Tuple[int, float]] = []
    for i, (tick, bpm) in enumerate(raw_segments):
        lo = max(0, i - smooth_window + 1)
        hi = min(len(raw_segments), i + smooth_window)
        vals = [raw_segments[j][1] for j in range(lo, hi)]
        smoothed.append((tick, sum(vals) / len(vals)))

    events: Dict[int, int] = {}

    # Default base tempo at start.
    events[0] = mido.bpm2tempo(base_bpm)

    for tick, bpm in smoothed:
        events[int(tick)] = mido.bpm2tempo(bpm)

    # Continue after last anchor at base tempo.
    events[int(unique[-1][0])] = mido.bpm2tempo(base_bpm)

    return sorted(events.items(), key=lambda x: x[0])


def seconds_at_tick(tick: int, ppq: int, tempo_events: List[Tuple[int, int]]) -> float:
    if not tempo_events:
        return mido.tick2second(tick, ppq, mido.bpm2tempo(120))

    events = sorted(tempo_events, key=lambda x: x[0])
    total = 0.0
    last_tick = 0
    tempo = events[0][1]

    if events[0][0] > 0:
        tempo = events[0][1]

    for ev_tick, ev_tempo in events:
        if ev_tick <= 0:
            tempo = ev_tempo
            last_tick = ev_tick
            continue

        if tick <= ev_tick:
            total += mido.tick2second(tick - last_tick, ppq, tempo)
            return total

        total += mido.tick2second(ev_tick - last_tick, ppq, tempo)
        tempo = ev_tempo
        last_tick = ev_tick

    total += mido.tick2second(tick - last_tick, ppq, tempo)
    return total


def score_candidate(
    name: str,
    grid_div: int,
    group_ms: float,
    preserve_roll: bool,
    max_roll_ms: float,
    max_note_beats: float,
    out_notes: List[OutNote],
    tempo_events: List[Tuple[int, int]],
    ppq: int,
) -> Candidate:
    start_errors = []
    duration_errors = []

    for n in out_notes:
        out_start_sec = seconds_at_tick(n.start_tick, ppq, tempo_events)
        out_end_sec = seconds_at_tick(n.start_tick + n.length_ticks, ppq, tempo_events)

        start_errors.append((out_start_sec - n.orig_start_sec) * 1000.0)
        orig_dur = max(0.001, n.orig_end_sec - n.orig_start_sec)
        out_dur = max(0.001, out_end_sec - out_start_sec)
        duration_errors.append((out_dur - orig_dur) * 1000.0)

    timing_rms_ms = rms(start_errors)
    timing_p95_ms = percentile([abs(x) for x in start_errors], 0.95)
    duration_rms_ms = rms(duration_errors)

    bpms = [bpm_from_tempo(t) for _, t in tempo_events]
    if len(bpms) >= 2:
        jumps = [bpms[i + 1] - bpms[i] for i in range(len(bpms) - 1)]
        tempo_jump_rms = rms(jumps)
        tempo_range = max(bpms) - min(bpms)
    else:
        tempo_jump_rms = 0.0
        tempo_range = 0.0

    # Visual/readability penalty:
    # - finer grid = visually busier
    # - too many tempo events = messy tempo map
    # - preserve roll = sounds human but may look less grid-clean
    # - very small groups = many separate events
    grid_penalty = {1: 0.0, 2: 7.0, 3: 10.0, 4: 18.0, 6: 24.0, 8: 28.0}.get(grid_div, 20.0)
    group_penalty = max(0.0, (70.0 - group_ms) * 0.08)
    roll_penalty = 6.0 if preserve_roll else 0.0
    tempo_count_penalty = len(tempo_events) * 0.05

    visual_penalty = grid_penalty + group_penalty + roll_penalty + tempo_count_penalty

    # Three goals:
    # accurate: closer to original timing
    # readable: cleaner FL view / less tiny grid
    # balanced: useful default
    score_accurate = (
        timing_rms_ms * 1.00
        + timing_p95_ms * 0.15
        + duration_rms_ms * 0.05
        + tempo_jump_rms * 0.15
        + tempo_range * 0.03
        + visual_penalty * 0.25
    )

    score_readable = (
        timing_rms_ms * 0.55
        + timing_p95_ms * 0.08
        + duration_rms_ms * 0.03
        + tempo_jump_rms * 0.25
        + tempo_range * 0.05
        + visual_penalty * 1.35
    )

    score_balanced = (
        timing_rms_ms * 0.75
        + timing_p95_ms * 0.10
        + duration_rms_ms * 0.04
        + tempo_jump_rms * 0.20
        + tempo_range * 0.04
        + visual_penalty * 0.85
    )

    return Candidate(
        name=name,
        grid_div=grid_div,
        group_ms=group_ms,
        preserve_roll=preserve_roll,
        max_roll_ms=max_roll_ms,
        max_note_beats=max_note_beats,
        out_notes=out_notes,
        tempo_events=tempo_events,
        timing_rms_ms=timing_rms_ms,
        timing_p95_ms=timing_p95_ms,
        duration_rms_ms=duration_rms_ms,
        tempo_event_count=len(tempo_events),
        tempo_jump_rms=tempo_jump_rms,
        tempo_range=tempo_range,
        visual_penalty=visual_penalty,
        score_balanced=score_balanced,
        score_readable=score_readable,
        score_accurate=score_accurate,
    )


def write_midi(out_path: Path, ppq: int, out_notes: List[OutNote], tempo_events: List[Tuple[int, int]]) -> None:
    mid = mido.MidiFile(ticks_per_beat=ppq)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    events: List[Tuple[int, int, object]] = []

    for tick, tempo in tempo_events:
        events.append((max(0, int(tick)), 0, mido.MetaMessage("set_tempo", tempo=int(tempo), time=0)))

    for n in out_notes:
        start = max(0, int(n.start_tick))
        end = max(start + 1, int(n.start_tick + n.length_ticks))

        events.append((start, 2, mido.Message("note_on", note=n.note, velocity=n.velocity, channel=n.channel, time=0)))
        events.append((end, 1, mido.Message("note_off", note=n.note, velocity=0, channel=n.channel, time=0)))

    events.sort(key=lambda x: (x[0], x[1]))

    last_tick = 0
    for abs_tick, _order, msg in events:
        delta = abs_tick - last_tick
        msg.time = max(0, int(delta))
        track.append(msg)
        last_tick = abs_tick

    track.append(mido.MetaMessage("end_of_track", time=0))
    mid.save(out_path)


def write_report(report_path: Path, candidates: List[Candidate]) -> None:
    fields = [
        "name",
        "grid_div",
        "group_ms",
        "preserve_roll",
        "max_roll_ms",
        "max_note_beats",
        "timing_rms_ms",
        "timing_p95_ms",
        "duration_rms_ms",
        "tempo_event_count",
        "tempo_jump_rms",
        "tempo_range",
        "visual_penalty",
        "score_balanced",
        "score_readable",
        "score_accurate",
    ]

    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for c in candidates:
            writer.writerow({k: getattr(c, k) for k in fields})


def build_candidates(
    notes: List[Note],
    ppq: int,
    base_bpm: float,
    min_bpm: float,
    max_bpm: float,
    mode: str,
) -> List[Candidate]:
    tiny_gap_ticks = max(1, int(ppq / 64))

    if mode == "fast":
        grid_divs = [1, 2, 3]
        group_mss = [45, 70, 95]
        roll_modes = [(False, 0), (True, 60)]
    else:
        grid_divs = [1, 2, 3, 4, 6]
        group_mss = [25, 35, 50, 70, 90, 120]
        roll_modes = [(False, 0), (True, 35), (True, 70), (True, 110)]

    # Do not trim by default for auto search. Trimming makes comparison unfair.
    max_note_beats_options = [0.0]

    candidates: List[Candidate] = []
    count = 0

    for grid_div in grid_divs:
        for group_ms in group_mss:
            for preserve_roll, max_roll_ms in roll_modes:
                for max_note_beats in max_note_beats_options:
                    out_notes, anchors = convert_to_grid(
                        notes=notes,
                        ppq=ppq,
                        base_bpm=base_bpm,
                        grid_div=grid_div,
                        group_ms=group_ms,
                        preserve_roll=preserve_roll,
                        max_roll_ms=max_roll_ms,
                        max_note_beats=max_note_beats,
                        min_note_ms=25.0,
                        tiny_gap_ticks=tiny_gap_ticks,
                    )

                    if not out_notes or not anchors:
                        continue

                    tempo_events = build_tempo_events(
                        anchor_pairs=anchors,
                        ppq=ppq,
                        base_bpm=base_bpm,
                        min_bpm=min_bpm,
                        max_bpm=max_bpm,
                        smooth_window=1,
                    )

                    count += 1
                    name = (
                        f"cand_{count:03d}_g{grid_div}_grp{int(group_ms)}"
                        f"_{'roll'+str(int(max_roll_ms)) if preserve_roll else 'noroll'}"
                    )

                    candidates.append(
                        score_candidate(
                            name=name,
                            grid_div=grid_div,
                            group_ms=group_ms,
                            preserve_roll=preserve_roll,
                            max_roll_ms=max_roll_ms,
                            max_note_beats=max_note_beats,
                            out_notes=out_notes,
                            tempo_events=tempo_events,
                            ppq=ppq,
                        )
                    )

    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-convert rubato/off-grid MIDI into grid MIDI + tempo map."
    )

    parser.add_argument("input", help="Input MIDI file")
    parser.add_argument("--bpm", type=float, default=118.0, help="Base BPM")
    parser.add_argument("--min-bpm", type=float, default=45.0, help="Min tempo map BPM")
    parser.add_argument("--max-bpm", type=float, default=220.0, help="Max tempo map BPM")
    parser.add_argument("--mode", choices=["full", "fast"], default="full", help="Search mode")
    parser.add_argument("--prefix", default=None, help="Output filename prefix")
    parser.add_argument("--export-top", type=int, default=0, help="Also export top N balanced candidates")

    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")

    prefix = args.prefix or in_path.with_suffix("").name

    mid = mido.MidiFile(in_path)
    ppq = mid.ticks_per_beat

    notes = extract_notes(mid)
    if not notes:
        raise SystemExit("No notes found in the MIDI file.")

    candidates = build_candidates(
        notes=notes,
        ppq=ppq,
        base_bpm=args.bpm,
        min_bpm=args.min_bpm,
        max_bpm=args.max_bpm,
        mode=args.mode,
    )

    if not candidates:
        raise SystemExit("No candidates generated.")

    candidates_by_balanced = sorted(candidates, key=lambda c: c.score_balanced)
    candidates_by_readable = sorted(candidates, key=lambda c: c.score_readable)
    candidates_by_accurate = sorted(candidates, key=lambda c: c.score_accurate)

    best_balanced = candidates_by_balanced[0]
    best_readable = candidates_by_readable[0]
    best_accurate = candidates_by_accurate[0]

    outputs = [
        (f"{prefix}_AUTO_BALANCED.mid", best_balanced),
        (f"{prefix}_AUTO_READABLE.mid", best_readable),
        (f"{prefix}_AUTO_ACCURATE.mid", best_accurate),
    ]

    for filename, cand in outputs:
        write_midi(Path(filename), ppq, cand.out_notes, cand.tempo_events)

    if args.export_top > 0:
        for idx, cand in enumerate(candidates_by_balanced[: args.export_top], start=1):
            write_midi(Path(f"{prefix}_TOP_{idx:02d}_{cand.name}.mid"), ppq, cand.out_notes, cand.tempo_events)

    report_path = Path(f"{prefix}_auto_report.csv")
    write_report(report_path, sorted(candidates, key=lambda c: c.score_balanced))

    print("Done.")
    print(f"Input: {in_path}")
    print(f"Notes: {len(notes)}")
    print(f"Candidates checked: {len(candidates)}")
    print()
    print("Exported:")
    print(f"  {prefix}_AUTO_BALANCED.mid")
    print(f"  {prefix}_AUTO_READABLE.mid")
    print(f"  {prefix}_AUTO_ACCURATE.mid")
    print(f"  {report_path}")
    print()
    print("Best balanced:")
    print(
        f"  {best_balanced.name} | grid_div={best_balanced.grid_div}, "
        f"group_ms={best_balanced.group_ms}, preserve_roll={best_balanced.preserve_roll}, "
        f"timing_rms={best_balanced.timing_rms_ms:.2f}ms, "
        f"visual_penalty={best_balanced.visual_penalty:.2f}, "
        f"tempo_events={best_balanced.tempo_event_count}"
    )
    print()
    print("FL Studio:")
    print("  Drag the exported MIDI into FL Studio and enable tempo changes.")
    print("  Start with AUTO_READABLE if Piano Roll view matters most.")
    print("  Start with AUTO_ACCURATE if matching the original performance matters most.")


if __name__ == "__main__":
    main()
