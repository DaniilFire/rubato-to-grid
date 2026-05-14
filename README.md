# Rubato to Grid

Convert rubato/off-grid piano MIDI to a clean grid-aligned MIDI with an accurate tempo map.

Built for producers who extract MIDI from AI-generated audio (Suno, Udio) and need it production-ready for a DAW like FL Studio.

## The Problem

When you extract MIDI from audio, the notes are slightly off-grid — the performance has natural timing drift (rubato). Simply snapping notes to a fixed BPM breaks the melody: notes jump to wrong beats, phrasing is destroyed.

The correct approach is the opposite: **keep the notes exactly where they are, and move the grid to match them** — generating a tempo map that follows the natural rhythm of the performance.

## Result

Tested on a 172-second solo piano piece (786 notes extracted from Suno audio via ai-midi.com):

- **786/786 notes within 20ms of their musical beat position**
- **RMS timing error: 9.3ms**
- Tempo map with 323 tempo events — follows every natural phrase and rubato moment

## How It Works

`rubato_to_grid_v3.py` uses a beat-aware pipeline:

1. **IOI histogram** — finds the real beat period from inter-onset intervals, weighted by note velocity and register
2. **Strong beat detection** — selects structurally important notes (top 35% by weight, min gap = 0.75× beat period) to anchor the tempo map
3. **Viterbi DP** — finds the optimal sequence of beat intervals (1–8 beats per segment) with a smoothness penalty
4. **Gaussian BPM smoothing** — removes jitter from the tempo curve
5. **1/16 grid quantization** — snaps all notes to the final grid

The tempo map moves to the notes. The notes never jump to the wrong beat.

## Installation

```
pip install mido numpy scipy
```

scipy is optional — the script falls back to pure-Python Gaussian smoothing if not installed.

## Usage

```
python rubato_to_grid_v3.py 1.mid
```

Output: `1_grid.mid` — a 2-track MIDI (Right Hand / Left Hand split at C4) with the full tempo map embedded.

### Import into FL Studio

Drag `1_grid.mid` into FL Studio. When prompted, enable **"Import tempo changes"** / **"Use tempo map"** so the variable BPM is preserved.

## Files

| File | Description |
|------|-------------|
| `rubato_to_grid_v3.py` | Main script — beat-aware Viterbi converter |
| `rubato_to_grid_auto.py` | v2 — brute-force parameter search (requires `--bpm`) |
| `midi_polish.py` | Optional post-processing — 8th-note visual cleanup for piano roll |

## v2 vs v3

| | v2 (`rubato_to_grid_auto.py`) | v3 (`rubato_to_grid_v3.py`) |
|---|---|---|
| BPM required | Yes (`--bpm 120`) | No — auto-detected |
| Approach | ~72 candidate grids, pick best | Beat detection + Viterbi DP |
| Output | 3 variants (balanced / readable / accurate) | Single best result |
| Speed | Slower | Fast |

v2 is still useful if you want to explore multiple grid interpretations of an ambiguous recording.

## Requirements

- Python 3.8+
- mido >= 1.2.10
- numpy >= 1.21.0 (optional but recommended)
- scipy >= 1.7.0 (optional)
