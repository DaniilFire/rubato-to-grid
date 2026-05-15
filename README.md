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

No parameters required — BPM and meter are auto-detected:

```
python rubato_to_grid_v3.py input.mid
```

Output: `input_grid.mid` with the full tempo map embedded.

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--subdiv N` | 16 | Grid resolution: 4, 8, 12, 16, 32 (16th notes default) |
| `--smooth SIGMA` | 2.0 | Gaussian BPM smoothing in beats |
| `--meter N/D` | auto | Force time signature, e.g. `3/4` |
| `--min-bpm BPM` | 50 | Minimum BPM for beat detection |
| `--max-bpm BPM` | 220 | Maximum BPM for beat detection |
| `--verbose` | off | Print step-by-step details |

### Examples

```
python rubato_to_grid_v3.py piece.mid
python rubato_to_grid_v3.py piece.mid out.mid --subdiv 8
python rubato_to_grid_v3.py piece.mid out.mid --meter 3/4 --smooth 1.5
python rubato_to_grid_v3.py piece.mid out.mid --min-bpm 80 --max-bpm 160 --verbose
```

### Import into FL Studio

Drag the output `.mid` into FL Studio. When prompted, enable **"Import tempo changes"** / **"Use tempo map"** so the variable BPM is preserved.

## Split into Melody / Middle / Bass

`midi_split_voices.py` splits any MIDI into 3 separate files by voice role:

```
python midi_split_voices.py input.mid
```

Output:
- `input_melody.mid` — top voice (highest notes at each moment)
- `input_middle.mid` — inner voices / chords
- `input_bass.mid` — bass voice (lowest notes at each moment)

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--method voice` | default | Time-based: at each moment, highest → melody, lowest → bass |
| `--method range` | — | Fixed pitch ranges (B2 and below = bass, C4 and above = melody) |
| `--method percentile` | — | Auto: splits by pitch percentiles (bottom 33% / middle / top 33%) |
| `--bass-max N` | 47 | Highest bass pitch for `range` method (default 47 = B2) |
| `--melody-min N` | 60 | Lowest melody pitch for `range` method (default 60 = C4) |
| `--out-dir DIR` | same folder | Output directory |

### Examples

```
python midi_split_voices.py song.mid
python midi_split_voices.py song.mid --method range --bass-max 47 --melody-min 72
python midi_split_voices.py song.mid --method percentile --out-dir ./split
```

## Files

| File | Description |
|------|-------------|
| `rubato_to_grid_v3.py` | Main script — beat-aware Viterbi converter |
| `midi_polish.py` | Optional post-processing — 8th-note visual cleanup for piano roll |
| `midi_split_voices.py` | Split MIDI into melody / middle / bass tracks |

## Requirements

- Python 3.8+
- mido >= 1.2.10
- numpy >= 1.21.0 (optional but recommended)
- scipy >= 1.7.0 (optional)
