@echo off
REM Put your MIDI next to this file and rename it to 1.mid
py -m pip install mido
py .\rubato_to_grid_auto.py "1.mid" --bpm 118 --export-top 5
pause
