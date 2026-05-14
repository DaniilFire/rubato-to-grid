@echo off
REM Put your MIDI next to this file and rename it to 1.mid
py -m pip install mido numpy scipy
py .\rubato_to_grid_v3.py "1.mid"
pause
