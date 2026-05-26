@echo off
:: One-time setup for the local FastVideo checkout.
::
:: Reuses MIND's venv (torch + CUDA already installed) and pip-installs this
:: repo in editable mode so consumers (e.g. game-streaming-poc) get the
:: fastvideo.entrypoints.streaming_generator and matrixgame.utils modules
:: that the public PyPI 'fastvideo' wheel does NOT ship.
::
:: Override the target venv with an env var if MIND's is not the one to use:
::   set FV_VENV_PY=C:\path\to\python.exe
::   setup.bat

setlocal enableextensions enabledelayedexpansion
cd /d "%~dp0"

:: Drop inherited venv vars before spawning a different interpreter —
:: otherwise that interpreter sees a stale VIRTUAL_ENV / PYTHONHOME /
:: PYTHONPATH and hits an SRE / stdlib mismatch on sys / re import.
set VIRTUAL_ENV=
set PYTHONHOME=
set PYTHONPATH=

if not defined FV_VENV_PY set FV_VENV_PY=C:\workspace\world\MIND\.venv\Scripts\python.exe
if not exist "!FV_VENV_PY!" (
    echo ERROR: python not found at !FV_VENV_PY!
    echo Set FV_VENV_PY to a venv that already has torch + CUDA installed.
    exit /b 2
)

echo === Using python: !FV_VENV_PY! ===
"!FV_VENV_PY!" -c "import sys, torch; print('python', sys.version); print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || (echo ERROR: torch import failed in target venv & exit /b 2)

echo.
echo === Installing FastVideo from %CD% in editable mode ===
:: --upgrade-strategy only-if-needed: pip leaves torch / heavy deps alone if
:: the installed version already satisfies the pin in pyproject.toml. The
:: editable install means future `git pull` in this directory is picked up
:: by the venv with no reinstall.
"!FV_VENV_PY!" -m pip install --upgrade-strategy only-if-needed -e . || (echo ERROR: fastvideo editable install failed & exit /b 1)

echo.
echo === Verifying torch + CUDA survived ===
"!FV_VENV_PY!" -c "import torch; assert torch.cuda.is_available(), 'CUDA gone after install'; print('torch still CUDA-ok:', torch.__version__)" || (echo ERROR: torch CUDA broke after install — pin torch in the target venv and retry & exit /b 1)

echo.
echo === Smoke test imports ===
"!FV_VENV_PY!" -c "from fastvideo.entrypoints.streaming_generator import StreamingVideoGenerator; from fastvideo.models.dits.matrixgame2.utils import CAMERA_MAP, KEYBOARD_MAP_2, KEYBOARD_MAP_4, KEYBOARD_MAP_7, expand_action_to_frames; print('imports OK')" || (echo ERROR: import smoke test failed — fastvideo source layout has changed & exit /b 1)

echo.
echo --- setup done ---
echo Installed editable: %CD%
echo Python:             !FV_VENV_PY!
echo.
echo The fastvideo package now resolves to this checkout. Re-run
echo game-streaming-poc\setup.bat afterwards if it was run earlier against
echo the PyPI wheel (the editable install overrides it cleanly).
endlocal
exit /b 0
