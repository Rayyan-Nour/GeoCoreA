#!/usr/bin/env bash
# GeoCore Analytics Studio - macOS / Linux launcher
set -e
cd "$(dirname "$0")"
command -v python3 >/dev/null || { echo "Python 3.10+ required"; exit 1; }
python3 -c "import PyQt6" 2>/dev/null || {
  echo "First run: installing dependencies..."
  python3 -m pip install -r requirements.txt
}
echo "Starting GeoCore Analytics Studio..."
python3 -m app.main
