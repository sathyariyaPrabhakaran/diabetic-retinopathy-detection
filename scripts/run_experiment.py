from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'retina'

if not DATA.exists():
    raise SystemExit('Dataset missing. Put the licensed retinal dataset under data/retina/ with one folder per class.')

cmd = [sys.executable, '-m', 'src.train', '--data-dir', str(DATA)]
raise SystemExit(subprocess.call(cmd, cwd=ROOT))
