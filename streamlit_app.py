"""Streamlit Cloud entrypoint.

Streamlit Cloud auto-discovers this file at repo root. It re-exports the
real dashboard module so deployment is one click.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from timestone.interfaces.web.dashboard import main

if __name__ == "__main__":
    main()
