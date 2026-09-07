"""Dependency-free command dispatch; help never initializes a model."""
import ast
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]


def dispatch(script: str, *, paid: bool = False) -> None:
    path = ROOT / script
    if "--help" in sys.argv or "-h" in sys.argv:
        print(ast.get_docstring(ast.parse(path.read_text())) or script)
        print("\nOutputs default to data/runs/. Set ENVIRONMENTS_DATA_ROOT to override.")
        return
    if paid and os.environ.get("ENVIRONMENTS_READ_ONLY_DATA") == "1":
        raise SystemExit("Published-data analysis mode cannot start experiments.")
    sys.path.insert(0, str(path.parent))
    sys.path.insert(0, str(ROOT))
    sys.argv[0] = str(path)
    runpy.run_path(str(path), run_name="__main__")
