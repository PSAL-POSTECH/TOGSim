"""Where the TOGSim tree is: $TOGSIM_DIR, else the checkout this package sits in (frontend/ under its root)."""
import os
from pathlib import Path


def root() -> Path:
    """The TOGSim root: the directory that holds include/, kernels/, configs/ and build/."""
    given = os.environ.get("TOGSIM_DIR")
    return Path(given).resolve() if given else Path(__file__).resolve().parents[2]


def kernel_source(stem: str) -> Path:
    return root() / "kernels" / f"{stem}_tog.cpp"


def include_dir() -> Path:
    return root() / "include"


def simulator() -> Path:
    return root() / "build" / "bin" / "Simulator"
