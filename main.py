"""Launch the webcam application."""

from __future__ import annotations

from src import app as _app
from src.app import *


def __getattr__(name: str):
    return getattr(_app, name)


if __name__ == "__main__":
    raise SystemExit(_app.main())
