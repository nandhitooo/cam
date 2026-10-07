"""Non-blocking camera reader used by the webcam application.

Reading a frame inline (read -> filter -> draw -> show) makes the preview fall
further and further behind reality whenever a filter is expensive: while the
filter works, the driver keeps filling its buffer with frames that are already
stale by the time they are shown. Reading on a separate thread fixes that in two
ways at once:

* the driver buffer is drained continuously, so only the newest frame is kept and
  outdated ones are discarded instead of displayed late;
* ``read()`` then waits for the *next* frame instead of returning the same one
  again, so a fast consumer does not re-render a stale frame (which would waste a
  whole core and make the on-screen FPS meaningless).

The wrapper is deliberately transparent: ``read()`` and ``release()`` behave like
``cv2.VideoCapture``, and any other attribute (``get``, ``set``, ``isOpened``,
``getBackendName``, ...) is forwarded to the wrapped capture.
"""

from __future__ import annotations

import threading
import time

import numpy as np

Frame = np.ndarray


class ThreadedCamera:
    """Grab frames on a background thread and serve the most recent one."""

    def __init__(self, capture, threaded: bool = True, first_frame_timeout: float = 5.0,
                 failure_limit: int = 30) -> None:
        self._capture = capture
        self.threaded = threaded
        self.first_frame_timeout = first_frame_timeout
        self.failure_limit = max(1, int(failure_limit))
        self._lock = threading.Lock()
        self._frame: Frame | None = None
        self._errors = 0
        self._stopped = threading.Event()
        self._new_frame = threading.Event()
        self._thread: threading.Thread | None = None
        self.frames_grabbed = 0
        self.frames_served = 0
        self._delivered = 0

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> "ThreadedCamera":
        if self.threaded and self._thread is None:
            self._thread = threading.Thread(target=self._grab_loop, name="frame-grabber", daemon=True)
            self._thread.start()
        return self

    def _grab_loop(self) -> None:
        while not self._stopped.is_set():
            ok, frame = self._capture.read()
            if ok and frame is not None:
                with self._lock:
                    self._frame = frame
                    self.frames_grabbed += 1
                    self._errors = 0
                self._new_frame.set()
            else:
                self._errors += 1
                if self._errors >= self.failure_limit:
                    break  # device is gone; stop spinning on it
                time.sleep(0.01)

    def release(self) -> None:
        self._stopped.set()
        self._new_frame.set()  # wake up anyone blocked in read()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)
        self._capture.release()

    # -- reading -----------------------------------------------------------

    def read(self, timeout: float | None = None) -> tuple[bool, Frame | None]:
        """Wait for the next frame and return it.

        Frames that piled up while the caller was busy are skipped: the caller
        gets the newest one. Returns ``(False, None)`` when the camera failed or
        no frame arrived within ``timeout`` (default ``first_frame_timeout``).
        """
        if not self.threaded:
            ok, frame = self._capture.read()
            if ok and frame is not None:
                self.frames_grabbed += 1
                self.frames_served += 1
            return ok, frame

        if self._thread is None:
            self.start()
        wait = self.first_frame_timeout if timeout is None else timeout
        deadline = time.perf_counter() + wait
        while True:
            if self._stopped.is_set():
                return False, None
            with self._lock:
                if self._frame is not None and self.frames_grabbed > self._delivered:
                    self._delivered = self.frames_grabbed
                    self.frames_served += 1
                    return True, self._frame
                errors = self._errors
            if errors >= self.failure_limit:
                return False, None
            if time.perf_counter() >= deadline:
                return False, None
            self._new_frame.wait(0.002)
            self._new_frame.clear()

    @property
    def frames_dropped(self) -> int:
        """Frames the camera produced that were never shown (we were too slow)."""
        with self._lock:
            return max(0, self.frames_grabbed - self.frames_served)

    @property
    def last_error_count(self) -> int:
        return self._errors

    def __getattr__(self, item: str):
        # Forward everything else to the wrapped capture. Guard `_capture` so a
        # missing attribute can never recurse back into __getattr__.
        if item == "_capture":
            raise AttributeError(item)
        return getattr(self._capture, item)
