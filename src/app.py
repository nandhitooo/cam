#!/usr/bin/env python3
"""Application entry point for the real-time webcam filter app.

Reads frames from a camera with OpenCV, applies one of several artistic
filters in real time, and shows the result in a window. Filters can be
switched with the number keys or the arrow keys, snapshots can be saved
with `s`, and a face tracker can label, blur, pixelate, add sunglasses to, or
zoom into the faces it finds.

Frames are grabbed on a background thread, so a slow filter drops frames
instead of falling behind the camera.

Usage:
    python main.py                 # default camera, 1280x720
    python main.py --camera 1      # pick another device
    python main.py --width 640 --height 480
    python main.py --filter cartoon
    python main.py --face-effect glasses   # start with sunglasses
    python main.py --face-zoom             # start in face-cam mode
    python main.py --face-attributes       # start with age/expression labels
    python main.py --list-filters
    python main.py --self-test     # verify filters + face code, no camera
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

import cv2
import numpy as np

from . import face_attributes as fa
from . import face_tracking as ft
from .threaded_camera import ThreadedCamera

Frame = np.ndarray
FilterFn = Callable[[Frame], Frame]

WINDOW_NAME = "Webcam Filters - Buffy"

# BGR channel gains, baked into per-channel lookup tables (much faster than
# multiplying a float32 copy of every frame).
WARM_GAIN = np.array([0.85, 1.00, 1.20], dtype=np.float32)
COOL_GAIN = np.array([1.25, 1.00, 0.80], dtype=np.float32)


def _gain_lut(gain: np.ndarray) -> Frame:
    values = np.clip(np.arange(256, dtype=np.float32)[:, None] * gain[None, :], 0, 255)
    return values.astype(np.uint8).reshape(1, 256, 3)


WARM_LUT = _gain_lut(WARM_GAIN)
COOL_LUT = _gain_lut(COOL_GAIN)

# Output channel order matches the input (B, G, R), so each row maps every
# input channel onto one output channel.
SEPIA_MATRIX = np.array(
    [
        [0.131, 0.534, 0.272],  # blue  out
        [0.168, 0.686, 0.349],  # green out
        [0.189, 0.769, 0.393],  # red   out
    ],
    dtype=np.float32,
)

_vignette_cache: dict[tuple[int, int], Frame] = {}


def _masked_multiply(frame: Frame, mask: Frame) -> Frame:
    """frame * (mask / 255) in integer maths, one full-frame pass."""
    return cv2.multiply(frame, mask, scale=1.0 / 255.0)


def _to_bgr(gray: Frame) -> Frame:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _vignette_mask(height: int, width: int) -> Frame:
    """Radial darkening mask as uint8, cached per frame size."""
    key = (height, width)
    mask = _vignette_cache.get(key)
    if mask is None:
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
        radius = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
        # 0.35 keeps the corners around 58% brightness (~1.45x centre/corner),
        # the usual strength for a vintage vignette rather than a heavy fade.
        levels = np.clip((1.0 - 0.35 * radius) * 255.0, 0, 255).astype(np.uint8)
        mask = np.repeat(levels[..., None], 3, axis=2)
        _vignette_cache[key] = mask
    return mask


def _posterize(frame: Frame, levels: int = 8) -> Frame:
    """Quantise each channel to `levels` steps, flattening gradients to bands."""
    step = 256 // levels
    table = np.clip((np.arange(256) // step) * (255 // (levels - 1)), 0, 255).astype(np.uint8)
    return cv2.LUT(frame, table.reshape(1, 256))


def _resize_for_effect(frame: Frame, max_side: int = 480) -> tuple[Frame, tuple[int, int]]:
    """Downscale big frames so expensive filters stay interactive."""
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return frame, (width, height)
    scale = max_side / longest
    small = cv2.resize(
        frame, (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return small, (width, height)


# --------------------------------------------------------------------------
# Filters. Every filter takes and returns an 8-bit BGR frame of equal shape.
# --------------------------------------------------------------------------


def f_normal(frame: Frame) -> Frame:
    return frame


def f_grayscale(frame: Frame) -> Frame:
    return _to_bgr(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))


def f_sepia(frame: Frame) -> Frame:
    out = cv2.transform(frame.astype(np.float32), SEPIA_MATRIX)
    return np.clip(out, 0, 255).astype(np.uint8)


def f_invert(frame: Frame) -> Frame:
    return cv2.bitwise_not(frame)


def f_blur(frame: Frame) -> Frame:
    return cv2.GaussianBlur(frame, (15, 15), 0)


def f_sharpen(frame: Frame) -> Frame:
    blurred = cv2.GaussianBlur(frame, (0, 0), 3)
    return cv2.addWeighted(frame, 1.9, blurred, -0.9, 0)


def f_edges(frame: Frame) -> Frame:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 170)
    return _to_bgr(edges)


def f_cartoon(frame: Frame) -> Frame:
    # Smooth, ink the edges, then posterise last so the colour stays flat
    # (upscaling first would interpolate the bands back into gradients).
    src, size = _resize_for_effect(frame, max_side=640)
    color = cv2.bilateralFilter(src, 9, 120, 120)
    gray = cv2.medianBlur(cv2.cvtColor(src, cv2.COLOR_BGR2GRAY), 5)
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9
    )
    out = cv2.resize(cv2.bitwise_and(color, _to_bgr(edges)), size, interpolation=cv2.INTER_LINEAR)
    return _posterize(out, levels=8)


def f_sketch(frame: Frame) -> Frame:
    gray = cv2.medianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 3)
    inverted = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    dodge = cv2.divide(gray, cv2.bitwise_not(blurred), scale=256.0)
    # Colour dodge maps flat areas to pure white and leaves everything else as
    # faint lines, so most pixels pile up near 255. Stretch the 2nd..98th
    # percentile so the lines become visible. Percentiles are taken from a
    # subsample for speed, and the range has a floor so a flat scene amplifies
    # detail instead of sensor noise.
    sample = dodge[::4, ::4]
    low = float(np.percentile(sample, 2))
    high = max(float(np.percentile(sample, 98)), low + 8.0)
    stretched = (dodge.astype(np.float32) - low) * (255.0 / (high - low))
    return _to_bgr(np.clip(stretched, 0, 255).astype(np.uint8))


def f_warm(frame: Frame) -> Frame:
    return cv2.LUT(frame, WARM_LUT)


def f_cool(frame: Frame) -> Frame:
    return cv2.LUT(frame, COOL_LUT)


def f_vintage(frame: Frame) -> Frame:
    faded = cv2.convertScaleAbs(f_sepia(frame), alpha=0.86, beta=12)
    return _masked_multiply(faded, _vignette_mask(*faded.shape[:2]))


def f_noir(frame: Frame) -> Frame:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    return _to_bgr(cv2.convertScaleAbs(equalized, alpha=1.35, beta=-24))


def f_dream(frame: Frame) -> Frame:
    # The glow is a wide, low-frequency blur: compute it small, then upscale.
    small, size = _resize_for_effect(frame, max_side=320)
    glow = cv2.resize(cv2.GaussianBlur(small, (0, 0), 8), size, interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(frame, 0.55, glow, 0.65, 12)


def f_pixelate(frame: Frame) -> Frame:
    height, width = frame.shape[:2]
    blocks = 14
    small = cv2.resize(
        frame, (max(1, width // blocks), max(1, height // blocks)),
        interpolation=cv2.INTER_AREA,
    )
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)


def f_thermal(frame: Frame) -> Frame:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)


def f_glitch(frame: Frame) -> Frame:
    blue, green, red = cv2.split(frame)
    red = np.roll(red, 9, axis=1)
    blue = np.roll(blue, -9, axis=1)
    out = cv2.merge((blue, green, red))
    out[::4] = (out[::4].astype(np.int16) * 3 // 4).astype(np.uint8)  # scanlines
    return out


@dataclass(frozen=True)
class FilterSpec:
    slug: str
    label: str
    fn: FilterFn


FILTERS: tuple[FilterSpec, ...] = (
    FilterSpec("normal", "Normal", f_normal),
    FilterSpec("grayscale", "Grayscale", f_grayscale),
    FilterSpec("sepia", "Sepia", f_sepia),
    FilterSpec("invert", "Invert", f_invert),
    FilterSpec("blur", "Blur", f_blur),
    FilterSpec("sharpen", "Sharpen", f_sharpen),
    FilterSpec("edges", "Edge", f_edges),
    FilterSpec("cartoon", "Cartoon", f_cartoon),
    FilterSpec("sketch", "Sketch", f_sketch),
    FilterSpec("warm", "Warm", f_warm),
    FilterSpec("cool", "Cool", f_cool),
    FilterSpec("vintage", "Vintage", f_vintage),
    FilterSpec("noir", "Noir", f_noir),
    FilterSpec("dream", "Dream", f_dream),
    FilterSpec("pixelate", "Pixelate", f_pixelate),
    FilterSpec("thermal", "Thermal", f_thermal),
    FilterSpec("glitch", "Glitch", f_glitch),
)

FILTER_INDEX = {spec.slug: i for i, spec in enumerate(FILTERS)}

FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_COLOR = (240, 240, 240)
ACTIVE_COLOR = (120, 255, 120)
SHADOW_COLOR = (0, 0, 0)


def _text(
    image: Frame,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.5,
    color: tuple[int, int, int] = TEXT_COLOR,
    thickness: int = 1,
) -> None:
    cv2.putText(image, text, origin, FONT, scale, SHADOW_COLOR, thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, origin, FONT, scale, color, thickness, cv2.LINE_AA)


def _banner(frame: Frame, top: int, bottom: int, alpha: float = 0.55) -> Frame:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, top), (frame.shape[1], bottom), (0, 0, 0), -1)
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def draw_hud(
    frame: Frame, index: int, fps: float, mirrored: bool, hint: str, face_status: str = ""
) -> Frame:
    """Draw the filter name, FPS, controls hint, face status and filter strip."""
    spec = FILTERS[index]
    height, width = frame.shape[:2]

    out = _banner(frame, 0, 34)
    _text(out, f"[{index + 1}/{len(FILTERS)}] {spec.label}", (12, 24), 0.62, ACTIVE_COLOR, 2)
    _text(out, f"{fps:5.1f} FPS", (width - 110, 24), 0.55)
    _text(out, hint, (12, 54), 0.45, (200, 200, 200))
    if face_status:
        _text(out, face_status, (12, 74), 0.48, (120, 200, 255))

    # Numbered filter strip, wrapped across as many rows as it needs.
    out = _banner(out, height - 62, height)
    x, y = 12, height - 40
    for i, item in enumerate(FILTERS):
        token = f"{i + 1} {item.label}"
        (token_w, _), _ = cv2.getTextSize(token, FONT, 0.45, 1)
        if x + token_w > width - 12:
            x, y = 12, y + 20
        _text(out, token, (x, y), 0.45, ACTIVE_COLOR if i == index else (170, 170, 170))
        x += token_w + 18

    if mirrored:
        _text(out, "MIRROR", (width - 96, height - 40), 0.45, (120, 200, 255))
    return out


def open_camera(camera: int, width: int, height: int) -> cv2.VideoCapture:
    # V4L2 is the reliable backend on Linux; others auto-select their platform default.
    backends = [cv2.CAP_V4L2, cv2.CAP_ANY] if sys.platform.startswith("linux") else [cv2.CAP_ANY]
    for backend in backends:
        capture = cv2.VideoCapture(camera, backend) if backend != cv2.CAP_ANY else cv2.VideoCapture(camera)
        if capture.isOpened():
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            return capture
        capture.release()
    raise RuntimeError(f"could not open camera index {camera}")


HINT = (
    "n/p or arrows: filter   v: face mode   z: face zoom   a: age/expression   "
    "s: snapshot   m: mirror   h: hide   f: fullscreen   q: quit"
)


def load_tracker(args: argparse.Namespace) -> ft.FaceTracker | None:
    """Build the face tracker, or warn and return None when the model is missing."""
    try:
        return ft.FaceTracker(
            args.face_model,
            detect_width=args.face_detect_width,
            detect_every=args.face_detect_every,
        )
    except ft.ModelMissingError as exc:
        print(f"warning: face features are disabled.\n{exc}", file=sys.stderr)
        return None


def load_attributes(args: argparse.Namespace) -> fa.AttributeEstimator:
    """Build the age/expression estimator, reporting whichever model is unavailable."""
    estimator = fa.AttributeEstimator(
        expression_model_path=args.expression_model,
        age_model_path=args.age_model,
        every_n=args.face_attr_every,
    )
    for warning in estimator.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return estimator


def run(args: argparse.Namespace) -> int:
    try:
        capture = open_camera(args.camera, args.width, args.height)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print("hint: check that a webcam is connected and not in use by another app.", file=sys.stderr)
        print("      on WSL/containers, pass through the device or run on the host.", file=sys.stderr)
        return 1

    camera = ThreadedCamera(capture, threaded=not args.no_thread).start()
    index = FILTER_INDEX[args.filter]
    mirrored = not args.no_mirror
    hud_visible = True
    fullscreen = args.fullscreen
    effect_index = ft.FACE_EFFECTS.index(args.face_effect)
    face_zoom = args.face_zoom
    face_labels = args.face_attributes
    tracker: ft.FaceTracker | None = None
    tracker_unavailable = False
    faces: list[ft.Face] = []
    attributes: list[fa.FaceAttributes] = []
    attributes_estimator = load_attributes(args)
    snapshot_dir = os.path.abspath(args.snapshot_dir)
    os.makedirs(snapshot_dir, exist_ok=True)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, args.width, args.height)
    if fullscreen:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print(f"camera {args.camera} opened - {len(FILTERS)} filters available")
    print(f"capture: {'background thread' if camera.threaded else 'inline'}")
    print(f"face: mode={ft.FACE_EFFECTS[effect_index]}  zoom={'on' if face_zoom else 'off'}")
    print(
        f"attributes: labels={'on' if face_labels else 'off'}  "
        f"expression={'on' if attributes_estimator.expression_enabled else 'off'}  "
        f"age={'on' if attributes_estimator.age_enabled else 'off'}"
    )
    print(HINT)
    print(f"snapshots -> {snapshot_dir}")

    fps = 0.0
    previous = time.perf_counter()
    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                print("error: failed to read a frame from the camera", file=sys.stderr)
                return 1

            if mirrored:
                frame = cv2.flip(frame, 1)

            spec = FILTERS[index]
            try:
                frame = spec.fn(frame)
            except cv2.error as exc:  # keep the window alive if one filter trips
                print(f"warning: filter '{spec.slug}' failed: {exc}", file=sys.stderr)
                index = 0
                continue

            effect = ft.FACE_EFFECTS[effect_index]
            if effect != "off" or face_zoom or face_labels:
                if tracker is None and not tracker_unavailable:
                    tracker = load_tracker(args)
                    tracker_unavailable = tracker is None
                if tracker is not None:
                    try:
                        faces = tracker.update(frame)
                    except cv2.error as exc:
                        print(f"warning: face tracking failed ({exc}); disabled", file=sys.stderr)
                        tracker, tracker_unavailable, faces = None, True, []
                attributes = attributes_estimator.update(frame, faces) if faces else []
                # Draw the overlays first, then crop: all then share one coordinate space.
                if faces:
                    frame = ft.apply_face_effect(frame, faces, effect)
                    if face_labels:
                        frame = fa.draw_face_attributes(frame, faces, attributes)
                    if face_zoom:
                        frame = ft.zoom_to_face(frame, faces)
            else:
                faces = []
                attributes = []

            now = time.perf_counter()
            delta = now - previous
            previous = now
            if delta > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / delta)

            face_status = ""
            if effect != "off" or face_zoom or face_labels:
                face_status = f"faces: {len(faces)}   mode: {effect}"
                if face_zoom:
                    face_status += "   zoom: ON"
                if face_labels:
                    summary = attributes[0].label() if attributes else ""
                    face_status += f"   attr: {summary or 'n/a'}"
            display = (
                draw_hud(frame, index, fps, mirrored, HINT, face_status) if hud_visible else frame
            )
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("n"), 83):  # n or right arrow
                index = (index + 1) % len(FILTERS)
            elif key in (ord("p"), 81):  # p or left arrow
                index = (index - 1) % len(FILTERS)
            elif ord("1") <= key <= ord("9"):
                index = key - ord("1")
            elif key == ord("0"):
                index = 9
            elif key == ord("v"):
                effect_index = (effect_index + 1) % len(ft.FACE_EFFECTS)
                print(f"face mode: {ft.FACE_EFFECTS[effect_index]}")
            elif key == ord("z"):
                face_zoom = not face_zoom
                print(f"face zoom: {'on' if face_zoom else 'off'}")
            elif key == ord("a"):
                face_labels = not face_labels
                print(f"age/expression labels: {'on' if face_labels else 'off'}")
            elif key == ord("m"):
                mirrored = not mirrored
            elif key == ord("h"):
                hud_visible = not hud_visible
            elif key == ord("f"):
                fullscreen = not fullscreen
                cv2.setWindowProperty(
                    WINDOW_NAME,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
                )
            elif key == ord("s"):
                name = f"{spec.slug}_{datetime.now():%Y%m%d_%H%M%S}.png"
                path = os.path.join(snapshot_dir, name)
                cv2.imwrite(path, frame)
                print(f"saved {path}")
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print(
            f"camera: {camera.frames_grabbed} frames grabbed, {camera.frames_served} shown, "
            f"{camera.frames_dropped} dropped"
        )
    return 0


# --------------------------------------------------------------------------
# Selftest: exercises every filter, the face code and the camera reader on
# synthetic input, so no camera or real face is required.
# --------------------------------------------------------------------------


def _changed_bbox(before: Frame, after: Frame) -> tuple[int, int, int, int] | None:
    """Bounding box of the pixels that differ, or None when nothing changed."""
    changed = np.any(before != after, axis=2)
    if not changed.any():
        return None
    ys, xs = np.where(changed)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _face_self_test(failures: list[str]) -> None:
    """Exercise the face effects and tracker without a camera or a real face."""
    print("\nface effects (synthetic geometry, no camera)")

    def check(name: str, condition: bool, detail: str = "") -> None:
        print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"  -> {detail}" if detail and not condition else ""))
        if not condition:
            failures.append(f"face: {name} {detail}".strip())

    def within(name: str, before: Frame, after: Frame, bounds: tuple[int, int, int, int]) -> None:
        bbox = _changed_bbox(before, after)
        if bbox is None:
            check(name, False, "nothing changed")
            return
        x0, y0, x1, y1 = bounds
        check(
            name,
            bbox[0] >= x0 and bbox[1] >= y0 and bbox[2] <= x1 and bbox[3] <= y1,
            f"changed {bbox} outside {(x0, y0, x1, y1)}",
        )

    height, width = 240, 320
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    # A textured frame keeps colour/detail changes measurable.
    textured = np.stack([xx % 256, (xx * 2 + yy) % 256, (yy * 3) % 256], -1).astype(np.uint8)
    flat = np.full((height, width, 3), 120, np.uint8)
    face = ft.Face(
        1,
        np.array([100.0, 60.0, 80.0, 90.0], np.float32),
        0.9,
        np.array(
            [[120.0, 90.0], [160.0, 92.0], [140.0, 108.0], [125.0, 124.0], [155.0, 126.0]], np.float32
        ),
    )
    x, y, w, h = ft.clamp_box(face.box, width, height)

    boxed = ft.draw_face_boxes(textured, [face])
    check("draw_face_boxes keeps size and type", boxed.shape == textured.shape and boxed.dtype == np.uint8)
    within("draw_face_boxes paints only around the face", textured, boxed, (x - 3, y - 25, x + w + 95, y + h + 4))
    check(
        "draw_face_boxes marks the landmarks",
        bool((boxed[106:111, 138:143] == np.array(ft.LANDMARK_COLOR, np.uint8)).any()),
    )
    check("draw_face_boxes is a no-op with no faces", np.array_equal(ft.draw_face_boxes(textured, []), textured))

    region = lambda a: float(cv2.Laplacian(cv2.cvtColor(a[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY), cv2.CV_32F).var())
    uniq = lambda a: len(np.unique(a[y : y + h, x : x + w].reshape(-1, 3), axis=0))
    blurred = ft.blur_faces(textured, [face])
    check("blur_faces removes nearly all face detail", region(blurred) < region(textured) * 0.05, f"{region(blurred):.0f} vs {region(textured):.0f}")
    within("blur_faces paints only inside the face", textured, blurred, (x, y, x + w - 1, y + h - 1))
    check("blur_faces is a no-op with no faces", np.array_equal(ft.blur_faces(textured, []), textured))

    pix = ft.pixelate_faces(textured, [face])
    check("pixelate_faces flattens the face colours", uniq(pix) < uniq(textured) * 0.5, f"{uniq(pix)} vs {uniq(textured)}")
    within("pixelate_faces paints only inside the face", textured, pix, (x, y, x + w - 1, y + h - 1))

    glasses = ft.draw_sunglasses(flat, [face])
    dark = lambda a: int((a.astype(int).sum(axis=2) < 150).sum())
    check("draw_sunglasses paints dark lenses", dark(glasses) > 250, str(dark(glasses)))
    within("draw_sunglasses paints only around the eyes", flat, glasses, (x - 20, y - 15, x + w + 20, y + h))
    check("draw_sunglasses is a no-op with no faces", np.array_equal(ft.draw_sunglasses(flat, []), flat))

    tilted = face.copy()
    tilted.landmarks = np.array(
        [[120.0, 80.0], [160.0, 110.0], [140.0, 118.0], [125.0, 132.0], [155.0, 138.0]], np.float32
    )
    tilted_glasses = ft.draw_sunglasses(flat, [tilted])
    ys, xs = np.where(tilted_glasses.astype(int).sum(axis=2) < 150)
    tilt_corr = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 10 else 0.0
    check("draw_sunglasses rotates with the face", tilt_corr > 0.25, f"corr={tilt_corr:.2f}")

    tiny = ft.Face(2, np.array([1.0, 1.0, 4.0, 4.0], np.float32), 0.5, np.array([[2.0, 2.0], [3.0, 2.0], [2.5, 2.5], [2.2, 3.0], [2.8, 3.0]], np.float32))
    check("draw_sunglasses skips faces that are too small", np.array_equal(ft.draw_sunglasses(flat, [tiny]), flat))

    zoom_src = np.full((height, width, 3), 120, np.uint8)
    zoom_src[95:115, 130:150] = 255  # a white square at the face centre
    before = int((cv2.cvtColor(zoom_src, cv2.COLOR_BGR2GRAY) > 200).sum())
    zoomed = ft.zoom_to_face(zoom_src, [face])
    after = int((cv2.cvtColor(zoomed, cv2.COLOR_BGR2GRAY) > 200).sum())
    check("zoom_to_face keeps the frame size", zoomed.shape == zoom_src.shape)
    check("zoom_to_face enlarges the face", after > before * 1.15, f"{after} vs {before}")
    cx0, cy0, cw, ch = ft.face_crop_box(zoom_src.shape, face.box, margin=2.3)
    expected = cv2.resize(zoom_src[cy0 : cy0 + ch, cx0 : cx0 + cw], (width, height), interpolation=cv2.INTER_LINEAR)
    check("zoom_to_face is a crop+resize of the face box", np.array_equal(zoomed, expected))
    check("zoom_to_face is a no-op with no faces", np.array_equal(ft.zoom_to_face(zoom_src, []), zoom_src))

    for shape, box in (
        ((240, 320, 3), np.array([0.0, 0.0, 30.0, 30.0], np.float32)),
        ((240, 320, 3), np.array([290.0, 210.0, 30.0, 30.0], np.float32)),
        ((480, 640, 3), np.array([100.0, 50.0, 300.0, 400.0], np.float32)),
    ):
        bx, by, bw_, bh_ = ft.face_crop_box(shape, box)
        aspect_ok = abs((bw_ / bh_) - (shape[1] / shape[0])) < 0.02
        in_bounds = bx >= 0 and by >= 0 and bx + bw_ <= shape[1] and by + bh_ <= shape[0]
        check(f"face_crop_box stays in bounds and keeps the aspect {box[:2].tolist()}", in_bounds and aspect_ok, f"{(bx, by, bw_, bh_)} aspect={bw_ / bh_:.3f}")

    check("face effect dispatch: 'off' is a no-op", ft.apply_face_effect(textured, [face], "off") is textured)
    for effect in ("box", "blur", "pixelate", "glasses"):
        check(f"face effect dispatch: '{effect}' changes the frame", not np.array_equal(ft.apply_face_effect(textured, [face], effect), textured))

    _face_tracker_self_test(check)


def _face_tracker_self_test(check) -> None:
    """Exercise FaceTracker's matching, smoothing and retirement with injected
detections, so the tracking logic is covered even without a real face."""
    missing = os.path.join(os.path.dirname(ft.default_model_path()), "definitely-not-here.onnx")
    try:
        ft.FaceTracker(missing)
        check("FaceTracker rejects a missing model", False, "no exception raised")
    except ft.ModelMissingError:
        check("FaceTracker raises ModelMissingError for a missing model", True)
    except Exception as exc:  # noqa: BLE001
        check("FaceTracker raises ModelMissingError for a missing model", False, type(exc).__name__)

    model = ft.default_model_path()
    if not os.path.exists(model):
        print(f"  SKIP  tracker matching tests (model not present: {model})")
        return

    box_a = np.array([100.0, 60.0, 80.0, 90.0], np.float32)
    box_b = np.array([110.0, 70.0, 80.0, 90.0], np.float32)
    far = np.array([10.0, 150.0, 40.0, 40.0], np.float32)
    lm_a = np.array([[120.0, 90.0], [160.0, 92.0], [140.0, 108.0], [125.0, 124.0], [155.0, 126.0]], np.float32)
    lm_b = np.array([[130.0, 100.0], [170.0, 102.0], [150.0, 118.0], [135.0, 134.0], [165.0, 136.0]], np.float32)
    lm_far = np.array([[20.0, 160.0], [35.0, 161.0], [27.0, 170.0], [22.0, 178.0], [33.0, 179.0]], np.float32)

    passes = [[(box_a, 0.9, lm_a)], [(box_b, 0.9, lm_b)], [], [(far, 0.8, lm_far)]]
    calls = {"n": 0}

    def fake_detect(_frame):
        index = min(calls["n"], len(passes) - 1)
        calls["n"] += 1
        return passes[index]

    tracker = ft.FaceTracker(model, detect_every=2, smoothing=0.5, max_missed=2)
    tracker._detect = fake_detect
    frame = np.zeros((240, 320, 3), np.uint8)

    first = tracker.update(frame)  # index 0 -> detection pass 0
    check("tracker reports the first face with id 1", len(first) == 1 and first[0].id == 1, str(len(first)))
    check("tracker takes the detected box unmodified", np.allclose(first[0].box, box_a))

    held = tracker.update(frame)  # index 1 -> no detection pass
    check("tracker holds the box between detections", np.allclose(held[0].box, first[0].box))

    smoothed = tracker.update(frame)  # index 2 -> detection pass 1
    check("tracker smooths halfway toward the new box", np.allclose(smoothed[0].box, (box_a + box_b) / 2.0), str(smoothed[0].box))
    check("tracker keeps the id across detections", smoothed[0].id == 1 and len(smoothed) == 1)

    tracker.update(frame)  # index 3 -> no detection pass
    after_miss = tracker.update(frame)  # index 4 -> detection pass 2 (face lost)
    check("tracker keeps a briefly lost face", len(after_miss) == 1 and after_miss[0].missed == 1, str(len(after_miss)))

    for _ in range(8):
        tracker.update(frame)
    check("tracker retires the face that stayed gone", all(t.id != 1 for t in tracker.tracks), str([(t.id, t.missed) for t in tracker.tracks]))
    check("tracker gives a new face elsewhere a new id", any(t.id != 1 and np.allclose(t.box, far) for t in tracker.tracks), str([t.id for t in tracker.tracks]))
    check("tracker throttles detection to every Nth frame", tracker.detect_runs == 7, str(tracker.detect_runs))


def _face_attributes_self_test(failures: list[str]) -> None:
    """Exercise the age/expression code - alignment, decoders, throttle, drawing."""
    print("\nface attributes (synthetic geometry, no camera)")

    def check(name: str, condition: bool, detail: str = "") -> None:
        print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"  -> {detail}" if detail and not condition else ""))
        if not condition:
            failures.append(f"attributes: {name} {detail}".strip())

    # -- decoders ---------------------------------------------------------
    probabilities = fa.softmax(np.array([0.0, 1.0, 2.0], np.float32))
    check("softmax sums to one", abs(float(probabilities.sum()) - 1.0) < 1e-6, str(probabilities.sum()))
    check("softmax is monotone in the logits", bool((np.diff(probabilities) > 0).all()))
    check("softmax survives huge logits", np.isfinite(fa.softmax(np.array([1e4, -1e4], np.float32))).all())

    logits = np.array([[0.0, 0.0, 0.0, 6.0, 1.0, 0.0, 0.0]], np.float32)
    expression, confidence, scores = fa.decode_expression(logits)
    check("decode_expression names the argmax class", expression == "happy", expression)
    check("decode_expression agrees with a raw argmax", int(scores.argmax()) == int(logits.argmax()))
    check("decode_expression reports the peak as confidence", abs(confidence - float(scores.max())) < 1e-6)

    bins = np.full(100, -20.0, np.float32)
    bins[40] = 20.0
    age, age_confidence = fa.decode_age(bins)
    expected = float((fa.softmax(bins) * np.arange(100)).sum())
    check("decode_age turns 100 bins into the expected age", abs(age - expected) < 1e-3, f"{age} vs {expected}")
    check("decode_age reports the peak bin as confidence", 0.5 < age_confidence <= 1.0, str(age_confidence))
    check("decode_age reads a single value as a 0..1 age", fa.decode_age(np.array([[0.42]], np.float32))[0] == 42.0)
    check("decode_age rejects an unexpected shape", fa.decode_age(np.zeros(7, np.float32))[0] is None)

    # -- alignment --------------------------------------------------------
    angle = np.deg2rad(20.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], np.float32
    ) * 1.7
    # Landmarks built from the template by a known similarity must map back onto it.
    landmarks = (fa.ALIGN_TEMPLATE @ rotation.T + np.array([60.0, 40.0], np.float32)).astype(np.float32)
    matrix = fa.similarity_transform(landmarks, fa.ALIGN_TEMPLATE)
    remapped = np.c_[landmarks, np.ones(5, np.float32)] @ matrix.T
    error = float(np.abs(remapped - fa.ALIGN_TEMPLATE).max())
    check("similarity_transform is the inverse of a known similarity", error < 0.5, f"max err {error:.3f}px")

    dotted = np.zeros((240, 320, 3), np.uint8)
    for point in landmarks.astype(int):
        cv2.circle(dotted, (int(point[0]), int(point[1])), 2, (255, 255, 255), -1)
    aligned = fa.align_face(dotted, landmarks)
    check("align_face returns a 112x112 uint8 crop", aligned.shape == (112, 112, 3) and aligned.dtype == np.uint8)
    landed = all(
        aligned[int(point[1]) - 3 : int(point[1]) + 4, int(point[0]) - 3 : int(point[0]) + 4, 0].max() > 128
        for point in fa.ALIGN_TEMPLATE
    )
    check("align_face puts the landmarks on the alignment template", landed)

    centred = np.zeros((240, 320, 3), np.uint8)
    cv2.circle(centred, (140, 105), 3, (255, 255, 255), -1)  # centre of the box below
    age_crop = fa.crop_for_age(centred, np.array([100.0, 60.0, 80.0, 90.0], np.float32))
    check("crop_for_age returns the model's input size", age_crop.shape == (224, 224, 3))
    ys, xs = np.where(age_crop[..., 0] > 128)
    centroid = (float(xs.mean()), float(ys.mean())) if len(xs) else (-1.0, -1.0)
    check(
        "crop_for_age centres the face box on the crop",
        abs(centroid[0] - 112.0) < 3.0 and abs(centroid[1] - 112.0) < 3.0,
        f"centroid {centroid[0]:.1f},{centroid[1]:.1f}",
    )

    # -- estimator --------------------------------------------------------
    placeholder_estimator = fa.AttributeEstimator(expression_model_path=os.path.join("src", "models", "nope.onnx"))
    check("a missing model disables only that attribute", not placeholder_estimator.enabled)
    check("a missing model explains how to fetch it", any("curl" in w for w in placeholder_estimator.warnings))
    frame = np.full((240, 320, 3), 90, np.uint8)
    face = ft.Face(1, np.array([100.0, 60.0, 80.0, 90.0], np.float32), 0.9,
                   (fa.ALIGN_TEMPLATE + np.array([60.0, 40.0], np.float32)).astype(np.float32))
    idle = placeholder_estimator.update(frame, [face])[0]
    check("a disabled estimator renders an empty label", idle.label(show_age=True, show_expression=True) == "")
    check("a disabled estimator still returns one entry per face", len(placeholder_estimator.update(frame, [face])) == 1)
    check("the estimator is a no-op on an empty face list", placeholder_estimator.update(frame, []) == [])

    class FakeAgeNet:
        """Stands in for an age model so the age plumbing is covered without one."""

        def __init__(self, output: np.ndarray) -> None:
            self.output = output
            self.blobs: list[tuple[int, ...]] = []

        def setInput(self, blob: np.ndarray) -> None:
            self.blobs.append(tuple(int(v) for v in blob.shape))

        def forward(self) -> np.ndarray:
            return self.output

    age_estimator = fa.AttributeEstimator(expression_model_path=os.path.join("src", "models", "nope.onnx"))
    fake_age = FakeAgeNet(np.array([[0.34]], np.float32))
    age_estimator._age_net = fake_age
    check("an age model alone enables the estimator", age_estimator.enabled)
    aged = age_estimator.update(frame, [face])[0]
    check("the age path decodes a normalised regression output", aged.age == 34.0, str(aged.age))
    check("the age path feeds a 1x3x224x224 blob", fake_age.blobs == [(1, 3, 224, 224)], str(fake_age.blobs))
    check("the age path renders an age-only label", aged.label(show_expression=False) == "~34y", aged.label())
    check("the age path leaves the expression unknown", aged.expression == "unknown")

    if not os.path.exists(fa.default_expression_model_path()):
        print(f"  SKIP  expression model tests (not present: {fa.default_expression_model_path()})")
    else:
        estimator = fa.AttributeEstimator(every_n=2)
        check("the bundled expression model loads", estimator.expression_enabled)
        first = estimator.update(frame, [face])[0]
        check("the expression model returns a known class", first.expression in fa.EXPRESSIONS, first.expression)
        check("the expression scores sum to one", abs(float(first.expression_scores.sum()) - 1.0) < 1e-5)
        check("the expression model ran once", estimator.expression_runs == 1, str(estimator.expression_runs))
        cached = estimator.update(frame, [face])[0]
        check("the throttled call reuses the cached result", estimator.expression_runs == 1)
        check("the throttled call still reports the label", cached.expression == first.expression)
        estimator.update(frame, [face])
        check("the model runs again on the next refresh tick", estimator.expression_runs == 2, str(estimator.expression_runs))
        tiny_estimator = fa.AttributeEstimator(every_n=1)
        tiny = ft.Face(2, np.array([0.0, 0.0, 4.0, 4.0], np.float32), 0.5, fa.ALIGN_TEMPLATE)
        tiny_attrs = tiny_estimator.update(frame, [tiny])[0]
        check(
            "a tiny face skips the model entirely",
            tiny_attrs.expression == "unknown" and tiny_estimator.expression_runs == 0,
            f"expression={tiny_attrs.expression} runs={tiny_estimator.expression_runs}",
        )
        estimator.update(frame, [])  # a face that left must drop out of the cache
        check("a departed face leaves no cached entry", estimator._cache == {})

    # -- drawing ----------------------------------------------------------
    tagged = fa.FaceAttributes(age=34.0, expression="happy", expression_confidence=0.8)
    check("the label carries age and expression", tagged.label() == "~34y  happy 0.80", tagged.label())
    check("the label can hide either half", tagged.label(show_age=False) == "happy 0.80" and tagged.label(show_expression=False) == "~34y")
    check("an empty label hides everything", fa.FaceAttributes().label() == "")
    check("drawing with no faces is a no-op", np.array_equal(fa.draw_face_attributes(frame, [], []), frame))
    drawn = fa.draw_face_attributes(frame, [face], [tagged])
    check("drawing tags the frame", drawn.shape == frame.shape and drawn.dtype == np.uint8 and not np.array_equal(drawn, frame))
    check("drawing stays inside a 1px-tall frame", fa.draw_face_attributes(np.full((1, 1, 3), 90, np.uint8), [face], [tagged]).shape == (1, 1, 3))


def _camera_self_test(failures: list[str]) -> None:
    """Exercise the threaded camera reader against fake devices."""
    print("\ncamera reader (fake device, no hardware)")
    frame = np.full((48, 64, 3), 120, np.uint8)

    def check(name: str, condition: bool, detail: str = "") -> None:
        print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"  -> {detail}" if detail and not condition else ""))
        if not condition:
            failures.append(f"camera: {name} {detail}".strip())

    class FakeCapture:
        def __init__(self, ok: bool = True) -> None:
            self.ok = ok
            self.reads = 0
            self.released = False

        def read(self):
            self.reads += 1
            return (True, frame.copy()) if self.ok else (False, None)

        def release(self) -> None:
            self.released = True

    capture = FakeCapture()
    camera = ThreadedCamera(capture, failure_limit=3).start()
    ok, got = camera.read(timeout=2.0)
    check("threaded read returns the wrapped frame", ok and got is not None and np.array_equal(got, frame))
    check("threaded camera counted the grab", camera.frames_grabbed >= 1, str(camera.frames_grabbed))
    check("attribute access is forwarded to the capture", camera.released is False and capture.reads >= 1)
    camera.release()
    check("release() releases the wrapped capture", capture.released)

    dead = FakeCapture(ok=False)
    dead_camera = ThreadedCamera(dead, failure_limit=3).start()
    ok2, got2 = dead_camera.read(timeout=0.5)
    check("a dead camera reports failure instead of blocking forever", not ok2 and got2 is None)
    check("the dead camera stopped after the failure limit", dead_camera.last_error_count >= 3, str(dead_camera.last_error_count))
    dead_camera.release()

    inline = FakeCapture()
    inline_camera = ThreadedCamera(inline, threaded=False).start()
    ok3, got3 = inline_camera.read()
    check("inline mode reads synchronously", ok3 and np.array_equal(got3, frame))
    inline_camera.release()
    check("inline mode still releases the capture", inline.released)


def self_test() -> int:
    height, width = 240, 320
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    base = np.stack(
        [
            (xx / width * 255.0),          # blue ramp
            (yy / height * 255.0),         # green ramp
            ((xx + yy) % 256.0),           # red gradient with a wrap
        ],
        axis=-1,
    ).astype(np.uint8)
    base[40:120, 60:200] = (30, 200, 220)  # a bright block to give edges content

    failures: list[str] = []
    print(f"running {len(FILTERS)} filters on a {width}x{height} synthetic frame\n")
    print(f"{'filter':<12} {'ms':>7}  {'mean':>6}  {'max':>4}  changed")
    print("-" * 46)

    for spec in FILTERS:
        started = time.perf_counter()
        try:
            out = spec.fn(base.copy())
        except Exception as exc:  # noqa: BLE001 - report anything the filter raises
            failures.append(f"{spec.slug}: raised {type(exc).__name__}: {exc}")
            print(f"{spec.slug:<12} FAILED: {exc}")
            continue
        elapsed_ms = (time.perf_counter() - started) * 1000

        if not isinstance(out, np.ndarray):
            failures.append(f"{spec.slug}: returned {type(out).__name__}, not an ndarray")
        elif out.dtype != np.uint8:
            failures.append(f"{spec.slug}: dtype {out.dtype}, expected uint8")
        elif out.shape != base.shape:
            failures.append(f"{spec.slug}: shape {out.shape}, expected {base.shape}")
        elif not np.isfinite(out).all():
            failures.append(f"{spec.slug}: produced non-finite values")

        changed = not np.array_equal(out, base)
        print(
            f"{spec.slug:<12} {elapsed_ms:7.1f}  {out.mean():6.1f}  "
            f"{int(out.max()):4d}  {'yes' if changed else 'no'}"
        )

    # Targeted checks so the table above cannot pass with a no-op registry.
    if np.array_equal(f_invert(base), base):
        failures.append("invert: output identical to input")
    gray_out = f_grayscale(base)
    if not np.array_equal(gray_out[..., 0], gray_out[..., 1]) or not np.array_equal(
        gray_out[..., 0], gray_out[..., 2]
    ):
        failures.append("grayscale: output channels differ")
    if np.array_equal(f_pixelate(base), base):
        failures.append("pixelate: output identical to input")
    if not any(not np.array_equal(spec.fn(base.copy()), base) for spec in FILTERS):
        failures.append("no filter changed the input frame")

    print("-" * 46)
    _face_self_test(failures)
    _face_attributes_self_test(failures)
    _camera_self_test(failures)

    if failures:
        print("SELF-TEST FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"\nSELF-TEST PASSED ({len(FILTERS)} filters, face effects, face attributes, camera reader)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Real-time webcam filter app.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--camera", type=int, default=0, help="camera device index")
    parser.add_argument("--width", type=int, default=1280, help="capture width")
    parser.add_argument("--height", type=int, default=720, help="capture height")
    parser.add_argument(
        "--filter",
        dest="filter",
        default=FILTERS[0].slug,
        choices=[spec.slug for spec in FILTERS],
        help="filter to start with",
    )
    parser.add_argument("--snapshot-dir", default="snapshots", help="folder for saved snapshots")
    parser.add_argument("--no-mirror", action="store_true", help="do not mirror the preview")
    parser.add_argument("--fullscreen", action="store_true", help="start in fullscreen")
    parser.add_argument(
        "--face-effect",
        default=ft.FACE_EFFECTS[0],
        choices=list(ft.FACE_EFFECTS),
        help="face overlay to start with (`v` cycles)",
    )
    parser.add_argument("--face-zoom", action="store_true", help="start with face auto-zoom (`z` toggles)")
    parser.add_argument(
        "--face-attributes", action="store_true", help="start with age/expression labels (`a` toggles)"
    )
    parser.add_argument(
        "--face-attr-every", type=int, default=5, help="run the age/expression models every N frames"
    )
    parser.add_argument(
        "--expression-model",
        default=fa.default_expression_model_path(),
        help="path to the ONNX facial-expression model",
    )
    parser.add_argument("--age-model", default=None, help="path to an optional ONNX age model")
    parser.add_argument("--face-model", default=ft.default_model_path(), help="path to the YuNet ONNX face model")
    parser.add_argument(
        "--face-detect-every", type=int, default=3, help="run face detection every N frames"
    )
    parser.add_argument(
        "--face-detect-width", type=int, default=320, help="width face detection runs at"
    )
    parser.add_argument(
        "--no-thread", action="store_true", help="read the camera inline instead of on a background thread"
    )
    parser.add_argument("--list-filters", action="store_true", help="print the filters and exit")
    parser.add_argument("--self-test", action="store_true", help="run filters on a fake frame and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_filters:
        for i, spec in enumerate(FILTERS):
            print(f"{i + 1:2d}. {spec.slug:<10} {spec.label}")
        return 0
    if args.self_test:
        return self_test()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
