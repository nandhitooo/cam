"""Face detection, tracking and face effects.

Detection uses OpenCV's YuNet DNN detector (``cv2.FaceDetectorYN``), which returns
a bounding box plus five landmarks per face. A detection pass costs ~20 ms at a
320 px input, so :class:`FaceTracker` runs the detector only every ``detect_every``
frames and carries boxes forward in between, smoothing them and keeping stable
ids so the overlays do not flicker.

Nothing here touches the camera, so every function can be exercised on a plain
BGR frame (see ``--self-test``).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import cv2
import numpy as np

Frame = np.ndarray

MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"
MODEL_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    f"models/face_detection_yunet/{MODEL_FILENAME}"
)

# Effect names, in the order the `v` key cycles through them.
FACE_EFFECTS: tuple[str, ...] = ("off", "box", "blur", "pixelate", "glasses")

BOX_COLOR = (80, 220, 255)
LANDMARK_COLOR = (120, 255, 120)
GLASSES_COLOR = (18, 18, 18)


def default_model_path() -> str:
    """Path of the bundled YuNet model."""
    package_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(package_dir, "models", MODEL_FILENAME)


class ModelMissingError(FileNotFoundError):
    """Raised when the face model is not on disk."""

    def __init__(self, path: str) -> None:
        super().__init__(
            f"face model not found at {path}\n"
            f"download it with:\n  curl -L -o {path} \\\n    {MODEL_URL}"
        )
        self.path = path


@dataclass
class Face:
    """A tracked face, all coordinates in full-frame pixels."""

    id: int
    box: np.ndarray  # x, y, w, h
    score: float
    landmarks: np.ndarray  # (5, 2): right eye, left eye, nose, right mouth, left mouth
    missed: int = 0

    def copy(self) -> "Face":
        return Face(self.id, self.box.copy(), self.score, self.landmarks.copy(), self.missed)

    @property
    def center(self) -> tuple[float, float]:
        return float(self.box[0] + self.box[2] / 2.0), float(self.box[1] + self.box[3] / 2.0)

    @property
    def area(self) -> float:
        return float(self.box[2] * self.box[3])

    @property
    def eye_span(self) -> float:
        if self.landmarks.shape[0] < 2:
            return 0.0
        return float(np.hypot(*(self.landmarks[1] - self.landmarks[0])))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two x, y, w, h boxes."""
    ax0, ay0, aw, ah = (float(v) for v in a)
    bx0, by0, bw, bh = (float(v) for v in b)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax0 + aw, bx0 + bw), min(ay0 + ah, by0 + bh)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class FaceTracker:
    """Runs YuNet every N frames and tracks faces in between.

    Detection is the expensive part, so it is throttled to every ``detect_every``
    frames while the tracks themselves are updated on every frame: matched faces
    are smoothed towards the new detection, unconfirmed tracks are kept for
    ``max_missed`` detection passes (a face that turns away briefly keeps its id),
    and brand new boxes get a fresh id.
    """

    def __init__(
        self,
        model_path: str | None = None,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 50,
        detect_width: int = 320,
        detect_every: int = 3,
        smoothing: float = 0.5,
        match_iou: float = 0.2,
        max_missed: int = 8,
        max_faces: int = 10,
    ) -> None:
        path = model_path or default_model_path()
        if not os.path.exists(path):
            raise ModelMissingError(path)
        self.model_path = path
        self.detect_width = max(32, int(detect_width))
        self.detect_every = max(1, int(detect_every))
        self.smoothing = float(min(max(smoothing, 0.0), 1.0))
        self.match_iou = float(match_iou)
        self.max_missed = int(max_missed)
        self.max_faces = int(max_faces)
        self._detector = cv2.FaceDetectorYN.create(
            path, "", (self.detect_width, self.detect_width),
            float(score_threshold), float(nms_threshold), int(top_k),
        )
        self._input_size: tuple[int, int] | None = None
        self._tracks: list[Face] = []
        self._next_id = 1
        self._frame_index = 0
        self.detect_runs = 0
        self.last_detections = 0

    @property
    def tracks(self) -> list[Face]:
        """Current tracks (copies, safe for the caller to keep or mutate)."""
        return [t.copy() for t in self._tracks]

    def update(self, frame: Frame) -> list[Face]:
        """Advance the tracker by one frame and return the current faces."""
        if self._frame_index % self.detect_every == 0:
            detections = self._detect(frame)
            self.detect_runs += 1
            self.last_detections = len(detections)
            self._assimilate(detections)
        self._frame_index += 1
        return self.tracks

    # -- internals ---------------------------------------------------------

    def _detect(self, frame: Frame) -> list[tuple[np.ndarray, float, np.ndarray]]:
        """Detect faces, returning full-resolution boxes, scores and landmarks."""
        height, width = frame.shape[:2]
        if height < 8 or width < 8:
            return []
        scale = min(1.0, self.detect_width / float(max(height, width)))
        if scale < 1.0:
            # YuNet wants a size it can stride through; multiples of 32 are safe.
            small_w = max(32, int(round(width * scale / 32.0)) * 32)
            small_h = max(32, int(round(height * scale / 32.0)) * 32)
            small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)
        else:
            small_w, small_h = width, height
            small = frame
        if self._input_size != (small_w, small_h):
            self._detector.setInputSize((small_w, small_h))
            self._input_size = (small_w, small_h)

        _, raw = self._detector.detect(small)
        if raw is None or len(raw) == 0:
            return []

        sx = width / float(small_w)
        sy = height / float(small_h)
        out: list[tuple[np.ndarray, float, np.ndarray]] = []
        for row in raw:
            values = np.asarray(row, dtype=np.float32).ravel()
            box = np.array(
                [values[0] * sx, values[1] * sy, values[2] * sx, values[3] * sy], dtype=np.float32
            )
            landmarks = values[4:14].reshape(5, 2) * np.array([sx, sy], dtype=np.float32)
            out.append((box, float(values[-1]), landmarks))
        return out

    def _assimilate(self, detections: list[tuple[np.ndarray, float, np.ndarray]]) -> None:
        """Match detections onto existing tracks, then add/retire tracks."""
        pairs: dict[int, int] = {}
        taken: set[int] = set()
        for det_index, (box, _score, _lm) in enumerate(detections):
            best: Face | None = None
            best_iou = self.match_iou
            for track in self._tracks:
                if track.id in taken:
                    continue
                iou = _iou(track.box, box)
                if iou > best_iou:
                    best, best_iou = track, iou
            if best is not None:
                pairs[best.id] = det_index
                taken.add(best.id)

        alpha = self.smoothing
        kept: list[Face] = []
        for track in self._tracks:
            if track.id in pairs:
                box, score, landmarks = detections[pairs[track.id]]
                track.box = (1.0 - alpha) * track.box + alpha * box
                track.landmarks = (1.0 - alpha) * track.landmarks + alpha * landmarks
                track.score = float(score)
                track.missed = 0
                kept.append(track)
            else:
                track.missed += 1
                if track.missed <= self.max_missed:
                    kept.append(track)

        for det_index in sorted(set(range(len(detections))) - set(pairs.values())):
            if len(kept) >= self.max_faces:
                break
            box, score, landmarks = detections[det_index]
            kept.append(Face(self._next_id, box.copy(), float(score), landmarks.copy()))
            self._next_id += 1

        self._tracks = kept


# --------------------------------------------------------------------------
# Face effects. Each takes and returns an 8-bit BGR frame.
# --------------------------------------------------------------------------


def clamp_box(box: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    """Clip an x, y, w, h box to the frame, returning integer pixel bounds."""
    x = int(round(float(box[0])))
    y = int(round(float(box[1])))
    w = int(round(float(box[2])))
    h = int(round(float(box[3])))
    x0 = min(max(x, 0), max(width - 1, 0))
    y0 = min(max(y, 0), max(height - 1, 0))
    x1 = min(max(x + w, x0 + 1), width)
    y1 = min(max(y + h, y0 + 1), height)
    return x0, y0, x1 - x0, y1 - y0


def blur_faces(frame: Frame, faces: list[Face]) -> Frame:
    """Blur every detected face, leaving the rest of the frame sharp.

    The blur happens on a heavily downscaled copy of the face and is scaled back
    up. That is a genuinely strong blur (a large face collapses to roughly 30 px,
    which no amount of upscaling can restore) while costing a fraction of a large
    Gaussian kernel applied at full resolution.
    """
    if not faces:
        return frame
    out = frame.copy()
    height, width = out.shape[:2]
    for face in faces:
        x, y, w, h = clamp_box(face.box, width, height)
        factor = min(16, max(3, max(w, h) // 32))
        small = cv2.resize(
            out[y : y + h, x : x + w], (max(1, w // factor), max(1, h // factor)),
            interpolation=cv2.INTER_AREA,
        )
        small = cv2.GaussianBlur(small, (5, 5), 0)
        out[y : y + h, x : x + w] = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return out


def pixelate_faces(frame: Frame, faces: list[Face]) -> Frame:
    """Pixelate every detected face, leaving the rest of the frame sharp."""
    if not faces:
        return frame
    out = frame.copy()
    height, width = out.shape[:2]
    for face in faces:
        x, y, w, h = clamp_box(face.box, width, height)
        block = max(4, h // 12)
        small = cv2.resize(
            out[y : y + h, x : x + w],
            (max(1, w // block), max(1, h // block)),
            interpolation=cv2.INTER_AREA,
        )
        out[y : y + h, x : x + w] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return out


def draw_face_boxes(frame: Frame, faces: list[Face], show_landmarks: bool = True) -> Frame:
    """Outline each face with its id and confidence, plus landmark dots."""
    if not faces:
        return frame
    out = frame.copy()
    height, width = out.shape[:2]
    for face in faces:
        x, y, w, h = clamp_box(face.box, width, height)
        cv2.rectangle(out, (x, y), (x + w, y + h), BOX_COLOR, 2, cv2.LINE_AA)

        label = f"face {face.id}  {face.score:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        label_y = y - text_h - 6 if y - text_h - 6 > 0 else y + h + 2
        cv2.rectangle(
            out, (x, label_y), (x + text_w + 8, label_y + text_h + 6), BOX_COLOR, -1
        )
        cv2.putText(
            out, label, (x + 4, label_y + text_h + 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA,
        )

        if show_landmarks:
            for point in face.landmarks:
                cv2.circle(out, (int(round(point[0])), int(round(point[1]))), 2, LANDMARK_COLOR, -1, cv2.LINE_AA)
    return out


def draw_sunglasses(frame: Frame, faces: list[Face], color: tuple[int, int, int] = GLASSES_COLOR) -> Frame:
    """Draw sunglasses aligned to each face via its eye landmarks.

    The lens geometry is built in a local frame whose x axis runs eye to eye, so
    a tilted or turned head rotates the glasses with it.
    """
    if not faces:
        return frame
    out = frame.copy()
    for face in faces:
        if face.landmarks.shape[0] < 2:
            continue
        right_eye = face.landmarks[0].astype(float)
        left_eye = face.landmarks[1].astype(float)
        span = float(np.hypot(*(left_eye - right_eye)))
        if span < 10.0:  # too small to draw anything legible
            continue
        angle = math.atan2(left_eye[1] - right_eye[1], left_eye[0] - right_eye[0])
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        mid_x = (right_eye[0] + left_eye[0]) / 2.0
        mid_y = (right_eye[1] + left_eye[1]) / 2.0

        def to_image(px: float, py: float) -> tuple[int, int]:
            return (
                int(round(mid_x + px * cos_a - py * sin_a)),
                int(round(mid_y + px * sin_a + py * cos_a)),
            )

        def fill(points: list[tuple[int, int]]) -> None:
            cv2.fillConvexPoly(out, np.array(points, dtype=np.int32), color, cv2.LINE_AA)

        half_w = 0.33 * span  # half lens width
        half_h = 0.25 * span  # half lens height
        lens_y = -0.05 * span  # sit the lenses slightly high, like real glasses
        for eye_x in (-span / 2.0, span / 2.0):
            fill([
                to_image(eye_x - half_w, lens_y - half_h),
                to_image(eye_x + half_w, lens_y - half_h),
                to_image(eye_x + half_w, lens_y + half_h),
                to_image(eye_x - half_w, lens_y + half_h),
            ])
        # bridge over the nose
        fill([
            to_image(-0.16 * span, lens_y - 0.10 * span),
            to_image(0.16 * span, lens_y - 0.10 * span),
            to_image(0.16 * span, lens_y + 0.04 * span),
            to_image(-0.16 * span, lens_y + 0.04 * span),
        ])
        # arms out towards the ears
        for sign in (-1, 1):
            outer = sign * (span / 2.0 + half_w)
            fill([
                to_image(outer, lens_y - 0.20 * span),
                to_image(sign * 1.15 * span, lens_y - 0.16 * span),
                to_image(sign * 1.15 * span, lens_y - 0.02 * span),
                to_image(outer, lens_y + 0.02 * span),
            ])
    return out


def face_crop_box(
    shape: tuple[int, ...], box: np.ndarray, margin: float = 2.3, min_height: float = 0.25
) -> tuple[int, int, int, int]:
    """Crop rectangle that frames ``box`` with margin, matching the frame aspect."""
    height, width = shape[:2]
    crop_h = min(max(float(box[3]) * margin, min_height * height), float(height))
    crop_w = crop_h * width / float(height)
    if crop_w > width:  # very wide frame: cap by width and keep the aspect intact
        crop_w = float(width)
        crop_h = crop_w * height / float(width)
    cx = float(box[0]) + float(box[2]) / 2.0
    cy = float(box[1]) + float(box[3]) / 2.0
    x0 = int(round(min(max(cx - crop_w / 2.0, 0.0), width - crop_w)))
    y0 = int(round(min(max(cy - crop_h / 2.0, 0.0), height - crop_h)))
    x0 = min(max(x0, 0), width - 1)
    y0 = min(max(y0, 0), height - 1)
    w = min(int(round(crop_w)), width - x0)
    h = min(int(round(crop_h)), height - y0)
    return x0, y0, max(1, w), max(1, h)


def zoom_to_face(frame: Frame, faces: list[Face], margin: float = 2.3) -> Frame:
    """Crop and enlarge around the largest face, like a portrait/face-cam view."""
    if not faces:
        return frame
    face = max(faces, key=lambda f: f.area)
    height, width = frame.shape[:2]
    x, y, w, h = face_crop_box(frame.shape, face.box, margin=margin)
    crop = frame[y : y + h, x : x + w]
    if crop.size == 0 or (w == width and h == height):
        return frame
    return cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)


def apply_face_effect(frame: Frame, faces: list[Face], effect: str) -> Frame:
    """Apply a named face effect; unknown or ``off`` returns the frame unchanged."""
    if effect == "box":
        return draw_face_boxes(frame, faces)
    if effect == "blur":
        return blur_faces(frame, faces)
    if effect == "pixelate":
        return pixelate_faces(frame, faces)
    if effect == "glasses":
        return draw_sunglasses(frame, faces)
    return frame
