"""Age and facial-expression estimation for tracked faces.

Expression recognition uses OpenCV Zoo's *Progressive Teacher* MobileFaceNet
model (``facial_expression_recognition_mobilefacenet_2022july.onnx``), which
classifies the seven basic expressions (angry, disgust, fearful, happy,
neutral, sad, surprised) at 88.27% on RAF-DB. The model expects the face
cropped and rotated onto the standard 5-point template its alignment was
trained with, so :func:`align_face` reuses the very landmarks YuNet already
returns (see :mod:`src.face_tracking`) to build that crop.

Age uses a second, *optional* ONNX model. No age model is bundled: several
public ones were measured and every one of them turned out to be degenerate
(see the "Model umur" section of README.md). When no age model is supplied the
estimator simply reports no age, and the rest of the app keeps working.

Nothing here touches the camera, so every function can be exercised on a plain
BGR frame and synthetic landmarks (see ``--self-test``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np

from .face_tracking import Face, clamp_box

Frame = np.ndarray

EXPRESSION_MODEL_FILENAME = "facial_expression_recognition_mobilefacenet_2022july.onnx"
EXPRESSION_MODEL_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    f"models/facial_expression_recognition/{EXPRESSION_MODEL_FILENAME}"
)

# The expression model's own class order (opencv_zoo `FacialExpressionRecog.getDesc`).
EXPRESSIONS: tuple[str, ...] = (
    "angry",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
)

# 5-point template the expression model was aligned with (opencv_zoo FaceAlignment).
ALIGN_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],  # right eye
        [73.5318, 51.5014],  # left eye
        [56.0252, 71.7366],  # nose tip
        [41.5493, 92.3655],  # right mouth corner
        [70.7299, 92.2041],  # left mouth corner
    ],
    dtype=np.float32,
)
ALIGN_SIZE = (112, 112)

# Age models are fed a square crop around the face; this is how much of the
# frame the face is expected to fill (InsightFace-style framing).
AGE_INPUT_SIZE = (224, 224)
AGE_CROP_SCALE = 1.5
AGE_BIN_COUNT = 100  # the classic 0..99 age-class head

LABEL_COLOR = (255, 214, 120)
LABEL_TEXT_COLOR = (25, 25, 25)


def default_expression_model_path() -> str:
    """Path of the bundled expression model."""
    package_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(package_dir, "models", EXPRESSION_MODEL_FILENAME)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    values = np.asarray(logits, dtype=np.float64).ravel()
    values = values - values.max()
    exponentials = np.exp(values)
    total = float(exponentials.sum())
    return (exponentials / total if total > 0 else np.full_like(exponentials, 1.0 / len(values))).astype(
        np.float32
    )


def decode_expression(output: np.ndarray) -> tuple[str, float, np.ndarray]:
    """Turn a raw model output into ``(expression, confidence, probabilities)``.

    The model returns logits (they are not normalised and can be negative), so
    they are softmaxed here.
    """
    probabilities = softmax(output)
    index = int(probabilities.argmax())
    return EXPRESSIONS[index], float(probabilities[index]), probabilities


def decode_age(output: np.ndarray) -> tuple[float | None, float]:
    """Turn a raw model output into ``(age_in_years, confidence)``.

    Two output shapes are accepted, which covers the two ways age models are
    normally exported:

    * ``(..., 100)`` - a softmax over the 0..99 age classes; the prediction is
      the expectation of that distribution and the confidence is its peak.
    * ``(...)`` with a single value - a normalised age in ``[0, 1]``, read as
      ``value * 100``; the confidence is always 1.0, since a regressor exposes
      no distribution to measure.
    """
    values = np.asarray(output, dtype=np.float32).ravel()
    if values.size == 0 or not np.isfinite(values).all():
        return None, 0.0
    if values.size == AGE_BIN_COUNT:
        probabilities = softmax(values)
        ages = np.arange(AGE_BIN_COUNT, dtype=np.float32)
        return float((probabilities * ages).sum()), float(probabilities.max())
    if values.size == 1:
        return float(np.clip(values[0], 0.0, 1.0) * 100.0), 1.0
    return None, 0.0


def similarity_transform(source: np.ndarray, target: np.ndarray) -> Frame:
    """Least-squares similarity (scale + rotation + translation) mapping source onto target.

    Same estimator as ``opencv_zoo``'s ``FaceAlignment.__findNonreflectiveSimilarity``,
    but solved directly with :func:`numpy.linalg.lstsq` instead of a RANSAC
    wrapper, so the result is deterministic.
    """
    src = np.asarray(source, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(target, dtype=np.float64).reshape(-1, 2)
    count = dst.shape[0]
    x, y = dst[:, 0].reshape(-1, 1), dst[:, 1].reshape(-1, 1)
    design = np.vstack(
        (
            np.hstack((x, y, np.ones((count, 1)), np.zeros((count, 1)))),
            np.hstack((y, -x, np.zeros((count, 1)), np.ones((count, 1)))),
        )
    )
    wanted = np.vstack((src[:, 0].reshape(-1, 1), src[:, 1].reshape(-1, 1)))
    scale, shear, tx, ty = np.squeeze(np.linalg.lstsq(design, wanted, rcond=-1)[0])
    inverse = np.array([[scale, -shear, 0.0], [shear, scale, 0.0], [tx, ty, 1.0]], dtype=np.float64)
    transform = np.linalg.inv(inverse)
    transform[:, 2] = (0.0, 0.0, 1.0)
    return transform[:, 0:2].T.astype(np.float32)


def align_face(frame: Frame, landmarks: np.ndarray, size: tuple[int, int] = ALIGN_SIZE) -> Frame:
    """Crop and rotate ``frame`` so ``landmarks`` land on the standard template."""
    matrix = similarity_transform(np.asarray(landmarks, dtype=np.float32).reshape(-1, 2), ALIGN_TEMPLATE)
    return cv2.warpAffine(frame, matrix, size, flags=cv2.INTER_LINEAR)


def crop_for_age(frame: Frame, box: np.ndarray, size: tuple[int, int] = AGE_INPUT_SIZE) -> Frame:
    """Square crop centred on ``box``, with the face filling about 1/``AGE_CROP_SCALE``."""
    height, width = frame.shape[:2]
    x, y, w, h = (float(v) for v in box[:4])
    center_x, center_y = x + w / 2.0, y + h / 2.0
    side = max(w, h) * AGE_CROP_SCALE
    scale = size[0] / max(side, 1.0)
    matrix = np.array(
        [[scale, 0.0, size[0] / 2.0 - center_x * scale], [0.0, scale, size[1] / 2.0 - center_y * scale]],
        dtype=np.float32,
    )
    return cv2.warpAffine(frame, matrix, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _expression_blob(aligned: Frame) -> Frame:
    """Match opencv_zoo's preprocessing: RGB, ``(value - 127.5) / 127.5``, NCHW."""
    rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32)
    return cv2.dnn.blobFromImage(rgb, 1.0 / 127.5, ALIGN_SIZE, (127.5, 127.5, 127.5))


def _age_blob(crop: Frame) -> Frame:
    """RGB, ``(value - 127.5) / 128``, NCHW - the framing age heads are trained on."""
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
    return cv2.dnn.blobFromImage(rgb, 1.0 / 128.0, AGE_INPUT_SIZE, (127.5, 127.5, 127.5))


@dataclass
class FaceAttributes:
    """What was estimated for one face; ``None``/``unknown`` means "not available"."""

    expression: str = "unknown"
    expression_confidence: float = 0.0
    expression_scores: np.ndarray | None = None
    age: float | None = None
    age_confidence: float = 0.0

    def label(self, show_age: bool = True, show_expression: bool = True) -> str:
        """Compact one-line rendering for the on-screen tag."""
        parts: list[str] = []
        if show_age and self.age is not None:
            parts.append(f"~{int(round(self.age))}y")
        if show_expression and self.expression != "unknown":
            parts.append(f"{self.expression} {self.expression_confidence:.2f}")
        return "  ".join(parts)


class AttributeEstimator:
    """Estimate age and expression for tracked faces.

    Detection models are far too slow to run on every frame, so like
    :class:`src.face_tracking.FaceTracker` this throttles: the models run once
    every ``every_n`` calls and the last result per face id is reused in
    between. Faces that disappear have their result dropped.

    A missing model only disables that one attribute - the other keeps working.
    """

    def __init__(
        self,
        expression_model_path: str | None = None,
        age_model_path: str | None = None,
        every_n: int = 5,
        min_face_px: int = 48,
    ) -> None:
        self.every_n = max(1, int(every_n))
        self.min_face_px = max(8, int(min_face_px))
        self.warnings: list[str] = []
        self._expression_net: cv2.dnn.Net | None = None
        self._age_net: cv2.dnn.Net | None = None
        self._cache: dict[int, FaceAttributes] = {}
        self._frame_index = 0
        self.expression_runs = 0
        self.age_runs = 0

        expression_path = expression_model_path or default_expression_model_path()
        if os.path.exists(expression_path):
            try:
                self._expression_net = cv2.dnn.readNetFromONNX(expression_path)
            except cv2.error as exc:
                self.warnings.append(f"expression model failed to load ({exc}); expression disabled")
        else:
            self.warnings.append(
                f"expression model not found at {expression_path}\n"
                f"download it with:\n  curl -L -o {expression_path} \\\n    {EXPRESSION_MODEL_URL}"
            )

        if age_model_path:
            if os.path.exists(age_model_path):
                try:
                    self._age_net = cv2.dnn.readNetFromONNX(age_model_path)
                except cv2.error as exc:
                    self.warnings.append(f"age model failed to load ({exc}); age disabled")
            else:
                self.warnings.append(f"age model not found at {age_model_path}; age disabled")

    @property
    def expression_enabled(self) -> bool:
        return self._expression_net is not None

    @property
    def age_enabled(self) -> bool:
        return self._age_net is not None

    @property
    def enabled(self) -> bool:
        return self.expression_enabled or self.age_enabled

    def update(self, frame: Frame, faces: list[Face]) -> list[FaceAttributes]:
        """Return attributes for ``faces``, refreshing the models every ``every_n`` calls."""
        active = {face.id for face in faces}
        self._cache = {face_id: attrs for face_id, attrs in self._cache.items() if face_id in active}
        if not faces:
            self._frame_index += 1
            return []

        refresh = self._frame_index % self.every_n == 0
        self._frame_index += 1
        if refresh and self.enabled:
            for face in faces:
                self._cache[face.id] = self._estimate(frame, face)
        return [self._cache.get(face.id, FaceAttributes()) for face in faces]

    def _estimate(self, frame: Frame, face: Face) -> FaceAttributes:
        x, y, w, h = clamp_box(face.box, frame.shape[1], frame.shape[0])
        if max(w, h) < self.min_face_px:
            return FaceAttributes()
        crop = frame[y : y + h, x : x + w]
        if crop.size == 0:
            return FaceAttributes()

        attributes = FaceAttributes()
        if self._expression_net is not None and face.landmarks.shape[0] >= 5:
            aligned = align_face(frame, face.landmarks)
            self._expression_net.setInput(_expression_blob(aligned), "data")
            logits = np.asarray(self._expression_net.forward("label"))
            expression, confidence, scores = decode_expression(logits)
            self.expression_runs += 1
            attributes.expression = expression
            attributes.expression_confidence = confidence
            attributes.expression_scores = scores
        if self._age_net is not None:
            self._age_net.setInput(_age_blob(crop_for_age(frame, face.box)))
            age, confidence = decode_age(np.asarray(self._age_net.forward()))
            self.age_runs += 1
            attributes.age = age
            attributes.age_confidence = confidence
        return attributes


def draw_face_attributes(
    frame: Frame,
    faces: list[Face],
    attributes: list[FaceAttributes],
    show_age: bool = True,
    show_expression: bool = True,
) -> Frame:
    """Tag every face with its age and expression, above (or below) its box."""
    if not faces:
        return frame
    out = frame.copy()
    height, width = out.shape[:2]
    for face, attrs in zip(faces, attributes):
        label = attrs.label(show_age=show_age, show_expression=show_expression)
        if not label:
            continue
        x, y, w, h = clamp_box(face.box, width, height)
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # Sit the tag under the box, moving above it when the box starts at the top.
        tag_y = y + h + text_h + 8
        if tag_y + 6 > height:
            tag_y = max(0, y - 6)
        cv2.rectangle(out, (x, tag_y - text_h - 6), (x + text_w + 10, tag_y + 4), LABEL_COLOR, -1)
        cv2.putText(
            out,
            label,
            (x + 5, tag_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            LABEL_TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )
    return out
