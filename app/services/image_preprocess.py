"""
Image preprocessing helpers applied before pages are sent to Gemini.

The goal is to nudge OCR/vision quality up for low-grade scans without
introducing heavy dependencies. Everything degrades gracefully: if the
optional libraries (numpy, OpenCV) are missing, the original image is
returned untouched and the pipeline continues exactly as before.

Three transforms are exposed via a single ``preprocess_page`` entry point:

1. **Auto-rotation** — uses Pillow's EXIF transpose to honour camera
   metadata, then optionally rotates by 90/180/270° if the page is
   detected as sideways.
2. **Deskew** — uses OpenCV's minAreaRect on a binarised copy to find
   the dominant text angle and rotate within ±15°.
3. **Contrast normalisation** — uses Pillow's ``ImageOps.autocontrast``
   to stretch the histogram. Cheap and reversible.

All transforms are individually toggleable via env vars (see
``config.settings``) so they can be disabled if they ever introduce
regressions on a particular customer corpus.
"""

from __future__ import annotations

import logging
from typing import Optional

from PIL import Image, ImageOps

logger = logging.getLogger("image_preprocess")

try:  # pragma: no cover - optional dependency probe
    import numpy as np  # type: ignore
    import cv2  # type: ignore

    _CV_AVAILABLE = True
except Exception:  # pragma: no cover
    _CV_AVAILABLE = False


def _autocontrast(image: Image.Image) -> Image.Image:
    try:
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        return ImageOps.autocontrast(image, cutoff=1)
    except Exception as exc:  # pragma: no cover
        logger.debug(f"autocontrast skipped: {exc}")
        return image


def _exif_orient(image: Image.Image) -> Image.Image:
    try:
        return ImageOps.exif_transpose(image)
    except Exception:  # pragma: no cover
        return image


def _detect_skew_angle(image: Image.Image) -> float:
    """Return a skew angle in degrees within (-15, 15), or 0.0 if undetectable."""
    if not _CV_AVAILABLE:
        return 0.0
    try:
        arr = np.array(image.convert("L"))
        # Invert + threshold so text is white on black for findContours.
        _, bw = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        coords = np.column_stack(np.where(bw > 0))
        if coords.size == 0:
            return 0.0
        angle = cv2.minAreaRect(coords)[-1]
        # OpenCV returns angle in [-90, 0). Normalise to (-45, 45).
        if angle < -45:
            angle = 90 + angle
        if abs(angle) > 15:
            return 0.0
        return float(angle)
    except Exception as exc:  # pragma: no cover
        logger.debug(f"skew detection failed: {exc}")
        return 0.0


def _rotate(image: Image.Image, angle: float) -> Image.Image:
    if abs(angle) < 0.25:
        return image
    try:
        return image.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))
    except Exception:  # pragma: no cover
        return image


def preprocess_page(
    image: Image.Image,
    *,
    auto_rotate: bool = True,
    deskew: bool = True,
    contrast: bool = True,
) -> Image.Image:
    """Apply enabled preprocessing steps and return the resulting image."""
    if image is None:
        return image
    out = image
    if auto_rotate:
        out = _exif_orient(out)
    if deskew:
        angle = _detect_skew_angle(out)
        if angle:
            logger.debug(f"deskew applied: {angle:+.2f}°")
            out = _rotate(out, -angle)
    if contrast:
        out = _autocontrast(out)
    return out


def preprocess_pages(images, *, auto_rotate: bool = True, deskew: bool = True, contrast: bool = True):
    """Apply ``preprocess_page`` to a sequence of images, in place where possible."""
    if not images:
        return images
    out = []
    for idx, img in enumerate(images):
        try:
            out.append(preprocess_page(img, auto_rotate=auto_rotate, deskew=deskew, contrast=contrast))
        except Exception as exc:  # pragma: no cover - never let preprocessing kill the run
            logger.warning(f"preprocess failed on page {idx}: {exc}")
            out.append(img)
    return out


__all__ = ["preprocess_page", "preprocess_pages"]
