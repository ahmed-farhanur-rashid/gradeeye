"""
Border crop + pad-to-square + resize.

Pipeline step 1-2 (plan Section 2):
  1. Border crop: threshold-based detection of circular retina boundary,
     crop tight to bounding box.
  2. Pad to square (not stretch, to preserve circular geometry), then
     resize to TARGET_SIZE x TARGET_SIZE.
"""
import cv2
import numpy as np

TARGET_SIZE = 384


def detect_retina_bbox(img: np.ndarray, thresh_ratio: float = 0.08) -> tuple[int, int, int, int]:
    """
    Threshold-based detection of the circular retina boundary.

    Uses the fact that fundus images have a near-black background outside
    the circular retina disc. Thresholds on a grayscale version, finds the
    largest contour, and returns its bounding box.

    Returns (x, y, w, h). Falls back to full-image bbox if detection fails.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h_img, w_img = gray.shape[:2]

    threshold_value = max(1, int(thresh_ratio * 255))
    _, mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

    # Clean up small noise specks so the contour search finds the retina, not artifacts.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, w_img, h_img

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # Sanity check: reject degenerate boxes (e.g. thresholding caught a thin
    # bright artifact rather than the retina disc).
    if w < 0.1 * w_img or h < 0.1 * h_img:
        return 0, 0, w_img, h_img

    return x, y, w, h


def border_crop(img: np.ndarray, thresh_ratio: float = 0.08) -> np.ndarray:
    """Crop tight to the detected retina bounding box."""
    x, y, w, h = detect_retina_bbox(img, thresh_ratio=thresh_ratio)
    return img[y:y + h, x:x + w]


def pad_to_square(img: np.ndarray) -> np.ndarray:
    """
    Pad (not stretch) to a square canvas, centered, preserving circular
    geometry of the retina disc.
    """
    h, w = img.shape[:2]
    size = max(h, w)

    if img.ndim == 3:
        canvas = np.zeros((size, size, img.shape[2]), dtype=img.dtype)
    else:
        canvas = np.zeros((size, size), dtype=img.dtype)

    y_off = (size - h) // 2
    x_off = (size - w) // 2
    canvas[y_off:y_off + h, x_off:x_off + w] = img
    return canvas


def crop_pad_resize(img: np.ndarray, target_size: int = TARGET_SIZE) -> np.ndarray:
    """Full step 1-2 pipeline: border crop -> pad to square -> resize."""
    cropped = border_crop(img)
    squared = pad_to_square(cropped)
    resized = cv2.resize(squared, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return resized
