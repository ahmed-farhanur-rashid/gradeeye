"""
Ben Graham local-average color subtraction, CLAHE, and circular masking.

Pipeline steps 3-5 (plan Section 2):
  3. Ben Graham local-average color subtraction: subtract Gaussian-blurred
     version of image from itself, rescale. Corrects uneven illumination
     across cameras/clinics.
  4. CLAHE: applied on green channel (primary). All-channel CLAHE is an
     optional cheap ablation, not required.
  5. Circular mask: zero out residual corner artifacts outside retina disc
     post-crop.
"""
import cv2
import numpy as np


def ben_graham_subtraction(img: np.ndarray, sigma_fraction: float = 10.0, alpha: float = 4.0,
                            beta: float = -4.0, gamma: float = 128.0) -> np.ndarray:
    """
    Ben Graham's local-average color subtraction, as used in the original
    Kaggle DR competition winning pipeline.

    img: BGR uint8 image.
    sigma_fraction: image_width / sigma_fraction sets the Gaussian blur sigma.
    alpha, beta, gamma: cv2.addWeighted params -> alpha*img + beta*blurred + gamma.
    """
    img = img.astype(np.float32)
    sigma = max(img.shape[1] / sigma_fraction, 1.0)
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    result = cv2.addWeighted(img, alpha, blurred, beta, gamma)
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_clahe_green_channel(img: np.ndarray, clip_limit: float = 2.0,
                               tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE to the green channel only (primary path). Green channel
    carries the most contrast for retinal lesions (vessels, microaneurysms,
    hemorrhages) due to hemoglobin absorption characteristics.

    img: BGR uint8 image. Returns BGR uint8 image with green channel CLAHE'd.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    b, g, r = cv2.split(img)
    g_eq = clahe.apply(g)
    return cv2.merge([b, g_eq, r])


def apply_clahe_all_channels(img: np.ndarray, clip_limit: float = 2.0,
                              tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Ablation option: CLAHE applied independently to all 3 BGR channels
    instead of green-only. Cheap secondary check, not the required path.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    channels = cv2.split(img)
    eq_channels = [clahe.apply(c) for c in channels]
    return cv2.merge(eq_channels)


def circular_mask(img: np.ndarray, margin_px: int = 2) -> np.ndarray:
    """
    Zero out residual corner artifacts outside the retina disc. Assumes the
    image is already square (post crop_and_resize) with the retina disc
    inscribed in the square.
    """
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    radius = min(h, w) // 2 - margin_px

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, thickness=-1)

    if img.ndim == 3:
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return cv2.bitwise_and(img, mask_3ch)
    return cv2.bitwise_and(img, mask)


def color_correction_pipeline(img: np.ndarray, use_all_channel_clahe: bool = False) -> np.ndarray:
    """Full steps 3-5: Ben Graham subtraction -> CLAHE -> circular mask."""
    img = ben_graham_subtraction(img)
    img = apply_clahe_all_channels(img) if use_all_channel_clahe else apply_clahe_green_channel(img)
    img = circular_mask(img)
    return img
