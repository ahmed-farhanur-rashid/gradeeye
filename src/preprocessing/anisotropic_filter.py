"""
Anisotropic (Perona-Malik) diffusion filtering.

Ablation candidate only (plan Section 2) — NOT part of the core pipeline.
Edge-aware denoising that preserves lesion boundaries, unlike isotropic
Gaussian blur. Run as a single on/off toggle on the full-method
configuration only, if time remains after the 3 core runs.
"""
import numpy as np


def perona_malik_diffusion(img: np.ndarray, num_iter: int = 10, kappa: float = 30.0,
                            gamma: float = 0.1, option: int = 1) -> np.ndarray:
    """
    Perona-Malik anisotropic diffusion, applied per-channel.

    img: BGR or grayscale uint8 image.
    num_iter: number of diffusion iterations.
    kappa: edge-sensitivity conductance parameter (higher = smooths more
           aggressively across weaker edges).
    gamma: step size (stability requires gamma <= 0.25 for 2D).
    option: 1 -> exponential conduction function (favors high-contrast edges),
            2 -> reciprocal conduction function (favors wide regions).
    """
    if img.ndim == 3:
        channels = [_diffuse_single_channel(img[:, :, c].astype(np.float32), num_iter, kappa, gamma, option)
                    for c in range(img.shape[2])]
        result = np.stack(channels, axis=2)
    else:
        result = _diffuse_single_channel(img.astype(np.float32), num_iter, kappa, gamma, option)

    return np.clip(result, 0, 255).astype(np.uint8)


def _diffuse_single_channel(channel: np.ndarray, num_iter: int, kappa: float,
                             gamma: float, option: int) -> np.ndarray:
    img = channel.copy()

    for _ in range(num_iter):
        # Gradients in the 4 cardinal directions.
        north = np.roll(img, 1, axis=0) - img
        south = np.roll(img, -1, axis=0) - img
        east = np.roll(img, -1, axis=1) - img
        west = np.roll(img, 1, axis=1) - img

        if option == 1:
            c_n = np.exp(-(north / kappa) ** 2)
            c_s = np.exp(-(south / kappa) ** 2)
            c_e = np.exp(-(east / kappa) ** 2)
            c_w = np.exp(-(west / kappa) ** 2)
        else:
            c_n = 1.0 / (1.0 + (north / kappa) ** 2)
            c_s = 1.0 / (1.0 + (south / kappa) ** 2)
            c_e = 1.0 / (1.0 + (east / kappa) ** 2)
            c_w = 1.0 / (1.0 + (west / kappa) ** 2)

        img += gamma * (c_n * north + c_s * south + c_e * east + c_w * west)

    return img


def apply_anisotropic_filter(img: np.ndarray, enabled: bool, **kwargs) -> np.ndarray:
    """Toggle wrapper — returns img unchanged if the ablation is disabled."""
    if not enabled:
        return img
    return perona_malik_diffusion(img, **kwargs)
