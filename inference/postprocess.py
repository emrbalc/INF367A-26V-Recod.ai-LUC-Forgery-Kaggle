import numpy as np

from util.pixelmapUtil import PixelMapUtil


def harden_probabilities(
    probs: np.ndarray,
    temperature: float,
    clip_low: float,
    clip_high: float,
) -> np.ndarray:
    eps = 1e-6
    probs = np.clip(probs, eps, 1.0 - eps)
    logits = np.log(probs / (1.0 - probs))
    hardened = 1.0 / (1.0 + np.exp(-(logits / temperature)))
    hardened = np.where(hardened <= clip_low, 0.0, hardened)
    hardened = np.where(hardened >= clip_high, 1.0, hardened)
    return hardened.astype(np.float32, copy=False)


def post_process_prediction(
    probs: np.ndarray,
    pixel_util: PixelMapUtil,
    threshold: float,
    harden_temperature: float,
    hard_clip_low: float,
    hard_clip_high: float,
) -> np.ndarray:
    probs = harden_probabilities(
        probs,
        temperature=harden_temperature,
        clip_low=hard_clip_low,
        clip_high=hard_clip_high,
    )
    return pixel_util.post_process_mask_probs(probs, threshold=threshold)
