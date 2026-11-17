import os

import numpy as np
from PIL import Image, ImageFilter


def load_rgb(path):
    """Load an image as float32 RGB in [0, 1], shape [H, W, 3]."""
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def degrade(image, blur_sigma, scale, noise_std, missing_prob, seed=0):
    """Blur, downsample, add Gaussian noise and drop pixels at random.

    `image` is a PIL RGB image. Returns the 8-bit observation [h, w, 3] and the
    boolean mask [h, w] of observed pixels; missing pixels are set to zero.
    """
    rng = np.random.default_rng(seed)
    if blur_sigma > 0:
        image = image.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
    if scale > 1:
        image = image.resize((image.width // scale, image.height // scale), Image.BICUBIC)
    x = np.asarray(image, dtype=np.float32) / 255.0
    x = np.clip(x + rng.normal(0.0, noise_std, x.shape), 0.0, 1.0)
    mask = rng.random(x.shape[:2]) > missing_prob
    x = x * mask[..., None]
    return (x * 255).astype(np.uint8), mask


def load_observation(cfg, data_dir):
    """Return (ground truth, observation, mask) for a config as float32 arrays.

    The degraded input is read from `<data_dir>/degraded`; if it is not there it
    is generated from the ground truth and stored for later runs.
    """
    truth_path = os.path.join(data_dir, cfg.IMAGE)
    stem = os.path.splitext(cfg.IMAGE)[0]
    name = (
        f"{stem}_blur{cfg.BLUR_SIGMA:g}_x{cfg.SCALE}"
        f"_noise{round(cfg.NOISE_STD * 255)}_miss{cfg.MISSING_PROB:g}"
    )
    observation_path = os.path.join(data_dir, "degraded", name + ".png")
    mask_path = os.path.join(data_dir, "degraded", name + "_mask.png")

    if not (os.path.exists(observation_path) and os.path.exists(mask_path)):
        observation, mask = degrade(
            Image.open(truth_path).convert("RGB"),
            cfg.BLUR_SIGMA, cfg.SCALE, cfg.NOISE_STD, cfg.MISSING_PROB,
        )
        os.makedirs(os.path.dirname(observation_path), exist_ok=True)
        Image.fromarray(observation).save(observation_path)
        Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)

    mask = np.asarray(Image.open(mask_path).convert("L")) > 0
    return load_rgb(truth_path), load_rgb(observation_path), mask.astype(np.float32)
