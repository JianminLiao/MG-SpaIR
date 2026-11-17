"""MG-SpaIR: multi-grade sparse-guided implicit representation for image restoration."""

import argparse
import csv
import importlib.util
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import data
import models
import utils


def load_config(path):
    spec = importlib.util.spec_from_file_location("config", path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


def pick_device(name, backbone):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MPS returns wrong gradients for the complex-valued WIRE layers (PyTorch 2.15).
    if torch.backends.mps.is_available() and backbone != "wire":
        return torch.device("mps")
    return torch.device("cpu")


def observe(x, kernel, scale):
    """Known degradation operator: Gaussian blur followed by bicubic downsampling."""
    if kernel is not None:
        x = utils.blur(x, kernel)
    return utils.downsample(x, scale)


def train_grade(net, grade_input, prev_output, observed, mask, u, kernel, cfg, lr, grade):
    """Train one grade on top of the frozen previous grades.

    Each epoch alternates a closed-form update of the auxiliary variable `u`
    (hard thresholding of the image gradients) with one optimizer step on the
    network. Returns the hidden features and the accumulated high-resolution
    output, which feed the next grade.
    """
    threshold, lam = cfg.THRESHOLD, cfg.LAMBDA
    # Hard thresholding at T is the proximal map of (lam / beta) * ||.||_0 for T = sqrt(2 * lam / beta).
    beta = 2 * lam / threshold ** 2
    epochs = cfg.EPOCHS
    decay = getattr(cfg, "LR_DECAY", 1.0)
    restore_best = getattr(cfg, "RESTORE_BEST", False)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: decay ** min(epoch / epochs, 1)
    )

    prev_low = observe(prev_output, kernel, cfg.SCALE)
    prev_grad = utils.gradients(prev_output)

    best_objective, best_state = float("inf"), None
    net.train()
    for epoch in range(epochs):
        _, out = net(grade_input)
        low = observe(out, kernel, cfg.SCALE) + prev_low
        grad = utils.gradients(out) + prev_grad

        with torch.no_grad():
            u.copy_(torch.where(grad.abs() > threshold, grad, torch.zeros_like(grad)))
            sparsity = lam * torch.count_nonzero(u).item()

        fidelity = (F.mse_loss(low, observed, reduction="none") * 0.5 * mask).sum()
        coupling = F.mse_loss(u, grad, reduction="sum") * (0.5 * beta)
        loss = fidelity + coupling

        optimizer.zero_grad()
        loss.backward()
        for p in net.parameters():
            if p.grad is not None:
                p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=1e3, neginf=-1e3)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 20)
        optimizer.step()
        scheduler.step()

        objective = loss.detach() + sparsity
        if restore_best and objective.item() < best_objective and torch.isfinite(objective):
            best_objective = objective.item()
            best_state = {k: v.clone() for k, v in net.state_dict().items()}

        if epoch % 1000 == 0:
            print(
                f"grade {grade} | epoch {epoch + 1}/{epochs} | objective {objective.item():.6f} "
                f"| fidelity {fidelity.item():.6f} | coupling {coupling.item():.6f} "
                f"| sparsity {sparsity:.6f}"
            )

    if best_state is not None:
        net.load_state_dict(best_state)

    net.eval()
    net.requires_grad_(False)
    with torch.no_grad():
        features, out = net(grade_input)
    return features, prev_output + out


def run(cfg, data_dir, device, seed=0):
    """Train all grades of a config. Returns the ground truth and, for each grade,
    the restored image with its PSNR and SSIM."""
    truth, observation, mask = data.load_observation(cfg, data_dir)
    observed = torch.from_numpy(observation).permute(2, 0, 1).unsqueeze(0).to(device)
    mask = torch.from_numpy(mask)[None, None].to(device)

    _, channels, h, w = observed.shape
    height, width = cfg.SCALE * h, cfg.SCALE * w
    kernel = utils.gaussian_kernel(cfg.BLUR_SIGMA, channels).to(device) if cfg.BLUR_SIGMA > 0 else None

    u = torch.zeros(1, 2 * channels, height, width, device=device)
    output = torch.zeros(1, channels, height, width, device=device)
    features = utils.coordinate_grid(height, width, device)
    lrs = cfg.LR if isinstance(cfg.LR, (list, tuple)) else [cfg.LR] * cfg.NUM_GRADES

    torch.manual_seed(seed)
    grades, results = [], []
    for g in range(cfg.NUM_GRADES):
        net = models.build_grade(
            cfg.BACKBONE, features.shape[-1], channels, cfg.LAYERS_PER_GRADE, cfg.HIDDEN_DIM
        ).to(device)
        features, output = train_grade(
            net, features, output, observed, mask, u, kernel, cfg, lrs[g], g + 1
        )
        grades.append(net)

        restored = models.render(grades, truth.shape[0], truth.shape[1], device)
        psnr, ssim = utils.psnr_ssim(truth, restored)
        results.append((restored, psnr, ssim))
        print(f"grade {g + 1} -> PSNR {psnr:.2f} dB, SSIM {ssim:.4f}")
    return truth, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a config file, e.g. configs/baboon_siren.py")
    parser.add_argument("--data", default="data", help="folder with the ground-truth images")
    parser.add_argument("--out", default="results", help="folder for restored images and metrics")
    parser.add_argument("--device", default=None, help="cuda, mps or cpu (default: best available)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    cfg = load_config(args.config)
    device = pick_device(args.device, cfg.BACKBONE)
    print(f"config {args.config} | device {device}")

    start = time.time()
    _, results = run(cfg, args.data, device, args.seed)
    print(f"total time: {time.time() - start:.1f} s")

    out_dir = os.path.join(args.out, os.path.splitext(os.path.basename(args.config))[0])
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["grade", "psnr", "ssim"])
        for g, (restored, psnr, ssim) in enumerate(results, start=1):
            writer.writerow([g, f"{psnr:.4f}", f"{ssim:.4f}"])
            image = (restored * 255).round().astype(np.uint8)
            Image.fromarray(image).save(os.path.join(out_dir, f"grade{g}.png"))
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
