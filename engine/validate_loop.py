import numpy as np
import torch
from tqdm import tqdm

from inference.postprocess import post_process_prediction
from recodai_f1 import calculate_f1_score


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
    sliding_window_fn,
    pixel_util,
    pred_threshold: float,
    harden_temperature: float,
    hard_clip_low: float,
    hard_clip_high: float,
    min_component_area: int,
    epoch_idx: int,
) -> float:
    model.eval()
    f1s = []

    for imgs, masks in tqdm(val_loader, desc=f"epoch {epoch_idx + 1} val"):
        imgs = imgs.to(device)
        masks = masks.to(device)

        for i in range(imgs.shape[0]):
            img = imgs[i]   # (C,H,W)
            mask = masks[i] # usually (1,H,W)

            # Get logits
            if sliding_window_fn is None:
                # direct forward: model expects (B,C,H,W)
                logits = model(img.unsqueeze(0).to(device))  # (1,1,H,W) ideally
            else:
                # sliding window expects single image (C,H,W) and returns (1,H,W) or (H,W)
                logits = sliding_window_fn(img, model, device)

            # Convert logits to probs (numpy)
            probs = torch.sigmoid(logits).detach().cpu().numpy()

            # ---- FIX: ensure probs is 2D (H,W) for scipy morphology ----
            # possible shapes here:
            # - (1,1,H,W) from direct forward
            # - (1,H,W) from sliding window
            # - (H,W) already OK
            if probs.ndim == 4:
                # (B,1,H,W) -> take first item
                probs = probs[0]
            if probs.ndim == 3 and probs.shape[0] == 1:
                # (1,H,W) -> (H,W)
                probs = probs[0]
            # -----------------------------------------------------------

            probs = post_process_prediction(
                probs=probs,
                pixel_util=pixel_util,
                threshold=pred_threshold,
                harden_temperature=harden_temperature,
                hard_clip_low=hard_clip_low,
                hard_clip_high=hard_clip_high,
                min_component_area=min_component_area,
            )

            gt = mask.detach().cpu().numpy()[0]  # (H,W)

            pred_bin = (probs >= pred_threshold).astype(np.uint8)
            gt_bin = (gt >= 0.5).astype(np.uint8)
            f1s.append(calculate_f1_score(pred_bin, gt_bin))

    return float(np.mean(f1s))