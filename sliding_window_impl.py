import numpy as np
import torch
import torch.nn.functional as F

PATCH_SIZE = 512 # DINO-dependent

STRIDE = PATCH_SIZE // 2

def gaussian_weight(patch_size, sigma=0.125):
    ax = np.linspace(-1, 1, patch_size)
    xx, yy = np.meshgrid(ax, ax)
    dist = np.sqrt(xx**2 + yy**2)
    weight = np.exp(-(dist**2) / (2 * sigma**2))
    return weight

def sliding_window(img, model):
    C, H, W = img.shape
    weight = gaussian_weight(PATCH_SIZE)[None]

    prob_map = torch.zeros((H, W))
    weight_map = torch.zeros((H, W))

    for y in range(0, H, STRIDE):
        for x in range(0, W, STRIDE):
            patch = img[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            pad_h = PATCH_SIZE - patch.shape[1]
            pad_w = PATCH_SIZE - patch.shape[2]

            patch = F.pad(patch, (0, pad_w, 0, pad_h), mode="reflect")

            pred = model(patch[None])[0]

            prob_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += pred * weight
            weight_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += weight

    final = prob_map / weight_map
    
    return final