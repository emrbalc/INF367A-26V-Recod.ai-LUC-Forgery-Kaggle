import torch
from tqdm import tqdm


def train_one_epoch(
    model: torch.nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    grad_clip_max_norm: float,
    epoch_idx: int,
) -> float:
    model.train()
    total_loss = 0.0

    for imgs, masks in tqdm(train_loader, desc=f"epoch {epoch_idx + 1} train"):
        imgs = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = loss_fn(logits, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
        optimizer.step()
        total_loss += float(loss.item())

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return total_loss / max(1, len(train_loader))
