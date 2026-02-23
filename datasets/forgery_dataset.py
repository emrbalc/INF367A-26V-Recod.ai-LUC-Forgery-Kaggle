import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from dataset_utils import load_image, load_union_mask


class ForgeryDataset(Dataset):
    def __init__(self, case_ids, target_size: int = 256):
        self.case_ids = case_ids
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.case_ids)

    def _resize(self, arr: np.ndarray, size: int, is_mask: bool = False) -> np.ndarray:
        arr = np.squeeze(arr)

        if arr.ndim == 1:
            arr = arr.reshape((1, -1))

        if arr.ndim == 3:
            arr = arr.mean(axis=2)

        img = Image.fromarray(arr.astype(np.uint8))
        if is_mask:
            img = img.resize((size, size), resample=Image.NEAREST)
        else:
            img = img.resize((size, size), resample=Image.BILINEAR)
        return np.array(img)

    def __getitem__(self, idx: int):
        cid = self.case_ids[idx]
        img = load_image(cid)
        mask = load_union_mask(cid)
        mask = np.asarray(mask)

        if mask.ndim == 3:
            mask = (mask.max(axis=0) > 0).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)

        img = self._resize(img, self.target_size, is_mask=False)
        mask = self._resize(mask * 255, self.target_size, is_mask=True)

        img = (img / 255.0).astype(np.float32)
        mask = (mask > 0).astype(np.float32)

        img = torch.from_numpy(img).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)
        return img, mask
