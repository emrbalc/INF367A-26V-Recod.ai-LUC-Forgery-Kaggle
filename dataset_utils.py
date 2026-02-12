from pathlib import Path
import numpy as np
from PIL import Image

DATA = Path("data")

def _to_gray(img: np.ndarray) -> np.ndarray:
    """RGB gelirse griye çevir. Zaten griyse aynen bırak."""
    if img.ndim == 3:
        img = img.mean(axis=2)
    return img

def find_image_path(case_id: str) -> Path:
    """
    case_id için görüntü dosyasını bulur.
    Önce forged/authentic klasörlerine bakar, sonra gerekirse kök train_images'a bakar.
    """
    candidates = [
        DATA / "train_images" / "forged" / f"{case_id}.png",
        DATA / "train_images" / "authentic" / f"{case_id}.png",
        DATA / "train_images" / f"{case_id}.png",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Image not found for case_id={case_id}. Tried: {candidates}")

def load_image(case_id: str) -> np.ndarray:
    """Görüntüyü numpy array olarak döndürür (H,W)."""
    p = find_image_path(case_id)
    img = np.array(Image.open(p))
    img = _to_gray(img).astype(np.float32)
    return img

def find_mask_paths(case_id: str) -> list[Path]:
    """
    case_id için maske dosyalarını listeler.
    '10.npy' ve '10_0.npy' gibi isimleri destekler.
    """
    mask_dir = DATA / "train_masks"
    masks = sorted(mask_dir.glob(f"{case_id}.npy")) + sorted(mask_dir.glob(f"{case_id}_*.npy"))
    return masks

def load_union_mask(case_id: str) -> np.ndarray:
    """
    Birden fazla maske varsa union (OR) alıp tek (H,W) binary maske döndürür.
    """
    mask_paths = find_mask_paths(case_id)
    if len(mask_paths) == 0:
        # Bu normalde train'de forged için olmaz ama güvenlik için:
        raise FileNotFoundError(f"No masks found for case_id={case_id}")

    union = None
    for mp in mask_paths:
        m = np.load(mp)

        # m bazen (1,H,W) geliyor -> (H,W) yap
        m = np.squeeze(m)

        # 0/1 yap
        m = (m > 0).astype(np.uint8)

        union = m if union is None else np.maximum(union, m)

    return union
