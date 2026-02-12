from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# PNG okumak için
from PIL import Image

DATA = Path("data")
img_dir = DATA / "train_images" / "forged"
mask_dir = DATA / "train_masks"

# 1) forged klasöründen bir örnek PNG seç
img_files = sorted(img_dir.glob("*.png"))
print("forged image count:", len(img_files))
img_path = img_files[0]              # ilk dosya
case_id = img_path.stem              # "10015" gibi

# 2) Bu case_id ile maskeleri bul (npy veya png olabilir)
mask_paths = sorted(mask_dir.glob(f"{case_id}.npy")) + sorted(mask_dir.glob(f"{case_id}_*.npy"))
print("example case_id:", case_id)
print("mask files:", [p.name for p in mask_paths])

# 3) Görüntüyü yükle (PNG)
img = np.array(Image.open(img_path))
# eğer RGB geldiyse griye çevir (3 kanal)
if img.ndim == 3:
    img = img.mean(axis=2)

# 4) Maskeleri yükle + union yap
union = None
for mp in mask_paths:
    if mp.suffix == ".npy":
        m = np.load(mp)
    elif mp.suffix == ".png":
        m = np.array(Image.open(mp))
    else:
        continue

    m = (m > 0).astype(np.uint8)     # 0/1
    union = m if union is None else np.maximum(union, m)

print("image shape:", img.shape, "dtype:", img.dtype, "min/max:", img.min(), img.max())
print("mask shape:", union.shape, "unique:", np.unique(union))
union = np.squeeze(union)

# 5) Göster
plt.figure()
plt.title(f"Image {case_id}")
plt.imshow(img, cmap="gray")
plt.axis("off")

plt.figure()
plt.title(f"Mask union {case_id}")
plt.imshow(union, cmap="gray")
plt.axis("off")

plt.figure()
plt.title(f"Overlay {case_id}")
plt.imshow(img, cmap="gray")
plt.imshow(union, alpha=0.35)
plt.axis("off")

plt.show()


