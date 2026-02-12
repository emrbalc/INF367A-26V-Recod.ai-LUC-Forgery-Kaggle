import random
import numpy as np

from dataset_utils import load_image, load_union_mask
from recodai_f1 import calculate_f1_score

# forged klasöründen rastgele 5 case seç
from pathlib import Path

forged_dir = Path("data/train_images/forged")
case_ids = [p.stem for p in forged_dir.glob("*.png")]
sample_ids = random.sample(case_ids, 5)

print("Testing case_ids:", sample_ids)
print("-" * 40)

for cid in sample_ids:
    img = load_image(cid)
    gt = load_union_mask(cid)

    # DUMMY prediction:
    # tamamen boş tahmin
    pred_empty = np.zeros_like(gt)
    f1_empty = calculate_f1_score(pred_empty, gt)

    # GT'yi aynen tahmin edersek
    f1_perfect = calculate_f1_score(gt, gt)

    print(f"case {cid}")
    print("  image shape:", img.shape)
    print("  gt mask pixels:", gt.sum())
    print("  F1(empty pred):", round(f1_empty, 4))
    print("  F1(perfect pred):", round(f1_perfect, 4))
    print()
