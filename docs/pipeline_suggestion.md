# Pipeline

_The following is a suggested pipeline, based on what is commonly done for this task on Kaggle. For an example, see: https://www.kaggle.com/code/gauravparkhedkar/dinov2-base-0-332-high-res-4500px-robust-inf_

1. Feature extraction on images using Dino (encoder) -> embeddings
2. Use CNN (decoder) to create pixel-level probability maps for each pixel
3. Do sliding window inference on each image (necessary because large images can eat up our GPU memory) and accumulate results. If the image is small enough to fit in memory, we can predict all at once.

At this point we could in theory just output pixel-predictions based on some probability threshold for each pixel, and output forgery if any pixel is 1. However, this is too sensitive, and we want to make predictions more robust. This is done twofold.

For the first part, a suggestion here that is used in an example code notebook is to do some flips of the image for each patch and do predictions for each flip. Then we take the average to compute the probabilities.

For the second part, we want a smoother mask than just the pixels that happen to be activated - our forged objects will have pixels that are connected. To handle this, we calculate sobel gradients to get edges, apply a gaussian blur to smooth them and apply some morphological close and open operations to connect the pixels.

Finally, we can remove some smaller pixel areas as it is unlikely that they represent true forged areas rather than some random pixels.

## Conceptual summary diagram (created by LLM)

```
Input Image (RGB)
       │
       ▼
  ┌───────────────┐
  │ Preprocessing │  (AutoImageProcessor)
  └───────────────┘
       │
       ▼
  ┌───────────────┐
  │  DINOv2 ViT   │  (Encoder / Patch embeddings)
  └───────────────┘
       │
       ▼
  ┌───────────────┐
  │  CNN Decoder  │  (DinoTinyDecoder)
  └───────────────┘
       │
       ▼
Sliding Window & TTA → Full Probability Map
       │
       ▼
Global + Local Fusion (pipeline_fusion)
       │
       ▼
Enhanced Mask (enhanced_adaptive_mask)
       │
       ▼
Filter small blobs & fill holes
       │
       ▼
Final Mask + Label ("forged"/"authentic")
       │
       ▼
Optional RLE Encoding (rle_encode)
```
