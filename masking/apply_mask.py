"""
masking/apply_mask.py
─────────────────────────────────────────────────────────────────────────────
Ghost-Free SLAM — Dynamic Pixel Masking
─────────────────────────────────────────────────────────────────────────────
Takes an RGB (or RGB-D) image and a binary segmentation mask produced by the
YOLO-based segmentation stage, blacks-out (or optionally zeros) every dynamic
pixel, and writes the result to masking/masked_frames/.

Inputs
------
  --image   Path to the input image  (JPEG / PNG / any OpenCV-readable format)
  --mask    Path to the binary mask  (white=dynamic/255, black=static/0)
            OR a comma-separated list of bounding boxes: x1,y1,x2,y2[;x1,y1,x2,y2;...]

Outputs
-------
  masking/masked_frames/<stem>_masked.<ext>   — masked RGB frame
  masking/masked_frames/<stem>_overlay.<ext>  — semi-transparent overlay for QC

Optional flags
--------------
  --output-dir   Override the output directory          [default: masking/masked_frames]
  --fill         Pixel value to fill masked regions     [default: 0  (black)]
  --alpha        Overlay transparency 0-1               [default: 0.45]
  --invert-mask  Treat white pixels as STATIC instead of dynamic
  --save-comparison  Save a side-by-side before/after PNG

Usage examples
--------------
  # With a pre-computed binary mask:
  python masking/apply_mask.py --image frames/rgb/0001.png --mask masks/0001_mask.png

  # With bounding boxes (quick test / fallback):
  python masking/apply_mask.py --image frames/rgb/0001.png \
      --mask "100,80,320,450;400,60,550,400"

  # Full options:
  python masking/apply_mask.py \
      --image  frames/rgb/0001.png  \
      --mask   masks/0001_mask.png  \
      --output-dir  masking/masked_frames \
      --fill   0    \
      --alpha  0.45 \
      --save-comparison
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import os
import sys
import numpy as np
import cv2


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    """Load an image from disk in BGR order (OpenCV native)."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img


def load_or_build_mask(mask_arg: str, height: int, width: int) -> np.ndarray:
    """
    Return a uint8 binary mask (0 = static, 255 = dynamic) of shape (H, W).

    mask_arg can be:
      • A path to an image file → loaded and thresholded at 127
      • A bounding-box string  → "x1,y1,x2,y2[;x1,y1,x2,y2;...]"
    """
    # ── try bounding-box string first ────────────────────────────────────────
    if not os.path.exists(mask_arg):
        mask = np.zeros((height, width), dtype=np.uint8)
        try:
            for box_str in mask_arg.strip().split(";"):
                parts = [int(v.strip()) for v in box_str.split(",")]
                if len(parts) != 4:
                    raise ValueError(f"Expected 4 values, got {len(parts)}: '{box_str}'")
                x1, y1, x2, y2 = parts
                x1, x2 = max(0, min(x1, width)),  max(0, min(x2, width))
                y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
                mask[y1:y2, x1:x2] = 255
        except ValueError as exc:
            sys.exit(
                f"[apply_mask] --mask is neither a valid file nor a valid bbox string.\n"
                f"  Details: {exc}"
            )
        return mask

    # ── load from file ───────────────────────────────────────────────────────
    raw = cv2.imread(mask_arg, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        sys.exit(f"[apply_mask] Could not load mask file: {mask_arg}")
    if raw.shape[:2] != (height, width):
        raw = cv2.resize(raw, (width, height), interpolation=cv2.INTER_NEAREST)
    _, binary = cv2.threshold(raw, 127, 255, cv2.THRESH_BINARY)
    return binary


def apply_mask(image: np.ndarray,
               mask: np.ndarray,
               fill: int = 0,
               invert: bool = False) -> np.ndarray:
    """
    Zero-out (or fill) every dynamic pixel.

    Parameters
    ----------
    image   : (H, W, 3) BGR image
    mask    : (H, W)    binary mask — 255 = dynamic, 0 = static
    fill    : scalar fill value for masked pixels  [default 0]
    invert  : if True, treat 255 as static (masks the complement)

    Returns
    -------
    masked  : (H, W, 3) image with dynamic regions replaced by `fill`
    """
    if invert:
        dynamic = (mask == 0)
    else:
        dynamic = (mask == 255)

    result = image.copy()
    result[dynamic] = fill
    return result


def build_overlay(original: np.ndarray,
                  mask: np.ndarray,
                  alpha: float = 0.45,
                  color_bgr: tuple = (0, 0, 220)) -> np.ndarray:
    """
    Blend a semi-transparent red highlight over dynamic regions for QC.

    Returns an (H, W, 3) BGR image.
    """
    overlay = original.copy()
    dynamic = mask == 255
    highlight = np.zeros_like(original)
    highlight[dynamic] = color_bgr
    overlay = cv2.addWeighted(overlay, 1.0, highlight, alpha, 0)
    # Draw a thin contour for readability
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 1)
    return overlay


def save_comparison(original: np.ndarray,
                    overlay: np.ndarray,
                    masked: np.ndarray,
                    out_path: str) -> None:
    """Save a three-panel before / highlight / after comparison image."""
    h = original.shape[0]
    label_h = 28
    panel_w = original.shape[1]

    def label_panel(img: np.ndarray, text: str) -> np.ndarray:
        canvas = np.zeros((h + label_h, panel_w, 3), dtype=np.uint8)
        canvas[label_h:, :] = img
        cv2.putText(canvas, text, (6, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        return canvas

    panels = [
        label_panel(original, "Original"),
        label_panel(overlay,  "Dynamic regions (overlay)"),
        label_panel(masked,   "Masked (output to SLAM)"),
    ]
    comparison = np.hstack(panels)
    cv2.imwrite(out_path, comparison)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Ghost-Free SLAM — apply dynamic-pixel mask to an RGB frame."
    )
    p.add_argument("--image",    required=True,
                   help="Path to the input RGB image.")
    p.add_argument("--mask",     required=True,
                   help="Binary mask file OR bbox string 'x1,y1,x2,y2[;...]'.")
    p.add_argument("--output-dir", default="masking/masked_frames",
                   help="Directory to write outputs  [default: masking/masked_frames]")
    p.add_argument("--fill",     type=int, default=0,
                   help="Fill value for masked pixels (0=black, 128=grey, …)")
    p.add_argument("--alpha",    type=float, default=0.45,
                   help="Overlay transparency 0-1  [default: 0.45]")
    p.add_argument("--invert-mask", action="store_true",
                   help="Treat white pixels as STATIC (invert mask polarity).")
    p.add_argument("--save-comparison", action="store_true",
                   help="Save a side-by-side before/after comparison image.")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load inputs ──────────────────────────────────────────────────────────
    image = load_image(args.image)
    h, w  = image.shape[:2]
    mask  = load_or_build_mask(args.mask, h, w)

    if args.invert_mask:
        mask = cv2.bitwise_not(mask)

    # ── Compute outputs ──────────────────────────────────────────────────────
    masked  = apply_mask(image, mask, fill=args.fill)
    overlay = build_overlay(image, mask, alpha=args.alpha)

    # ── Write outputs ────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    stem, ext = os.path.splitext(os.path.basename(args.image))

    masked_path  = os.path.join(args.output_dir, f"{stem}_masked{ext}")
    overlay_path = os.path.join(args.output_dir, f"{stem}_overlay{ext}")

    cv2.imwrite(masked_path,  masked)
    cv2.imwrite(overlay_path, overlay)

    print(f"[apply_mask] Wrote masked image → {masked_path}")
    print(f"[apply_mask] Wrote overlay      → {overlay_path}")

    if args.save_comparison:
        comp_path = os.path.join(args.output_dir, f"{stem}_comparison{ext}")
        save_comparison(image, overlay, masked, comp_path)
        print(f"[apply_mask] Wrote comparison   → {comp_path}")

    # ── Quick stats ──────────────────────────────────────────────────────────
    dynamic_px  = int(np.sum(mask == 255))
    total_px    = h * w
    pct         = 100.0 * dynamic_px / total_px
    print(f"[apply_mask] Masked {dynamic_px:,} / {total_px:,} pixels  ({pct:.1f}% dynamic)")


if __name__ == "__main__":
    main()
