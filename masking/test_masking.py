"""
masking/test_masking.py
────────────────────────────────────────────────────────────────────────────
Ghost-Free SLAM — Dummy-mask test suite
────────────────────────────────────────────────────────────────────────────
Creates 4 synthetic test cases, applies masks, verifies outputs pixel-level,
and saves a visual report to masking/test_results/.

Test cases
──────────
  1. Single bbox mask        — one rectangle blacked out
  2. Multi-bbox mask         — two disjoint rectangles blacked out
  3. Full binary mask file   — PNG mask loaded from disk
  4. Full-frame mask         — entire image masked (edge case)

Outputs per test
────────────────
  *_comparison.png   — 3-panel: Original / Overlay / Masked
  *_diff.png         — false-colour pixel difference heatmap
  report.csv         — per-image metrics (% masked, MAE, MSE, mean diff)

Run:
  python masking/test_masking.py
"""

import csv
import os
import sys
import numpy as np
import cv2

# ── paths ────────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR    = os.path.join(ROOT, "masking", "test_results")
MASK_DIR   = os.path.join(OUT_DIR, "masks")
FRAME_DIR  = os.path.join(OUT_DIR, "frames")
os.makedirs(OUT_DIR,   exist_ok=True)
os.makedirs(MASK_DIR,  exist_ok=True)
os.makedirs(FRAME_DIR, exist_ok=True)

# add project root so we can import apply_mask helpers directly
sys.path.insert(0, ROOT)
from masking.apply_mask import (load_or_build_mask, apply_mask, build_overlay,
                                save_comparison, compute_metrics, save_diff_image)

# ── helpers ───────────────────────────────────────────────────────────────────

def make_image(h=240, w=320, seed=42) -> np.ndarray:
    """Colourful synthetic BGR image."""
    rng = np.random.default_rng(seed)
    img = rng.integers(40, 220, (h, w, 3), dtype=np.uint8)
    # paint a few coloured blobs so the image looks interesting
    cv2.rectangle(img, (20,  20), (120, 100), (200,  80,  40), -1)
    cv2.rectangle(img, (160, 80), (280, 180), ( 40, 160, 200), -1)
    cv2.circle(img, (w//2, h//2), 40, (220, 220,  60), -1)
    return img


def verify(test_name: str,
           original: np.ndarray,
           masked:   np.ndarray,
           mask:     np.ndarray) -> bool:
    """
    Pixel-level assertions:
      • Every dynamic pixel (mask==255) must equal fill value (0).
      • Every static pixel  (mask==0)   must be unchanged.
    Returns True if all checks pass.
    """
    dynamic = mask == 255
    static  = mask == 0

    ok = True

    # masked pixels should be 0
    if np.any(masked[dynamic] != 0):
        bad = int(np.sum(np.any(masked[dynamic] != 0, axis=-1)))
        print(f"  [FAIL] {test_name}: {bad} dynamic pixels were NOT zeroed out")
        ok = False

    # static pixels should be untouched
    if not np.array_equal(masked[static], original[static]):
        bad = int(np.sum(np.any(masked[static] != original[static], axis=-1)))
        print(f"  [FAIL] {test_name}: {bad} static pixels were unexpectedly changed")
        ok = False

    if ok:
        dyn_px  = int(dynamic.sum())
        stat_px = int(static.sum())
        pct     = 100.0 * dyn_px / (dyn_px + stat_px)
        print(f"  [PASS] {test_name}: {dyn_px:,} dynamic / {stat_px:,} static pixels  ({pct:.1f}% masked)")

    return ok


def run_test(test_name: str,
             image:     np.ndarray,
             mask_arg:  str) -> tuple:
    """
    Full pipeline for one test case:
      load mask → apply → verify → save comparison + diff PNG → return metrics.
    Returns (passed: bool, metrics: dict).
    """
    print(f"\n── {test_name} ──")
    h, w = image.shape[:2]

    frame_path = os.path.join(FRAME_DIR, f"{test_name}.png")
    cv2.imwrite(frame_path, image)

    mask    = load_or_build_mask(mask_arg, h, w)
    masked  = apply_mask(image, mask, fill=0)
    overlay = build_overlay(image, mask)

    comp_path = os.path.join(OUT_DIR, f"{test_name}_comparison.png")
    save_comparison(image, overlay, masked, comp_path)

    diff_path = os.path.join(OUT_DIR, f"{test_name}_diff.png")
    save_diff_image(image, masked, diff_path)

    print(f"  Saved comparison → {comp_path}")
    print(f"  Saved diff       → {diff_path}")

    metrics = compute_metrics(image, masked, mask)
    print(f"  Metrics: {metrics['pct_masked']:.1f}% masked | "
          f"MAE={metrics['mae']} | MSE={metrics['mse']} | "
          f"mean_diff_masked={metrics['mean_diff_masked']}")

    passed = verify(test_name, image, masked, mask)
    return passed, {"test": test_name, **metrics}


# ── test cases ────────────────────────────────────────────────────────────────

def test_single_bbox():
    img = make_image(seed=1)
    return run_test("test1_single_bbox", img, "60,40,200,160")


def test_multi_bbox():
    img = make_image(seed=2)
    return run_test("test2_multi_bbox", img, "10,10,100,80;180,100,300,200")


def test_mask_file():
    img = make_image(seed=3)
    h, w = img.shape[:2]
    mask_img = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask_img, (50, 30), (270, 130), 255, -1)
    cv2.circle(mask_img, (80, 180), 35, 255, -1)
    mask_path = os.path.join(MASK_DIR, "test3_mask.png")
    cv2.imwrite(mask_path, mask_img)
    return run_test("test3_mask_file", img, mask_path)


def test_full_frame_mask():
    img = make_image(seed=4)
    h, w = img.shape[:2]
    return run_test("test4_full_frame", img, f"0,0,{w},{h}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Ghost-Free SLAM — Masking Tests")
    print("=" * 60)

    outcomes = [
        test_single_bbox(),
        test_multi_bbox(),
        test_mask_file(),
        test_full_frame_mask(),
    ]

    passed      = sum(r[0] for r in outcomes)
    all_metrics = [r[1] for r in outcomes]
    total       = len(outcomes)

    # ── write CSV report ─────────────────────────────────────────────────────
    csv_path = os.path.join(OUT_DIR, "report.csv")
    fieldnames = ["test", "pixels_total", "pixels_masked", "pct_masked",
                  "mae", "mse", "mean_diff_masked"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_metrics)

    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} passed")
    print(f"Report  → {csv_path}")
    print(f"\n{'Test':<28} {'%Masked':>8} {'MAE':>8} {'MSE':>10} {'MeanDiffMasked':>16}")
    print("-" * 72)
    for m in all_metrics:
        print(f"{m['test']:<28} {m['pct_masked']:>7.1f}% "
              f"{m['mae']:>8.2f} {m['mse']:>10.2f} {m['mean_diff_masked']:>16.2f}")

    if passed < total:
        print("\nSome tests FAILED — see details above.")
        sys.exit(1)
    else:
        print("\nAll tests PASSED.")


if __name__ == "__main__":
    main()
