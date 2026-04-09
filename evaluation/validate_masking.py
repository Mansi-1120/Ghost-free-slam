"""
Visual validation of masking quality across all three TUM RGBD sequences.

Checks for:
  - Missed detections  (frames where people exist but were not masked)
  - Partial masking    (mask region suspiciously small for a person)
  - Over-masking       (mask region is implausibly large)

Outputs:
  evaluation/validation_report/
    {seq}/comparison_*.png   – side-by-side originals vs masks (sampled frames)
    {seq}/failures_*.png     – flagged failure cases
    summary.txt              – per-sequence and overall statistics
"""

import os
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# ── paths ──────────────────────────────────────────────────────────────────────
BASE          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT  = "/projectnb/cs585/projects/dynamic_slam/dataset/tum_rgbd"
MASK_ROOT     = os.path.join(BASE, "masking", "masks")
OUTPUT_ROOT   = os.path.join(BASE, "evaluation", "validation_report")

SEQUENCES = [
    "rgbd_dataset_freiburg3_sitting_xyz",
    "rgbd_dataset_freiburg3_walking_static",
    "rgbd_dataset_freiburg3_walking_xyz",
]

# ── thresholds ─────────────────────────────────────────────────────────────────
# A connected black region smaller than this fraction of the image is "partial"
PARTIAL_MASK_FRAC  = 0.002   # < 0.2 % of pixels  (~614 px in 640×480)
# A connected black region larger than this fraction is "over-masking"
OVER_MASK_FRAC     = 0.30    # > 30 % of all pixels
# Minimum component area (px) to be counted as a masked person
MIN_COMPONENT_AREA = 200

# How many evenly-spaced frames to visualise per sequence
SAMPLE_COUNT = 12


# ── helpers ────────────────────────────────────────────────────────────────────

def load_rgb_list(seq_path: str) -> list[tuple[str, str]]:
    """Return list of (timestamp, relative_path) from rgb.txt."""
    rgb_txt = os.path.join(seq_path, "rgb.txt")
    pairs = []
    with open(rgb_txt) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 2:
                pairs.append((parts[0], parts[1]))
    return pairs


def mask_stats(masked_bgr: np.ndarray, orig_bgr: np.ndarray):
    """
    Analyse a masked image vs its original.

    Returns a dict with:
        black_frac       – fraction of pixels that are pure black in the masked img
        n_components     – number of connected black regions above MIN_COMPONENT_AREA
        largest_comp_frac– largest single component as fraction of image
        over_mask        – bool
        components       – list of {area, frac, partial, over} per component
    """
    H, W = masked_bgr.shape[:2]
    total = H * W

    # Binary: pixel is black (masked) only if ALL three channels == 0
    is_black = np.all(masked_bgr == 0, axis=2).astype(np.uint8)

    # Also mask pixels that were already black in the original
    orig_black = np.all(orig_bgr == 0, axis=2)
    new_black  = is_black & ~orig_black      # pixels blacked out by masking

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        new_black.astype(np.uint8), connectivity=8
    )

    components = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < MIN_COMPONENT_AREA:
            continue
        frac = area / total
        components.append({
            "area": area,
            "frac": frac,
            "partial": frac < PARTIAL_MASK_FRAC,
            "over":    frac > OVER_MASK_FRAC,
        })

    black_frac       = float(np.sum(new_black)) / total
    largest_frac     = max((c["frac"] for c in components), default=0.0)
    over_mask_global = black_frac > OVER_MASK_FRAC

    return {
        "black_frac":        black_frac,
        "n_components":      len(components),
        "largest_comp_frac": largest_frac,
        "over_mask":         over_mask_global,
        "components":        components,
    }


def make_comparison(orig: np.ndarray, masked: np.ndarray,
                    title: str, stats: dict) -> np.ndarray:
    """Create a side-by-side comparison image with annotation."""
    orig_rgb   = cv2.cvtColor(orig,   cv2.COLOR_BGR2RGB)
    masked_rgb = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(orig_rgb);   axes[0].set_title("Original", fontsize=11)
    axes[1].imshow(masked_rgb); axes[1].set_title("Masked",   fontsize=11)

    issues = []
    if stats["over_mask"]:
        issues.append(f"OVER-MASK  {stats['black_frac']*100:.1f}% blacked")
    for c in stats["components"]:
        if c["partial"]:
            issues.append(f"PARTIAL  {c['frac']*100:.2f}%")
    if not stats["components"]:
        issues.append("NO MASK REGION FOUND")

    colour = "red" if issues else "green"
    subtitle = "  |  ".join(issues) if issues else f"OK  ({stats['n_components']} regions, {stats['black_frac']*100:.1f}% masked)"
    fig.suptitle(f"{title}\n{subtitle}", fontsize=9, color=colour)
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img_arr = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return img_arr


def tile_images(imgs: list, cols: int = 2) -> np.ndarray:
    """Tile a list of same-shape images into a grid."""
    if not imgs:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    rows = (len(imgs) + cols - 1) // cols
    h, w = imgs[0].shape[:2]
    canvas = np.ones((rows * h, cols * w, 3), dtype=np.uint8) * 200
    for idx, img in enumerate(imgs):
        r, c = divmod(idx, cols)
        canvas[r*h:(r+1)*h, c*w:(c+1)*w] = img
    return canvas


# ── per-sequence validation ────────────────────────────────────────────────────

def validate_sequence(seq: str, summary_lines: list):
    print(f"\n{'='*60}")
    print(f"Validating: {seq}")
    print(f"{'='*60}")

    orig_seq  = os.path.join(DATASET_ROOT, seq)
    mask_seq  = os.path.join(MASK_ROOT,    seq)
    out_dir   = os.path.join(OUTPUT_ROOT,  seq)
    os.makedirs(out_dir, exist_ok=True)

    rgb_pairs   = load_rgb_list(orig_seq)        # all original frames
    mask_fnames = set(os.listdir(mask_seq))      # masked frames that exist

    total_frames  = len(rgb_pairs)
    missed_frames = [(ts, rp) for ts, rp in rgb_pairs
                     if f"{ts}.png" not in mask_fnames]
    masked_pairs  = [(ts, rp) for ts, rp in rgb_pairs
                     if f"{ts}.png" in mask_fnames]

    print(f"  Total frames   : {total_frames}")
    print(f"  Masked frames  : {len(masked_pairs)}")
    print(f"  Missed frames  : {len(missed_frames)}  "
          f"({100*len(missed_frames)/total_frames:.1f}%)")

    # ── analyse every masked frame for partial / over-mask ────────────────────
    partial_frames  = []
    over_frames     = []
    ok_frames       = []
    black_fracs     = []

    for ts, rel_path in masked_pairs:
        orig_path   = os.path.join(orig_seq, rel_path)
        masked_path = os.path.join(mask_seq, f"{ts}.png")
        orig   = cv2.imread(orig_path)
        masked = cv2.imread(masked_path)
        if orig is None or masked is None:
            continue

        st = mask_stats(masked, orig)
        black_fracs.append(st["black_frac"])

        entry = (ts, rel_path, orig, masked, st)
        has_partial = any(c["partial"] for c in st["components"])
        if st["over_mask"]:
            over_frames.append(entry)
        elif has_partial:
            partial_frames.append(entry)
        else:
            ok_frames.append(entry)

    avg_mask = float(np.mean(black_fracs)) if black_fracs else 0.0
    print(f"  Over-mask frames : {len(over_frames)}  "
          f"({100*len(over_frames)/max(1,len(masked_pairs)):.1f}%)")
    print(f"  Partial frames   : {len(partial_frames)}  "
          f"({100*len(partial_frames)/max(1,len(masked_pairs)):.1f}%)")
    print(f"  OK frames        : {len(ok_frames)}")
    print(f"  Avg masked area  : {avg_mask*100:.2f}%")

    # ── build summary ─────────────────────────────────────────────────────────
    seq_label = seq.split("freiburg3_")[-1]
    summary_lines += [
        f"\n[{seq_label}]",
        f"  Total frames          : {total_frames}",
        f"  Frames with mask      : {len(masked_pairs)}",
        f"  Missed (no mask saved): {len(missed_frames)} "
            f"({100*len(missed_frames)/total_frames:.1f}%)",
        f"  Over-mask cases       : {len(over_frames)} "
            f"({100*len(over_frames)/max(1,len(masked_pairs)):.1f}%)",
        f"  Partial-mask cases    : {len(partial_frames)} "
            f"({100*len(partial_frames)/max(1,len(masked_pairs)):.1f}%)",
        f"  Average masked area   : {avg_mask*100:.2f}%",
    ]

    # ── generate comparison grid (sampled frames) ─────────────────────────────
    # sample evenly across masked frames
    indices = np.linspace(0, len(masked_pairs)-1, SAMPLE_COUNT, dtype=int)
    sampled = [masked_pairs[i] for i in indices]

    comp_imgs = []
    for ts, rel_path in sampled:
        orig_path   = os.path.join(orig_seq, rel_path)
        masked_path = os.path.join(mask_seq, f"{ts}.png")
        orig   = cv2.imread(orig_path)
        masked = cv2.imread(masked_path)
        if orig is None or masked is None:
            continue
        st    = mask_stats(masked, orig)
        cimg  = make_comparison(orig, masked, f"ts={ts}", st)
        comp_imgs.append(cimg)

    if comp_imgs:
        grid = tile_images(comp_imgs, cols=2)
        out_path = os.path.join(out_dir, "comparison_grid.png")
        cv2.imwrite(out_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"  Saved comparison grid → {out_path}")

    # ── save failure panels ───────────────────────────────────────────────────
    def save_failure_panel(frames, label, filename):
        if not frames:
            return
        sample = frames[:8]   # at most 8 examples
        imgs = []
        for ts, rel_path, orig, masked, st in sample:
            cimg = make_comparison(orig, masked, f"ts={ts}", st)
            imgs.append(cimg)
        grid  = tile_images(imgs, cols=2)
        fpath = os.path.join(out_dir, filename)
        cv2.imwrite(fpath, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"  Saved {label} panel ({len(frames)} total) → {fpath}")

    save_failure_panel(over_frames,    "over-mask",    "failures_over_mask.png")
    save_failure_panel(partial_frames, "partial-mask", "failures_partial_mask.png")

    # ── missed-frame examples ─────────────────────────────────────────────────
    if missed_frames:
        sample_missed = missed_frames[::max(1, len(missed_frames)//8)][:8]
        miss_imgs = []
        for ts, rel_path in sample_missed:
            orig_path = os.path.join(orig_seq, rel_path)
            orig = cv2.imread(orig_path)
            if orig is None:
                continue
            orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.imshow(orig_rgb)
            ax.set_title(f"MISSED DETECTION\nts={ts}", fontsize=9, color="red")
            ax.axis("off")
            plt.tight_layout()
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            img_arr = np.frombuffer(fig.canvas.tostring_rgb(),
                                    dtype=np.uint8).reshape(h, w, 3)
            plt.close(fig)
            miss_imgs.append(img_arr)

        if miss_imgs:
            grid  = tile_images(miss_imgs, cols=2)
            fpath = os.path.join(out_dir, "failures_missed_detection.png")
            cv2.imwrite(fpath, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
            print(f"  Saved missed-detection panel → {fpath}")

    # ── mask coverage histogram ───────────────────────────────────────────────
    if black_fracs:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist([f*100 for f in black_fracs], bins=40, color="steelblue",
                edgecolor="white", linewidth=0.5)
        ax.axvline(PARTIAL_MASK_FRAC*100, color="orange", linestyle="--",
                   label=f"Partial threshold ({PARTIAL_MASK_FRAC*100:.1f}%)")
        ax.axvline(OVER_MASK_FRAC*100,    color="red",    linestyle="--",
                   label=f"Over-mask threshold ({OVER_MASK_FRAC*100:.0f}%)")
        ax.set_xlabel("Masked area (% of frame)")
        ax.set_ylabel("Number of frames")
        ax.set_title(f"Mask coverage distribution – {seq_label}")
        ax.legend()
        plt.tight_layout()
        hist_path = os.path.join(out_dir, "coverage_histogram.png")
        plt.savefig(hist_path, dpi=120)
        plt.close(fig)
        print(f"  Saved coverage histogram → {hist_path}")

    return {
        "total": total_frames,
        "masked": len(masked_pairs),
        "missed": len(missed_frames),
        "over": len(over_frames),
        "partial": len(partial_frames),
        "avg_mask_pct": avg_mask * 100,
    }


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    summary_lines = ["MASKING QUALITY VALIDATION REPORT",
                     "=" * 50]
    all_stats = {}

    for seq in SEQUENCES:
        seq_stats = validate_sequence(seq, summary_lines)
        all_stats[seq] = seq_stats

    # overall totals
    total_f   = sum(v["total"]   for v in all_stats.values())
    total_m   = sum(v["masked"]  for v in all_stats.values())
    total_mis = sum(v["missed"]  for v in all_stats.values())
    total_ov  = sum(v["over"]    for v in all_stats.values())
    total_pa  = sum(v["partial"] for v in all_stats.values())

    summary_lines += [
        "\n" + "=" * 50,
        "OVERALL TOTALS",
        f"  Total frames              : {total_f}",
        f"  Frames with mask saved    : {total_m} ({100*total_m/total_f:.1f}%)",
        f"  Missed detections (total) : {total_mis} ({100*total_mis/total_f:.1f}%)",
        f"  Over-mask cases           : {total_ov} ({100*total_ov/max(1,total_m):.1f}% of masked)",
        f"  Partial-mask cases        : {total_pa} ({100*total_pa/max(1,total_m):.1f}% of masked)",
    ]

    report_path = os.path.join(OUTPUT_ROOT, "summary.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"\nReport saved → {report_path}")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
