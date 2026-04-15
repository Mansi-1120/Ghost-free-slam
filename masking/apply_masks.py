"""
Apply masks to the full dataset.

For every original RGB frame:
  - If a mask exists in masking/masks/<seq>/: derive clean binary mask
    (pixels black in masked but non-black in original), zero those pixels
    in the original (no bounding-box overlays).
  - If no mask exists (missed detection): copy original unchanged.

Output: masking/masked_frames/<seq>/<timestamp>.png

Also saves example comparison strips to masking/masked_frames/examples/.
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE         = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE)
DATASET_ROOT = "/projectnb/cs585/projects/dynamic_slam/dataset/tum_rgbd"
MASK_SRC     = os.path.join(BASE, "masks")
OUT_ROOT     = os.path.join(BASE, "masked_frames")
EXAMPLE_DIR  = os.path.join(OUT_ROOT, "examples")

SEQUENCES = [
    "rgbd_dataset_freiburg3_sitting_xyz",
    "rgbd_dataset_freiburg3_walking_static",
    "rgbd_dataset_freiburg3_walking_xyz",
]

# How many example frames to save per sequence
N_EXAMPLES = 6


def load_rgb_list(seq_path: str) -> list[tuple[str, str]]:
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


def apply_clean_mask(orig: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Derive person mask from annotated masked image and apply cleanly to original.
    Person pixels = black in masked_annot AND non-black in original.
    """
    orig_black  = np.all(orig == 0, axis=2)
    mask_black  = np.all(mask == 0, axis=2)
    person_mask = mask_black & ~orig_black
    result      = orig.copy()
    result[person_mask] = 0
    return result


def save_example_strip(orig: np.ndarray, result: np.ndarray,
                       label: str, out_path: str, was_missed: bool) -> None:
    orig_rgb   = cv2.cvtColor(orig,   cv2.COLOR_BGR2RGB)
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].imshow(orig_rgb)
    axes[0].set_title("Original", fontsize=10)
    axes[1].imshow(result_rgb)
    status = "MISSED — original copied" if was_missed else "Masked (clean)"
    colour = "darkorange" if was_missed else "green"
    axes[1].set_title(status, fontsize=10, color=colour)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(label, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def process_sequence(seq: str) -> dict:
    orig_seq  = os.path.join(DATASET_ROOT, seq)
    mask_seq  = os.path.join(MASK_SRC, seq)
    out_rgb   = os.path.join(OUT_ROOT, seq, "rgb")
    # out_depth = os.path.join(OUT_ROOT, seq, "depth") #uncomment if need to re copy depth folders
    os.makedirs(out_rgb,   exist_ok=True)
    # os.makedirs(out_depth, exist_ok=True) #uncomment if need to re copy depth folders

    rgb_pairs  = load_rgb_list(orig_seq)
    mask_files = set(os.listdir(mask_seq))

    n_masked = n_missed = n_total = 0
    example_indices = set(
        int(i) for i in np.linspace(0, len(rgb_pairs) - 1, N_EXAMPLES)
    )
    examples = []

    # --- RGB masked frames ---
    for idx, (ts, rel_path) in enumerate(rgb_pairs):
        n_total += 1
        orig_path = os.path.join(orig_seq, rel_path)
        out_path  = os.path.join(out_rgb, f"{ts}.png")

        orig = cv2.imread(orig_path)
        if orig is None:
            print(f"  [WARN] Cannot read {orig_path}")
            continue

        mask_fname = f"{ts}.png"
        if mask_fname in mask_files:
            masked_annot = cv2.imread(os.path.join(mask_seq, mask_fname))
            if masked_annot is not None:
                result     = apply_clean_mask(orig, masked_annot)
                was_missed = False
                n_masked  += 1
            else:
                result     = orig.copy()
                was_missed = True
                n_missed  += 1
        else:
            result     = orig.copy()
            was_missed = True
            n_missed   += 1

        cv2.imwrite(out_path, result)

        if idx in example_indices:
            examples.append((orig.copy(), result.copy(), ts, was_missed))

    # UNCOMMENT CHUNK IF RE-COPYING DEPTH FRAMES
    # # --- Copy depth frames ---
    # depth_src = os.path.join(orig_seq, "depth")
    # print(f"  Copying depth frames from {depth_src} ...")
    # for depth_file in os.listdir(depth_src):
    #     if depth_file.endswith(".png"):
    #         shutil.copy2(
    #             os.path.join(depth_src, depth_file),
    #             os.path.join(out_depth, depth_file)
    #         )
    # print(f"  Depth frames copied to {out_depth}")

    return {
        "seq": seq, "total": n_total,
        "masked": n_masked, "missed": n_missed,
        "examples": examples,
    }


def main():
    os.makedirs(EXAMPLE_DIR, exist_ok=True)
    all_stats = []

    for seq in SEQUENCES:
        print(f"\nProcessing {seq} ...")
        stats = process_sequence(seq)
        all_stats.append(stats)
        print(f"  {stats['total']} frames written  "
              f"({stats['masked']} masked, {stats['missed']} copied as-is)")

        # Save example strips
        seq_label = seq.split("freiburg3_")[-1]
        for i, (orig, result, ts, was_missed) in enumerate(stats["examples"]):
            fname = os.path.join(EXAMPLE_DIR,
                                 f"{seq_label}_example_{i+1:02d}.png")
            save_example_strip(orig, result,
                               f"{seq_label}  ts={ts}", fname, was_missed)

    # Summary
    print("\n" + "=" * 55)
    print("SUMMARY")
    print("=" * 55)
    grand_total = grand_masked = grand_missed = 0
    for s in all_stats:
        label = s["seq"].split("freiburg3_")[-1]
        pct_m = 100 * s["masked"] / max(1, s["total"])
        pct_miss = 100 * s["missed"] / max(1, s["total"])
        print(f"  {label:<20}  "
              f"{s['total']} frames  |  "
              f"masked={s['masked']} ({pct_m:.1f}%)  |  "
              f"copied={s['missed']} ({pct_miss:.1f}%)")
        grand_total  += s["total"]
        grand_masked += s["masked"]
        grand_missed += s["missed"]

    print(f"\n  {'TOTAL':<20}  "
          f"{grand_total} frames  |  "
          f"masked={grand_masked} ({100*grand_masked/grand_total:.1f}%)  |  "
          f"copied={grand_missed} ({100*grand_missed/grand_total:.1f}%)")
    print(f"\nOutputs  → {OUT_ROOT}")
    print(f"Examples → {EXAMPLE_DIR}")


if __name__ == "__main__":
    main()
