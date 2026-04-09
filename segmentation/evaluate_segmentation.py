import os
import cv2
import numpy as np

def evaluate_metrics():
    base_dir = os.path.join(os.getcwd(), "segmentation/masks")
    
    sequences = [
        "rgbd_dataset_freiburg3_sitting_xyz",
        "rgbd_dataset_freiburg3_walking_static",
        "rgbd_dataset_freiburg3_walking_xyz"
    ]

    # this needs to be modified:
    # "ground truth 2 bc each frame in this set should have 2 people" is an incorrect assumption
    GROUND_TRUTH_COUNT = 2

    for seq in sequences:
        seq_path = os.path.join(base_dir, seq)
        if not os.path.exists(seq_path):
            print(f"Skipping {seq}: Directory not found.")
            continue

        tp, fp, fn = 0, 0, 0
        frame_files = [f for f in os.listdir(seq_path) if f.endswith('.png')]
        
        print(f"Analyzing {seq} ({len(frame_files)} frames)...")

        for frame_file in frame_files:
            mask_path = os.path.join(seq_path, frame_file)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            
            detected_count = 0
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] > 500:
                    detected_count += 1

            # metrics calculation
            tp += min(detected_count, GROUND_TRUTH_COUNT)
            fp += max(0, detected_count - GROUND_TRUTH_COUNT)
            fn += max(0, GROUND_TRUTH_COUNT - detected_count)

        # final metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

        # Save results to a file in the segmentation folder
        output_file = os.path.join(os.getcwd(), "segmentation", f"{seq}_metrics.txt")
        
        with open(output_file, 'w') as f:
            f.write(f"Results for sequence: {seq}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Frames: {len(frame_files)}\n")
            f.write(f"Ground Truth (People per frame): {GROUND_TRUTH_COUNT}\n\n")
            f.write("CONFUSION MATRIX DATA\n")
            f.write(f"True Positives (TP):  {tp}\n")
            f.write(f"False Positives (FP): {fp}\n")
            f.write(f"False Negatives (FN): {fn}\n")
            f.write(f"True Negatives (TN):  0 (N/A for detection)\n\n")
            f.write("PERFORMANCE METRICS\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall:    {recall:.4f}\n")
            f.write(f"F1 Score:  {f1_score:.4f}\n")
            f.write(f"Accuracy:  {accuracy:.4f}\n")

        print(f"Results saved to {output_file}")

if __name__ == "__main__":
    evaluate_metrics()