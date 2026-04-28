"""
    python reprojection/reproject.py --sequence sitting_xyz
    python reprojection/reproject.py --sequence walking_static
    python reprojection/reproject.py --sequence walking_xyz
    python reprojection/reproject.py --all
"""

import os
import sys
import argparse
import numpy as np
import cv2
from pathlib import Path

FX = 517.306
FY = 516.469
CX = 318.643
CY = 255.314
DEPTH_FACTOR = 5000.0
IMG_W = 640
IMG_H = 480


def parse_trajectory(traj_path):
    poses = {}
    with open(traj_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) != 8:
                continue
            ts = parts[0]
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])

            R = quat_to_rot(qx, qy, qz, qw)

            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = [tx, ty, tz]
            poses[ts] = T

    return poses


def quat_to_rot(qx, qy, qz, qw):
    n = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if n < 1e-10:
        return np.eye(3)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n

    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
    ])
    return R


def backproject_depth(depth_img):
    h, w = depth_img.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    z = depth_img.astype(np.float64) / DEPTH_FACTOR
    valid = (z > 0.1) & (z < 10.0)

    x = (u - CX) * z / FX
    y = (v - CY) * z / FY

    points_3d = np.stack([x, y, z], axis=-1)
    return points_3d, valid


def reproject_frame(prev_rgb, prev_depth, curr_masked, prev_pose, curr_pose):
    h, w = curr_masked.shape[:2]
    recovered = curr_masked.copy()

    mask = np.all(curr_masked == 0, axis=2)
    if not np.any(mask):
        return recovered

    points_3d, valid_depth = backproject_depth(prev_depth)

    T = np.linalg.inv(curr_pose) @ prev_pose
    R = T[:3, :3]
    t = T[:3, 3]

    pts = points_3d.reshape(-1, 3)
    valid_flat = valid_depth.reshape(-1)

    idx = np.where(valid_flat)[0]
    if len(idx) == 0:
        return recovered

    pts = pts[idx]
    pts = (R @ pts.T).T + t

    z = pts[:, 2]
    valid_z = z > 0.01

    u = (FX * pts[:, 0] / z + CX).astype(int)
    v = (FY * pts[:, 1] / z + CY).astype(int)

    in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h) & valid_z

    u = u[in_bounds]
    v = v[in_bounds]

    src_v, src_u = np.unravel_index(idx[in_bounds], (h, w))

    valid_mask = mask[v, u]

    if not np.any(valid_mask):
        return recovered

    recovered[v[valid_mask], u[valid_mask]] = prev_rgb[src_v[valid_mask], src_u[valid_mask]]
    return recovered


def find_closest_timestamp(query_ts, pose_ts):
    query = float(query_ts)
    best, diff = None, float('inf')
    for ts in pose_ts:
        d = abs(float(ts) - query)
        if d < diff:
            diff = d
            best = ts
    return best, diff


def process_sequence(sequence_name):

    full_seq = f"rgbd_dataset_freiburg3_{sequence_name}"

    # ✅ CORRECT PATHS (FIXED)
    base = "/projectnb/cs585/projects/dynamic_slam/Ghost-free-slam"

    rgb_dir = os.path.join(base, "masking", "masked_frames", full_seq, "rgb")
    depth_dir = os.path.join(base, "masking", "masked_frames", full_seq, "depth")

    traj_path = os.path.join(
        "/projectnb/cs585/projects/dynamic_slam/trajectories/masked/camera_trajectories",
        f"masked_camera_{sequence_name}.txt"
    )

    out_dir = os.path.join(base, "reprojection", "recovered_frames", full_seq)

    for p, desc in [(rgb_dir, "rgb"), (depth_dir, "depth"), (traj_path, "trajectory")]:
        if not os.path.exists(p):
            print(f"[ERROR] {desc} not found: {p}")
            return False

    os.makedirs(out_dir, exist_ok=True)

    poses = parse_trajectory(traj_path)
    pose_ts = sorted(poses.keys())

    frames = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png')])
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.png')])
    depth_ts = [(float(f.split('.')[0]), f) for f in depth_files]

    def find_depth(rgb):
        t = float(rgb.split('.')[0])
        best = min(depth_ts, key=lambda x: abs(x[0] - t))
        if abs(best[0] - t) < 0.05:
            return os.path.join(depth_dir, best[1])
        return None

    for i, f in enumerate(frames):
        img = cv2.imread(os.path.join(rgb_dir, f))
        if img is None:
            continue

        ts = f.split('.')[0]
        pose_ts_match, diff = find_closest_timestamp(ts, pose_ts)

        if diff > 0.05:
            cv2.imwrite(os.path.join(out_dir, f), img)
            continue

        pose = poses[pose_ts_match]

        depth_path = find_depth(f)
        if depth_path is None:
            cv2.imwrite(os.path.join(out_dir, f), img)
            continue

        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        if i > 0:
            prev = cv2.imread(os.path.join(rgb_dir, frames[i-1]))
            prev_depth = find_depth(frames[i-1])
            if prev_depth:
                prev_depth = cv2.imread(prev_depth, cv2.IMREAD_UNCHANGED)
                prev_ts, _ = find_closest_timestamp(frames[i-1].split('.')[0], pose_ts)
                prev_pose = poses[prev_ts]
                img = reproject_frame(prev, prev_depth, img, prev_pose, pose)

        cv2.imwrite(os.path.join(out_dir, f), img)

    print(f"[DONE] {sequence_name}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        seqs = ["sitting_xyz", "walking_static", "walking_xyz"]
    elif args.sequence:
        seqs = [args.sequence]
    else:
        print("Provide --sequence or --all")
        sys.exit(1)

    for s in seqs:
        print(f"Processing {s}")
        process_sequence(s)


if __name__ == "__main__":
    main()
