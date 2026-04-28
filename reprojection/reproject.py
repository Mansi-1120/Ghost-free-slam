"""
    python reprojection/reproject.py --sequence sitting_xyz
    python reprojection/reproject.py --sequence walking_static
    python reprojection/reproject.py --sequence walking_xyz
    python reprojection/reproject.py --all

Inputs:
    - Masked RGB frames:  masking/masked_frames/<sequence>/rgb/
    - Depth frames:       masking/masked_frames/<sequence>/depth/
    - Masked trajectory:  trajectories/masked/camera_trajectories/masked_camera_<sequence>.txt

Output: reprojection/recovered_frames/<sequence>/
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
        [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
    ])
    return R


def backproject_depth(depth_img):
    h, w = depth_img.shape

    u = np.arange(w)
    v = np.arange(h)
    u, v = np.meshgrid(u, v)

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

    T_curr_prev = np.linalg.inv(curr_pose) @ prev_pose
    R_rel = T_curr_prev[:3, :3]
    t_rel = T_curr_prev[:3, 3]

    pts = points_3d.reshape(-1, 3)
    valid_flat = valid_depth.reshape(-1)

    valid_idx = np.where(valid_flat)[0]
    if len(valid_idx) == 0:
        return recovered

    pts_valid = pts[valid_idx]
    pts_curr = (R_rel @ pts_valid.T).T + t_rel

    z_curr = pts_curr[:, 2]
    valid_z = z_curr > 0.01

    u_curr = (FX * pts_curr[:, 0] / z_curr + CX).astype(np.int32)
    v_curr = (FY * pts_curr[:, 1] / z_curr + CY).astype(np.int32)

    in_bounds = (u_curr >= 0) & (u_curr < w) & (v_curr >= 0) & (v_curr < h) & valid_z

    u_proj = u_curr[in_bounds]
    v_proj = v_curr[in_bounds]

    src_v, src_u = np.unravel_index(valid_idx[in_bounds], (h, w))

    is_masked = mask[v_proj, u_proj]

    fill_u = u_proj[is_masked]
    fill_v = v_proj[is_masked]
    fill_src_v = src_v[is_masked]
    fill_src_u = src_u[is_masked]
    fill_z = z_curr[in_bounds][is_masked]

    if len(fill_u) == 0:
        return recovered

    order = np.argsort(-fill_z)
    recovered[fill_v[order], fill_u[order]] = prev_rgb[fill_src_v[order], fill_src_u[order]]

    return recovered


def find_closest_timestamp(query_ts, pose_timestamps):
    query = float(query_ts)
    closest = None
    min_diff = float('inf')
    for ts in pose_timestamps:
        diff = abs(float(ts) - query)
        if diff < min_diff:
            min_diff = diff
            closest = ts
    return closest, min_diff


def process_sequence(repo_root, sequence_name, max_time_diff=0.05):

    full_seq = f"rgbd_dataset_freiburg3_{sequence_name}"

    rgb_dir = os.path.join(repo_root, "masking", "masked_frames", full_seq, "rgb")
    depth_dir = os.path.join(repo_root, "masking", "masked_frames", full_seq, "depth")

    # ✅ FIXED TRAJECTORY PATH
    traj_path = os.path.join(
        repo_root,
        "trajectories",
        "masked",
        "camera_trajectories",
        f"masked_camera_{sequence_name}.txt"
    )

    # fallback (optional)
    if not os.path.exists(traj_path):
        traj_path = os.path.join(repo_root, "trajectories", f"masked_{sequence_name}.txt")

    out_dir = os.path.join(repo_root, "reprojection", "recovered_frames", full_seq)

    for p, desc in [(rgb_dir, "masked RGB"), (depth_dir, "depth"), (traj_path, "trajectory")]:
        if not os.path.exists(p):
            print(f"[ERROR] {desc} not found: {p}")
            return False

    os.makedirs(out_dir, exist_ok=True)

    poses = parse_trajectory(traj_path)
    pose_timestamps = sorted(poses.keys())
    print(f"  Loaded {len(poses)} poses")

    frames = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png')])
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.png')])

    depth_ts_list = [(float(os.path.splitext(f)[0]), f) for f in depth_files]

    def find_depth_for_rgb(rgb_fname):
        rgb_ts = float(os.path.splitext(rgb_fname)[0])
        best = min(depth_ts_list, key=lambda x: abs(x[0] - rgb_ts))
        if abs(best[0] - rgb_ts) < max_time_diff:
            return os.path.join(depth_dir, best[1])
        return None

    for i, fname in enumerate(frames):
        ts = os.path.splitext(fname)[0]

        curr_masked = cv2.imread(os.path.join(rgb_dir, fname))
        if curr_masked is None:
            continue

        closest_ts, diff = find_closest_timestamp(ts, pose_timestamps)
        if diff > max_time_diff:
            cv2.imwrite(os.path.join(out_dir, fname), curr_masked)
            continue

        curr_pose = poses[closest_ts]

        depth_path = find_depth_for_rgb(fname)
        if depth_path is None:
            cv2.imwrite(os.path.join(out_dir, fname), curr_masked)
            continue

        curr_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        recovered = curr_masked.copy()

        if np.any(np.all(curr_masked == 0, axis=2)) and i > 0:
            for offset in [1, 2, 3, 5, 8]:
                idx = i - offset
                if idx < 0:
                    continue

                src = frames[idx]
                src_rgb = cv2.imread(os.path.join(rgb_dir, src))
                src_depth = cv2.imread(find_depth_for_rgb(src), cv2.IMREAD_UNCHANGED)

                src_ts, _ = find_closest_timestamp(os.path.splitext(src)[0], pose_timestamps)
                src_pose = poses[src_ts]

                recovered = reproject_frame(src_rgb, src_depth, recovered, src_pose, curr_pose)

        cv2.imwrite(os.path.join(out_dir, fname), recovered)

    print(f"Done: {sequence_name}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--repo", type=str)
    args = parser.parse_args()

    repo_root = args.repo if args.repo else str(Path(__file__).resolve().parent.parent)

    if args.all:
        sequences = ["sitting_xyz", "walking_static", "walking_xyz"]
    elif args.sequence:
        sequences = [args.sequence]
    else:
        print("Use --sequence or --all")
        sys.exit(1)

    for seq in sequences:
        print(f"\nReprojecting: {seq}")
        process_sequence(repo_root, seq)


if __name__ == "__main__":
    main()
