#!/bin/bash
# ==============================================================================
# run_slam.sh — One-command ORB-SLAM3 runner
# Author: Tianqin (based on Mansi's SLAM setup)
# Project: CS585 — Dynamic SLAM
#
# Usage (run from project root):
#   cd /projectnb/cs585/projects/dynamic_slam
#   bash slam/run_slam.sh
#   bash slam/run_slam.sh --sequence rgbd_dataset_freiburg3_walking_static
#   bash slam/run_slam.sh --sequence rgbd_dataset_freiburg3_sitting_xyz
#
# Available sequences:
#   rgbd_dataset_freiburg3_sitting_xyz
#   rgbd_dataset_freiburg3_walking_static
#   rgbd_dataset_freiburg3_walking_xyz
#
# Prerequisites:
#   - ORB-SLAM3 built at $ORBSLAM3_ROOT
#   - TUM RGB-D dataset in dataset/tum_rgbd/
#   - OpenCV, Eigen, Pangolin installed (see ORB-SLAM3 README)
# ==============================================================================

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — Paths specific to our SCC project
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT="/projectnb/cs585/projects/dynamic_slam"

# ORB-SLAM3 install location
# TODO: Confirm with Mansi where ORB-SLAM3 is built on SCC.
#       If it's in "trained models/":
ORBSLAM3_ROOT="${ORBSLAM3_ROOT:-${PROJECT_ROOT}/trained models/ORB_SLAM3}"
#       If Mansi built it elsewhere (e.g. her home dir), update this path.

DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset/tum_rgbd}"
VOCAB_FILE="${ORBSLAM3_ROOT}/Vocabulary/ORBvoc.txt"

# Default sequence — change this or pass --sequence flag
# Options: rgbd_dataset_freiburg3_sitting_xyz
#          rgbd_dataset_freiburg3_walking_static
#          rgbd_dataset_freiburg3_walking_xyz
SEQUENCE="${SEQUENCE:-rgbd_dataset_freiburg3_walking_xyz}"

MODE="${MODE:-rgbd}"

# freiburg3 sequences use TUM3.yaml (fr1 → TUM1.yaml, fr2 → TUM2.yaml)
SETTINGS_FILE="${ORBSLAM3_ROOT}/Examples/RGB-D/TUM3.yaml"

# Output paths
OUTPUT_DIR="${PROJECT_ROOT}/trajectories"
OUTPUT_FILE="${OUTPUT_DIR}/baseline.txt"
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/slam_run.txt"

# ──────────────────────────────────────────────────────────────────────────────
# PARSE ARGUMENTS
# ──────────────────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --sequence)   SEQUENCE="$2";       shift 2 ;;
        --mode)       MODE="$2";           shift 2 ;;
        --settings)   SETTINGS_FILE="$2";  shift 2 ;;
        --vocab)      VOCAB_FILE="$2";     shift 2 ;;
        --dataset)    DATASET_ROOT="$2";   shift 2 ;;
        --orbslam)    ORBSLAM3_ROOT="$2";  shift 2 ;;
        --output)     OUTPUT_FILE="$2";    shift 2 ;;
        -h|--help)
            echo "Usage: bash slam/run_slam.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --sequence   Sequence folder name         (default: rgbd_dataset_freiburg3_walking_xyz)"
            echo "  --mode       SLAM mode: rgbd|stereo|mono  (default: rgbd)"
            echo "  --settings   Path to .yaml config         (default: TUM3.yaml)"
            echo "  --vocab      Path to ORB vocabulary"
            echo "  --dataset    Dataset root directory"
            echo "  --orbslam    ORB-SLAM3 root directory"
            echo "  --output     Output trajectory file path"
            echo ""
            echo "Available sequences:"
            echo "  rgbd_dataset_freiburg3_sitting_xyz"
            echo "  rgbd_dataset_freiburg3_walking_static"
            echo "  rgbd_dataset_freiburg3_walking_xyz"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ──────────────────────────────────────────────────────────────────────────────
# PRE-FLIGHT CHECKS
# ──────────────────────────────────────────────────────────────────────────────
echo "============================================"
echo "  ORB-SLAM3 Runner — CS585 Dynamic SLAM"
echo "============================================"
echo ""

errors=0

if [ ! -f "$VOCAB_FILE" ]; then
    echo "[ERROR] Vocabulary file not found: $VOCAB_FILE"
    echo "        Is ORB-SLAM3 installed at: $ORBSLAM3_ROOT ?"
    echo "        → Ask Mansi for the correct ORB-SLAM3 path on SCC"
    errors=1
fi

if [ ! -f "$SETTINGS_FILE" ]; then
    echo "[ERROR] Settings file not found: $SETTINGS_FILE"
    errors=1
fi

SEQUENCE_PATH="${DATASET_ROOT}/${SEQUENCE}"
if [ ! -d "$SEQUENCE_PATH" ]; then
    echo "[ERROR] Sequence directory not found: $SEQUENCE_PATH"
    echo ""
    echo "        Available sequences in ${DATASET_ROOT}:"
    ls -1 "$DATASET_ROOT" 2>/dev/null || echo "        (could not list directory)"
    errors=1
fi

# Determine the SLAM executable based on mode
case $MODE in
    rgbd)
        SLAM_EXEC="${ORBSLAM3_ROOT}/Examples/RGB-D/rgbd_tum"
        ASSOC_FILE="${SEQUENCE_PATH}/associations.txt"
        if [ ! -f "$ASSOC_FILE" ]; then
            echo "[WARN]  associations.txt not found at: $ASSOC_FILE"
            echo "        TUM RGB-D sequences need this file."
            echo "        Generate it with: python associate.py rgb.txt depth.txt > associations.txt"
            echo "        (associate.py is from the TUM RGB-D tools)"
        fi
        ;;
    stereo)
        SLAM_EXEC="${ORBSLAM3_ROOT}/Examples/Stereo/stereo_euroc"
        ;;
    mono)
        SLAM_EXEC="${ORBSLAM3_ROOT}/Examples/Monocular/mono_tum"
        ;;
    *)
        echo "[ERROR] Unknown mode: $MODE (use rgbd, stereo, or mono)"
        exit 1
        ;;
esac

if [ ! -f "$SLAM_EXEC" ]; then
    echo "[ERROR] SLAM executable not found: $SLAM_EXEC"
    echo "        Have you built ORB-SLAM3?"
    echo "        → cd \"$ORBSLAM3_ROOT\" && ./build.sh"
    errors=1
fi

if [ "$errors" -eq 1 ]; then
    echo ""
    echo "[ABORT] Fix the errors above and re-run."
    exit 1
fi

# ──────────────────────────────────────────────────────────────────────────────
# PREPARE OUTPUT DIRECTORIES
# ──────────────────────────────────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# ──────────────────────────────────────────────────────────────────────────────
# RUN ORB-SLAM3
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "Project:    $PROJECT_ROOT"
echo "Mode:       $MODE"
echo "Sequence:   $SEQUENCE"
echo "Full path:  $SEQUENCE_PATH"
echo "Settings:   $SETTINGS_FILE"
echo "Output:     $OUTPUT_FILE"
echo ""
echo "Starting ORB-SLAM3..."
echo "--------------------------------------------"

START_TIME=$(date +%s)
START_TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# ---- The actual SLAM command ----
if [ "$MODE" = "rgbd" ]; then
    "$SLAM_EXEC" \
        "$VOCAB_FILE" \
        "$SETTINGS_FILE" \
        "$SEQUENCE_PATH" \
        "$ASSOC_FILE" \
        "$OUTPUT_FILE"
elif [ "$MODE" = "stereo" ]; then
    "$SLAM_EXEC" \
        "$VOCAB_FILE" \
        "$SETTINGS_FILE" \
        "$SEQUENCE_PATH" \
        "$OUTPUT_FILE"
elif [ "$MODE" = "mono" ]; then
    "$SLAM_EXEC" \
        "$VOCAB_FILE" \
        "$SETTINGS_FILE" \
        "$SEQUENCE_PATH" \
        "$OUTPUT_FILE"
fi

END_TIME=$(date +%s)
END_TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
ELAPSED=$(( END_TIME - START_TIME ))

echo ""
echo "--------------------------------------------"
echo "ORB-SLAM3 finished in ${ELAPSED}s"
echo "Trajectory saved to: $OUTPUT_FILE"

# ──────────────────────────────────────────────────────────────────────────────
# WRITE LOG → logs/slam_run.txt
# ──────────────────────────────────────────────────────────────────────────────
NUM_FRAMES="N/A"
if [ -f "$OUTPUT_FILE" ]; then
    NUM_FRAMES=$(wc -l < "$OUTPUT_FILE")
fi

cat > "$LOG_FILE" <<EOF
============================================
  SLAM Run Log — CS585 Dynamic SLAM
============================================

Run Info
--------
  Date started :  $START_TIMESTAMP
  Date ended   :  $END_TIMESTAMP
  Duration     :  ${ELAPSED} seconds
  Runner       :  $(whoami)@$(hostname)

Project
-------
  Project root :  $PROJECT_ROOT
  SCC path     :  /projectnb/cs585/projects/dynamic_slam/

Configuration
-------------
  Mode         :  $MODE
  ORB-SLAM3    :  $ORBSLAM3_ROOT
  Executable   :  $SLAM_EXEC
  Vocabulary   :  $VOCAB_FILE
  Settings     :  $SETTINGS_FILE
  Dataset root :  $DATASET_ROOT
  Sequence     :  $SEQUENCE
  Sequence path:  $SEQUENCE_PATH

Output
------
  Trajectory   :  $OUTPUT_FILE
  Frames in trajectory: $NUM_FRAMES

Command Executed
----------------
  $SLAM_EXEC \\
      $VOCAB_FILE \\
      $SETTINGS_FILE \\
      $SEQUENCE_PATH \\
      $([ "$MODE" = "rgbd" ] && echo "$ASSOC_FILE \\" || echo "\\")
      $OUTPUT_FILE

Notes
-----
  - Dataset: TUM RGB-D (freiburg3 sequences)
  - Trajectory format: TUM (timestamp tx ty tz qx qy qz qw)
  - Settings: TUM3.yaml (freiburg3 camera intrinsics)
  - Pass trajectories/baseline.txt to Brendan for EVO evaluation
  - Ground truth comparison output: results/baseline/
============================================
EOF

echo "Log written to: $LOG_FILE"
echo ""
echo "Next step: pass $OUTPUT_FILE to Brendan for evaluation."
