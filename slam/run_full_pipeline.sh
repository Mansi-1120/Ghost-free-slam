#!/bin/bash

# ==============================================================================

# run_full_pipeline.sh — Masked SLAM pipeline for all three freiburg3 sequences

# Runs ORB-SLAM3 on masked RGB frames (with original depth) and outputs:

#   trajectories/masked_sitting_xyz.txt

#   trajectories/masked_walking_static.txt

#   trajectories/masked_walking_xyz.txt

# ==============================================================================



module purge

module load gcc/12.2.0 cmake eigen opencv cuda/11.3 vtk

export LD_LIBRARY_PATH=/share/pkg.7/gcc/12.2.0/install/lib64:$LD_LIBRARY_PATH



BASE_PATH=/projectnb/cs585/projects/dynamic_slam

SLAM_PATH=$BASE_PATH/ORB_SLAM3/Examples/RGB-D
VOCAB=$BASE_PATH/ORB_SLAM3/Vocabulary/ORBvoc.txt

REPO_ROOT=/projectnb/cs585/students/finnfu/clone2/Ghost-free-slam

MASKED_ROOT=$REPO_ROOT/masking/masked_frames

WORK_ROOT=$REPO_ROOT/masked_input     # temp dir for symlinked rgb+depth



TRAJ_DIR=$REPO_ROOT/trajectories

LOG_FILE=$REPO_ROOT/logs/full_pipeline.txt
# Make a local copy of TUM1.yaml with the viewer disabled (headless runs)

LOCAL_YAML=$REPO_ROOT/slam/TUM1_headless.yaml

if [ ! -f "$LOCAL_YAML" ]; then

    cp "$SLAM_PATH/TUM1.yaml" "$LOCAL_YAML"

    # Toggle the viewer off 

    echo "" >> "$LOCAL_YAML"

    echo "Viewer.on: 0" >> "$LOCAL_YAML"

fi


mkdir -p "$TRAJ_DIR" "$(dirname "$LOG_FILE")" "$WORK_ROOT"

echo "=== Full pipeline run: $(date) ===" > "$LOG_FILE"



# Loop over all three sequences

for DATASET in sitting_xyz walking_static walking_xyz; do



    FULL_SEQ=rgbd_dataset_freiburg3_$DATASET

    ORIG_DATA=$BASE_PATH/dataset/tum_rgbd/$FULL_SEQ

    MASKED_RGB=$MASKED_ROOT/$FULL_SEQ

    WORK_DIR=$WORK_ROOT/$FULL_SEQ

    OUTPUT=masked_${DATASET}.txt



    echo ""

    echo "=============================================="

    echo "  Processing: $DATASET"

    echo "=============================================="



    # Build working directory: masked rgb + original depth side-by-side

    rm -rf "$WORK_DIR"

    mkdir -p "$WORK_DIR"

    ln -sf "$MASKED_RGB"      "$WORK_DIR/rgb"

    ln -sf "$ORIG_DATA/depth" "$WORK_DIR/depth"

    cp "$ORIG_DATA/depth.txt" "$WORK_DIR/depth.txt"



    # Build new rgb.txt pointing at masked frames

    python - "$WORK_DIR" <<'EOF'

import os, sys

work_dir = sys.argv[1]

rgb_dir = os.path.join(work_dir, "rgb")

with open(os.path.join(work_dir, "rgb.txt"), "w") as f:

    f.write("# masked rgb frames\n# timestamp filename\n")

    for fn in sorted(os.listdir(rgb_dir)):

        if fn.endswith(".png"):

            ts = os.path.splitext(fn)[0]

            f.write(f"{ts} rgb/{fn}\n")

EOF



    # Generate associations.txt using absolute paths

   python "$BASE_PATH/ORB_SLAM3/Examples/RGB-D/associate.py" "$WORK_DIR/rgb.txt" "$WORK_DIR/depth.txt" > "$WORK_DIR/associations.txt"


    # Run SLAM

    cd "$WORK_DIR"

    START=$(date +%s) 
    xvfb-run -a "$SLAM_PATH/rgbd_tum" "$VOCAB" "$LOCAL_YAML" "$WORK_DIR" "$WORK_DIR/associations.txt"
    ELAPSED=$(( $(date +%s) - START ))



    mv "$WORK_DIR/CameraTrajectory.txt" "$TRAJ_DIR/$OUTPUT"

    FRAMES=$(wc -l < $TRAJ_DIR/$OUTPUT)



    echo "Saved: $OUTPUT  ($FRAMES frames, ${ELAPSED}s)" | tee -a "$LOG_FILE"

done



echo ""

echo "All three sequences done. Trajectories are in $TRAJ_DIR/"
