#!/bin/bash

module purge
module load gcc/12.2.0 cmake eigen opencv cuda/11.3 vtk

export LD_LIBRARY_PATH=/share/pkg.7/gcc/12.2.0/install/lib64:$LD_LIBRARY_PATH

BASE_PATH=/projectnb/cs585/projects/dynamic_slam
SLAM_PATH=$BASE_PATH/ORB_SLAM3/Examples/RGB-D

DATASET=$1
MODE=$2   # baseline OR masked OR reprojection

# DATASET NAME

if [ "$DATASET" == "sitting_xyz" ]; then
    NAME=rgbd_dataset_freiburg3_sitting_xyz
elif [ "$DATASET" == "walking_static" ]; then
    NAME=rgbd_dataset_freiburg3_walking_static
elif [ "$DATASET" == "walking_xyz" ]; then
    NAME=rgbd_dataset_freiburg3_walking_xyz
else
    echo "Invalid dataset"
    exit 1
fi

# MODE PATH

if [ "$MODE" == "baseline" ]; then
    DATA_PATH=$BASE_PATH/dataset/tum_rgbd/$NAME

elif [ "$MODE" == "masked" ]; then
    DATA_PATH=$BASE_PATH/Ghost-free-slam/masking/masked_frames/$NAME

elif [ "$MODE" == "reprojection" ]; then
    DATA_PATH=$BASE_PATH/Ghost-free-slam/reprojection/recovered_frames/$NAME

else
    echo "Invalid mode"
    echo "Use: baseline OR masked OR reprojection"
    exit 1
fi

echo "Running SLAM on: $DATA_PATH"
echo "Mode: $MODE"

# ASSOCIATION FILE

if [ ! -f "$DATA_PATH/associations.txt" ]; then
    echo "Creating associations.txt..."
    cd $DATA_PATH
    python $BASE_PATH/ORB_SLAM3/Examples/RGB-D/associate.py rgb.txt depth.txt > associations.txt
fi

# RUN SLAM


cd $SLAM_PATH

./rgbd_tum ../../Vocabulary/ORBvoc.txt TUM1.yaml $DATA_PATH $DATA_PATH/associations.txt

# SAVE TRAJECTORIES

CAM_DIR=$BASE_PATH/trajectories/$MODE/camera_trajectories
KEY_DIR=$BASE_PATH/trajectories/$MODE/keyframe_trajectories

mkdir -p $CAM_DIR
mkdir -p $KEY_DIR

# Safety check
if [ -f "CameraTrajectory.txt" ]; then
    mv CameraTrajectory.txt $CAM_DIR/${MODE}_camera_${DATASET}.txt
else
    echo "[ERROR] CameraTrajectory.txt not found"
fi

if [ -f "KeyFrameTrajectory.txt" ]; then
    mv KeyFrameTrajectory.txt $KEY_DIR/${MODE}_keyframe_${DATASET}.txt
else
    echo "[ERROR] KeyFrameTrajectory.txt not found"
fi

echo "Saved camera + keyframe for $DATASET ($MODE)"
