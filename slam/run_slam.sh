
#!/bin/bash

# Load modules
module purge
module load gcc/12.2.0 cmake eigen opencv cuda/11.3 vtk

export LD_LIBRARY_PATH=/share/pkg.7/gcc/12.2.0/install/lib64:$LD_LIBRARY_PATH

# Base paths
BASE_PATH=/projectnb/cs585/projects/dynamic_slam
SLAM_PATH=$BASE_PATH/ORB_SLAM3/Examples/RGB-D
MASK_PATH=$BASE_PATH/Ghost-free-slam/masking/masked_frames
TRAJ_PATH=$BASE_PATH/trajectories

# Input argument
DATASET=$1

# Select dataset
if [ "$DATASET" == "sitting_xyz" ]; then
    DATA_PATH=$MASK_PATH/rgbd_dataset_freiburg3_sitting_xyz
    ASSOC_PATH=$DATA_PATH/associations.txt
    OUTPUT=masked_camera_sitting_xyz.txt

elif [ "$DATASET" == "walking_static" ]; then
    DATA_PATH=$MASK_PATH/rgbd_dataset_freiburg3_walking_static
    ASSOC_PATH=$DATA_PATH/associations.txt
    OUTPUT=masked_camera_walking_static.txt

elif [ "$DATASET" == "walking_xyz" ]; then
    DATA_PATH=$MASK_PATH/rgbd_dataset_freiburg3_walking_xyz
    ASSOC_PATH=$DATA_PATH/associations.txt
    OUTPUT=masked_camera_walking_xyz.txt

else
    echo "Invalid dataset"
    echo "Usage: ./run_slam.sh [sitting_xyz | walking_static | walking_xyz]"
    exit 1
fi

# Go to SLAM directory
cd $SLAM_PATH

echo "Running masked SLAM on: $DATASET"

# Run ORB-SLAM3
./rgbd_tum ../../Vocabulary/ORBvoc.txt TUM1.yaml $DATA_PATH $ASSOC_PATH

# Save trajectory
mv CameraTrajectory.txt $TRAJ_PATH/$OUTPUT

echo "Saved trajectory: $TRAJ_PATH/$OUTPUT"
