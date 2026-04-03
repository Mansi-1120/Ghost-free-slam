#!/bin/bash



module purge

module load gcc/12.2.0 cmake eigen opencv cuda/11.3 vtk

export LD_LIBRARY_PATH=/share/pkg.7/gcc/12.2.0/install/lib64:$LD_LIBRARY_PATH



BASE_PATH=/projectnb/cs585/projects/dynamic_slam

SLAM_PATH=$BASE_PATH/ORB_SLAM3/Examples/RGB-D



DATASET=$1



if [ "$DATASET" == "sitting_xyz" ]; then

    DATA_PATH=$BASE_PATH/dataset/tum_rgbd/rgbd_dataset_freiburg3_sitting_xyz

    OUTPUT=baseline_sitting_xyz.txt

elif [ "$DATASET" == "walking_static" ]; then

    DATA_PATH=$BASE_PATH/dataset/tum_rgbd/rgbd_dataset_freiburg3_walking_static

    OUTPUT=baseline_walking_static.txt

elif [ "$DATASET" == "walking_xyz" ]; then

    DATA_PATH=$BASE_PATH/dataset/tum_rgbd/rgbd_dataset_freiburg3_walking_xyz

    OUTPUT=baseline_walking_xyz.txt

else

    echo "Invalid dataset"

    echo "Usage: ./run_slam.sh [sitting_xyz | walking_static | walking_xyz]"

    exit 1

fi



if [ ! -f "$DATA_PATH/associations.txt" ]; then

    echo "Creating associations.txt..."

    cd $DATA_PATH

    python $BASE_PATH/ORB_SLAM3/Examples/RGB-D/associate.py rgb.txt depth.txt > associations.txt

fi



cd $SLAM_PATH



# Kept on one line to ensure no trailing space issues break the command

./rgbd_tum ../../Vocabulary/ORBvoc.txt TUM1.yaml $DATA_PATH $DATA_PATH/associations.txt



mv CameraTrajectory.txt $BASE_PATH/trajectories/$OUTPUT



echo "Saved: $OUTPUT"
