#!/bin/bash

#$-S /bin/bash
#$-cwd            # running from tha current directory
#$-V              # environment variables
#$-l gpus=1       # req 1 GPU
#$-l gpu_c=7.5   # GPU capability (optional)

# assumes you have loaded into a conda environment and installed the required dependnecies
# conda env named "yolo"
# source activate yolo --> I ended up not using this env anymore

# run the script
python segmentation/run_instance_segmentation.py