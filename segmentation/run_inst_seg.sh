#!/bin/bash

#$-S /bin/bash
#$-cwd            # running from tha current directory
#$-V              # environment variables
#$-l gpus=1       # req 1 GPU
#$-l gpu_c=7.5   # GPU capability (optional)

# load up Python and activate your environment
module load python3/3.10.12
source env/bin/activate

# run the script
python run_instance_segmentation.py