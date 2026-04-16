#!/bin/bash

#$-S /bin/bash
#$-cwd            # running from tha current directory
#$-V              # environment variables
#$-l gpus=1       # req 1 GPU
#$-l gpu_c=7.5   # GPU capability (optional)

module load python3/3.10.12

export PYTHONPATH=$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH

# run the script
python segmentation/run_instance_segmentation.py