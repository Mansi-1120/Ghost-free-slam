import torch
import cv2
from ultralytics import YOLO
import numpy as np
import os

# Before running this script, be sure to follow the documentation I put in the yolo25test requirements.txt file
# command that I used in terminal to run this file (assuming CD'd in main project directory)
# $ python3 segmentation/YOLO_testing/segmentation_test.py

# checks to make sure GPU is used rather than CPU
if torch.backends.mps.is_available():
    print("using GPU")
else:
    print("GPU unavailable")

output_directory = "segmentation/YOLO_testing/result_test_videos"
os.makedirs(output_directory, exist_ok = True)
output_path = os.path.join(output_directory, "devinn_webcam_segmentation_test.mp4")

# load in the model (specifically the segmentation one)
model = YOLO("yolo26n-seg.pt")

# these lines initialize the video capturing
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# test YOLO26 model + deletion of pixels by using device webcam
# -------------------------------------------------------------------------
try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run inference,stream=False means processing one frame at a time
        results = model.predict(source=frame, device='mps', classes=[0], verbose=False)
        result = results[0]
        
        frame_copy = frame.copy()

        # make sure peoepl are detected, then merge masks into a single layer
        if result.masks is not None:
            masks = result.masks.data
            combined_mask = torch.any(masks, dim=0).cpu().numpy().astype(np.uint8)

            # resizing the masking to overlay properly
            combined_mask_resized = cv2.resize(combined_mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)

            # match dims of deleted pixel masking
            frame_copy[combined_mask_resized > 0] = [0, 0, 0]

        # save result to output video mp4 file
        out.write(frame_copy)

        cv2.imshow("live feed test", frame_copy)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    
    # cleanup the stuff
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"test video saved to: {output_path} :):)):):):)")
# -------------------------------------------------------------------------