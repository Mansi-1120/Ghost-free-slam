import torch
import cv2
from ultralytics import YOLO

# command that I used in terminal to run this file (assuming CD'd in main project directory)
# python3 segmentation/YOLO_testing/YOLO_test.py

# checks to make sure GPU is used rather than CPU
if torch.backends.mps.is_available():
    print("using GPU")
else:
    print("GPU unavailable")

# load in the model (specifically the segmentation one)
model = YOLO("yolo26n-seg.pt")


# UNCOMMENT to test YOLO26 model by using your device webcam
# -------------------------------------------------------------------------
# results = model.predict(
#     source = "0",
#     device = "mps",
#     classes = [0],
#     show = True
# )

# for result in results:
    
#     # place frames around objects that are detected
#     annotated_frame = result.plot()

#     # show the live feed
#     cv2.imshow("live feed test", annotated_frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cv2.destroyAllWindows()
# -------------------------------------------------------------------------


# UNCOMMENT to test YOLO26 model on a video pulled from a URL
# -------------------------------------------------------------------------

# you need to grab the absolute path (different for each of us) 
# to the video in the test_videos folder
# basically, right click on "people_test.mp4", and select "copy path". Paste that here
vid_path = " REPLACE ME WITH PATH ^ "

results = model.predict(
    source = vid_path, 
    device = "mps", 
    stream = True, 
    classes = [0], # setting this to 0 means the only class that will be detected is "people"
    conf = 0.3
)

print("performing segmentation")

for result in results:
    annotated_frame = result.plot()

    cv2.imshow("URL test", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
# -------------------------------------------------------------------------