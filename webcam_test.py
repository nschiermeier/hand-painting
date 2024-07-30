from webcam import Webcam
import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from os.path import abspath

# Define a simple webcam object that will get video stream from webcam (src=0),
#  with a frame width of 640 (auto setting heigth to keep original aspect ratio)
webcam = Webcam(src=0, w=640)
print(f"Frame size: {webcam.w} x {webcam.h}")
mp_hands = mp.solutions.hands.Hands()

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

  # Draw the pose landmarks
  pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
  pose_landmarks_proto.landmark.extend([
    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
  ])

  mp.solutions.drawing_utils.draw_landmarks(
    annotated_image,
    pose_landmarks_proto,
    mp.solutions.pose.POSE_CONNECTIONS,
    mp.solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

# Set up PoseLandmarker object
model_path = r'C:/Users/Nick/Projects/mirror-the-mask/pose_landmarker.task'
base_options = python.BaseOptions(model_asset_buffer=open(model_path, 'rb').read()) # Get standard set of options for a posing task
#close(model_path)
options = vision.PoseLandmarkerOptions(
  base_options = base_options,
  output_segmentation_masks=True) # get an options object using specified base settings previously created
detector = vision.PoseLandmarker.create_from_options(options) # Get a detector object with our options for this task

for frame in webcam:
    # Show the frames in a cv2 window
    frame_as_img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #cv2.imshow('Webcam Frame', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    # Determine if a hand is on screen or not
    """results = mp_hands.process(frame)
    if results.multi_hand_landmarks:
      print("Successfully found hand on screen!")
    else:
      print("No hand was detected")
    """

    # Load input image/frame for detector
    # Detect pose landmarks from current frame
    rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_as_img)
    detection_result = detector.detect(rgb_frame)

    # Process the detection result, then display result
    annotated_image = draw_landmarks_on_image(rgb_frame.numpy_view(), detection_result)
    cv2.imshow('w', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    # Break the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
