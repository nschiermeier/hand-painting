import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# mp hands with options changed to work better with video stream
mp_hands = mp.solutions.hands.Hands(
            static_image_mode = False,
            max_num_hands = 1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

def draw_hand_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]

  # Draw the hand landmarks
  hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
  try: # Throw this section into try/catch so program doesn't crash when no hands on screen
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])

    mp.solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      mp.solutions.hands.HAND_CONNECTIONS,
      mp.solutions.drawing_styles.get_default_hand_landmarks_style())
  except (UnboundLocalError, ValueError) as e:
    pass
  return annotated_image


hand_path = r'C:/Users/Nick/Projects/mirror-the-mask/hand_landmarker.task'
base_hand_options = python.BaseOptions(model_asset_buffer=open(hand_path, 'rb').read()) # Open hand path for finger tracking

hand_options = vision.HandLandmarkerOptions( # options object using specified base settings
  base_options = base_hand_options)
hand_detector = vision.HandLandmarker.create_from_options(hand_options) # Detector object for displaying annotated hand

#TODO: Put this in a function and create a main method
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print(f"Frame size: {capture.get(cv2.CAP_PROP_FRAME_WIDTH)} x {capture.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
while capture.isOpened():
 
  # Grab next image in video stream, ret is False if webcam has issues (i.e. disconnects)
  ret, frame = capture.read()
  if not ret:
    break

  # Show the frames in a cv2 window
  frame_as_img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
  # Determine if a hand is on screen or not
  results = mp_hands.process(frame)
  if results.multi_hand_landmarks: # Only execute this if a hand is detected in webcam
    for hand_landmark in results.multi_hand_landmarks:
      #TODO: Need to find a way to detect if one or two fingers are raised
      #      (Even better if I can detect which of the two fingers those are...)
      #      Wondering if this would be better suited for the detector rather than landmarker?
      index_tip_x  = hand_landmark.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].x
      index_tip_y  = hand_landmark.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y

      middle_tip_x = hand_landmark.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP].x
      middle_tip_y = hand_landmark.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP].y
  # Load input image/frame for detector
  # Detect pose landmarks from current frame
  rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_as_img)
  hand_result = hand_detector.detect(rgb_frame)

  # Process the detection result, then display result
  annotated_hand = draw_hand_landmarks_on_image(rgb_frame.numpy_view(),  hand_result)
  cv2.imshow('Webcam Source', cv2.cvtColor(annotated_hand,  cv2.COLOR_RGB2BGR))

  # Break the loop if the user presses the 'q' key
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
