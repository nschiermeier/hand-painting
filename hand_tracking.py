import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from painting_tools import create_top_overlay, create_bottom_overlay, paint, blend

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

def detect_raised_fingers(handmarks): # handmarks = 'hand landmarks' portmanteau
  for hand_landmark in handmarks:
    # Get coordinates of index and middle fingers
    index_tip  = hand_landmark.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip  = hand_landmark.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP]

    middle_tip = hand_landmark.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = hand_landmark.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP]

    # Finger is "up" if tip.y < pip.y (Hand in fist has tip below pip)
    index_up  = index_tip.y < index_pip.y
    middle_up = middle_tip.y < middle_pip.y
  return ((index_up, middle_up), (index_tip, middle_tip))
 
