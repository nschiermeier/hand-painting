import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from painting_tools import create_top_overlay, create_bottom_overlay, paint, blend

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
 
hand_path = r'C:/Users/Nick/Projects/mirror-the-mask/data/hand_landmarker.task'
base_hand_options = python.BaseOptions(model_asset_buffer=open(hand_path, 'rb').read()) # Open hand path for finger tracking

hand_options = vision.HandLandmarkerOptions( # options object using specified base settings
  base_options = base_hand_options)
hand_detector = vision.HandLandmarker.create_from_options(hand_options) # Detector object for displaying annotated hand

#TODO: Put this in a function and create a main method
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print(f"Frame size: {capture.get(cv2.CAP_PROP_FRAME_WIDTH)} x {capture.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
curr_color = (10,10,10)
radius = 20
eraser = False
canvas = np.zeros((int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 3), dtype=np.uint8)
while capture.isOpened():
  # Grab next image in video stream, ret is False if webcam has issues (i.e. disconnects)
  ret, frame = capture.read()
  if not ret:
    break

  # Show the frames in a cv2 window
  frame = cv2.flip(frame, 1) # Set to "selfie mode" so it's easier to follow when drawing
  frame_as_img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
  frame_h, frame_w, _ = frame.shape
    
  # Determine if a hand is on screen or not
  results = mp_hands.process(frame_as_img)
 
  # Load input image/frame for detector
  # Detect pose landmarks from current frame
  rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_as_img)
  hand_result = hand_detector.detect(rgb_frame)

  # Process the detection result, then display result
  annotated_hand = draw_hand_landmarks_on_image(rgb_frame.numpy_view(),  hand_result)
  top_overlay_data = create_top_overlay(cv2.cvtColor(annotated_hand, cv2.COLOR_RGB2BGR))
  top_banner_overlay, banner_h, color_locs = top_overlay_data
  bottom_overlay_data = create_bottom_overlay(cv2.cvtColor(annotated_hand, cv2.COLOR_RGB2BGR))
  bottom_banner_overlay, bottom_banner_h, tool_locs = bottom_overlay_data
  alpha = 0.6

  if results.multi_hand_landmarks: # Only execute this if a hand is detected in webcam
    fingers, locations = detect_raised_fingers(results.multi_hand_landmarks)
    index, middle = fingers
    # 'locations' coords are from 0-1 rather than pixel values
    # To get pixel values, multiply by webcam resoltuion
    # Top left of screen (0,0), bottom right is (1280, 720)
    index_x = locations[0].x  * frame_w
    index_y = locations[0].y  * frame_h
    middle_x = locations[1].x * frame_w
    middle_y = locations[1].y * frame_h

    if index and middle: # Selection mode
      for color, (y_min, y_max, x_min, x_max) in color_locs:
        if y_min <= index_y <= y_max: # First see if fingers are in the selection bar 
          if x_min <= index_x <= x_max: # Next see specific coords of fingers.
                                        # If coords overlap with a colored square,
                                        # set to that color
            curr_color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            eraser = False

      for tool, (y_min, y_max, x_min, x_max) in tool_locs:
        if y_min <= index_y <= y_max:
          if x_min <= index_x <= x_max:
            if tool == 0:
              # Set color to true 0, which should erase
              curr_color = (0,0,0)
              eraser = True


            else:
              radius = tool
        
    elif index: # Drawing mode
      #TODO: Create something that doesn't allow user to draw on top banner
      if curr_color is not None:
        if any(canvas[int(index_y)][int(index_x)]) != 0 and eraser==False: 
          blend(canvas, curr_color, canvas[int(index_y)][int(index_x)], index_x, index_y, radius)
        else:
          paint(canvas, curr_color, index_x, index_y, radius)
      
    else: # Nothing, this might not be needed
      pass

  # Banner blending:
  # Create a base layer for the background and the mask
  # for the canvas that's drawn on in paint()
  base = cv2.cvtColor(annotated_hand, cv2.COLOR_RGB2BGR)
  mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
  _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

  # add the canvas drawings to the background while NOT including the masks
  # this fixes issue #12, because now there aren't multiple alpha layers
  # being multiplied onto eachother so it just pastes the drawings.
  bg = cv2.bitwise_and(base, base, mask=cv2.bitwise_not(mask))
  fg = cv2.bitwise_and(canvas, canvas, mask=mask)
  frame_with_canvas = cv2.add(bg, fg)

  # Add the banner and create an ROI to blend, so that this is
  # the ONLY region that the alpha gets appleid to, not the whole image
  roi = frame_with_canvas[0:banner_h, 0:frame_w]
  blended_roi = cv2.addWeighted(roi, 1.0 - alpha, top_banner_overlay, alpha, 0)
  frame_with_canvas[0:banner_h, 0:frame_w] = blended_roi

  # Repeat for bottom banner
  bottom_banner_overlay = bottom_banner_overlay[frame_h-bottom_banner_h:frame_h, 0:frame_w]
  roi = frame_with_canvas[frame_h-bottom_banner_h:frame_h, 0:frame_w]
  blended_roi = cv2.addWeighted(roi, 1.0 - alpha, bottom_banner_overlay, alpha, 0)
  frame_with_canvas[frame_h-bottom_banner_h:frame_h, 0:frame_w] = blended_roi

  cv2.circle(frame_with_canvas, (755, 660), radius, tuple(int(c) for c in cv2.mean(curr_color)[:3]), 1 if eraser else -1)
  cv2.putText(frame_with_canvas, "Current Selection:", (425, 630), cv2.FONT_HERSHEY_SIMPLEX, 1, (1, 1, 1), 2)
  cv2.putText(frame_with_canvas, "Press 'q' to Quit!", (10, 670),  cv2.FONT_HERSHEY_SIMPLEX, 1, (1, 1, 1), 2)

  cv2.imshow('Webcam Source', frame_with_canvas)

  # Break the loop if the user presses the 'q' key
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

capture.release()
cv2.destroyAllWindows()
