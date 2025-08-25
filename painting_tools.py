import cv2
import numpy as np

def create_overlay(video_source):
  
  color = cv2.imread("white_square.jpg")
  color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
  color = cv2.resize(color, (30, 30))
  
  frame_h, frame_w, frame_c = video_source.shape
  video_source[10:40, 10:40] = color
  return video_source
