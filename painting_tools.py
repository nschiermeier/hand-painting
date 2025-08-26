import cv2
import numpy as np


def load_and_process(path, size=(128, 128)):
  # In case I decide to add more colors, this could be useful
  color = cv2.imread(path)
  #color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
  return cv2.resize(color, size)

def create_overlay(video_source):
  
  red = load_and_process("data/red.png")
  blue = load_and_process("data/blue.png")
  yellow = load_and_process("data/yellow.png")

  frame_h, frame_w, frame_c = video_source.shape
  edge_size = 10
  color_list = [red, blue, yellow]
  
  space = (frame_w - edge_size*2) / len(color_list) # create the amount of space needed per color
  color_w, color_h, color_c = red.shape # should all be the same, so I can just take red
  for i, color in enumerate(color_list):
    start_x = color_w * i
    end_x = start_x + color_w
    i += 1
    esi = int(edge_size*i)
    video_source[edge_size:edge_size+color_h, esi+start_x:esi+end_x] = color
  return video_source
