import cv2
import numpy as np


def load_and_process(path, size=(128, 128)):
  # In case I decide to add more colors, this could be useful
  color = cv2.imread(path)
  return cv2.resize(color, size)

def create_overlay(video_source):
  
  red = load_and_process("data/red.png")
  blue = load_and_process("data/blue.png")
  yellow = load_and_process("data/yellow.png")

  frame_h, frame_w, frame_c = video_source.shape
  color_w, color_h, color_c = red.shape # should all be the same, so I can just take red
  color_list = [red, blue, yellow]
  edge_size = 10
  alpha = 0.4
  banner = cv2.resize(cv2.imread("data/banner.png"), (frame_w, int(color_h+edge_size*2)))
  overlay = video_source.copy()
  overlay[0:int(color_h+edge_size*2), 0:frame_w] = banner
  
  space = (frame_w - edge_size*2) / len(color_list) # create the amount of space needed per color
  for i, color in enumerate(color_list):
    # place colors on the overlay by incrementing the starting and ending positions
    #start_x = color_w * i
    start_x = int(space*i)
    end_x = start_x + color_w
    i += 1
    esi = int(edge_size*i)
    #overlay[edge_size:edge_size+color_h, esi+start_x:esi+end_x] = color
    overlay[edge_size:edge_size+color_h, esi+start_x:esi+end_x] = color
  video_source = cv2.addWeighted(overlay, alpha, video_source, 1-alpha, 0) 
  return video_source
