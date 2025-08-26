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
  color_list = [red, blue, yellow]
  color_w, color_h, color_c = color_list[0].shape # should all be the same, so I can just take first element
  edge_size = 10
  alpha = 0.4
  banner = cv2.resize(cv2.imread("data/banner.png"), (frame_w, int(color_h+edge_size*2)))
  overlay = video_source.copy()
  overlay[0:int(color_h+edge_size*2), 0:frame_w] = banner
  
  space = (frame_w - edge_size*2) / len(color_list) # create the amount of space needed per color
  color_loc_pair = []
  for i, color in enumerate(color_list):
    # place colors on the overlay by incrementing the starting and ending positions
    start_x = int(space*i)
    end_x = start_x + color_w
    i += 1
    esi = int(edge_size*i)
    overlay[edge_size:edge_size+color_h, esi+start_x:esi+end_x] = color
    # Change color to be same channels at video_source for selection
    color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    color_loc_pair.append((color, (edge_size, edge_size+color_h, esi+start_x, esi+end_x)))
  video_source = cv2.addWeighted(overlay, alpha, video_source, 1-alpha, 0) 
  return (video_source, color_loc_pair)

def paint(video_source, color, index_x, index_y):
  # Just create an overlay and add splotches of the color to that overlay?
  if color is not None:
    color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    overlay = video_source.copy()
    overlay[int(index_y-5):int(index_y+5), int(index_x-5):int(index_x+5)] = cv2.resize(color, (10, 10)) # fill some color somehow?
    return cv2.addWeighted(overlay, 0.5, video_source, 0.5, 0)
  else:
    return video_source # if user launches program and tries to draw without selecting a color first
