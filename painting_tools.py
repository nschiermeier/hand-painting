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
  banner_h = int(color_h+edge_size*2)

  # Load banner along with overlay that only captures the banner's region,
  # rather than the full image
  banner = cv2.resize(cv2.imread("data/banner.png"), (frame_w, banner_h))
  overlay = np.zeros((banner_h, frame_w, 3), dtype=np.uint8)
  overlay[:] = banner
  
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
  return (overlay, banner_h, color_loc_pair)

def paint(video_source, color, index_x, index_y):
  # Just create an overlay and add splotches of the color to that overlay
  color_tuple = tuple(int(c) for c in cv2.mean(color)[:3])
  # To persistantly add to drawing, use rectangle method to draw
  # Maybe try switching this to circle at some point to see if that draws more smoothly?
  cv2.rectangle(video_source,
    (int(index_x-5), int(index_y-5)),
    (int(index_x+5), int(index_y+5)),
    color_tuple, -1)
  return video_source
