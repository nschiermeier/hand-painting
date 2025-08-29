import cv2
import numpy as np

def load_and_process(path, size=(128, 128)):
  # In case I decide to add more colors, this could be useful
  color = cv2.imread(path)
  return cv2.resize(color, size)

def create_top_overlay(video_source):
  
  color_names = ["red", "blue", "yellow", "green", "white", "black"]
  color_list = [load_and_process(f"data/{color}.png") for color in color_names] 

  frame_h, frame_w, frame_c = video_source.shape
  color_w, color_h, color_c = color_list[0].shape # should all be the same, so I can just take first element
  edge_size = 10
  banner_h = int(color_h+edge_size*2)

  # Load banner along with overlay that only captures the banner's region,
  # rather than the full image
  banner = cv2.resize(cv2.imread("data/banner.png"), (frame_w, banner_h))
  top_overlay = np.zeros((banner_h, frame_w, frame_c), dtype=np.uint8)
  top_overlay[:] = banner
  
  space = (frame_w - edge_size*2) / len(color_list) # dynamically create the 
                                                    # amount of space needed
                                                    # per color in case more added
  color_loc_pair = []
  for i, color in enumerate(color_list):
    # place colors on the overlay by incrementing the starting and ending positions
    start_x = int(space*i)
    end_x = start_x + color_w
    i += 1
    esi = int(edge_size*i)
    top_overlay[edge_size:edge_size+color_h, esi+start_x:esi+end_x] = color
    # Change color to be same channels at video_source for selection
    color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    color_loc_pair.append((color, (edge_size, edge_size+color_h, esi+start_x, esi+end_x)))
  return (top_overlay, banner_h, color_loc_pair)

def create_bottom_overlay(video_source):
  
  frame_h, frame_w, frame_c = video_source.shape
  
  banner_h = 120 # Make the bottom banner a little shorter than top banner
  banner = cv2.resize(cv2.imread("data/banner.png"), (frame_w, banner_h))
  bottom_overlay = np.zeros((frame_h, frame_w, frame_c), dtype=np.uint8)
  bottom_overlay[frame_h - banner_h:frame_h, 0:frame_w] = banner

  bottom_overlay[607:715, 300:408] = load_and_process("data/eraser.png", (108, 108))

  radii = [10, 20, 30, 50]
  edge_gap = 30
  y = frame_h - banner_h + int(banner_h/2)

  x = 840 + radii[0]
  for i, r in enumerate(radii):
    cv2.circle(bottom_overlay, (x, y), r, (0, 0, 1), -1)
    if i < len(radii) - 1:
      # Move to the next center with formula curr center + curr radius + hgap + next radius
      x += r + edge_gap + radii[i+1]
    
  tool_loc_pair = None

  return (bottom_overlay, banner_h, tool_loc_pair)

def paint(video_source, color, index_x, index_y, radius=15):
  # Just create an overlay and add splotches of the color to that overlay
  if type(color) is not tuple:
    color_tuple = tuple(int(c) for c in cv2.mean(color)[:3])
  else:
    color_tuple = color
  # To persistantly add to drawing, use circle method to draw
  cv2.circle(video_source,
    center=(int(index_x), int(index_y)),
    radius=radius, # could eventually be a parameter?
    color =color_tuple, thickness=-1)
  return video_source

def blend(video_source, new_color, curr_color, index_x, index_y):
  # If a color is already painted, blend them together!
  # Maybe average the two RGB vals?
  color_tuple = tuple(int(c) for c in cv2.mean(new_color)[:3])
  curr_color = tuple(curr_color)

  blended_color = tuple(
                      int(curr_color[i] * 0.97 + color_tuple[i] * 0.03)
                      for i in range(3)
  )

  return paint(video_source, blended_color, index_x, index_y)
