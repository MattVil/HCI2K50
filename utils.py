import cv2
import numpy as np
import mediapipe as mp

class Window:
  def __init__(self, pos, size):
    self.pos = pos
    self.size = size
  
  def draw(self, img):
    h, w = self.size
    x, y = self.pos
    sub_img = img[y:y+h, x:x+w]
    white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
    res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
    img[y:y+h, x:x+w] = res
    
  def in_rectangle(self, point):
    return (self.pos[0] < point[0] < self.pos[0]+self.size[0]) and (self.pos[1] < point[1] < self.pos[1]+self.size[1])
  
  def set_pos(self, pos):
    self.pos[0] = pos[0] - self.size[0]//2
    self.pos[1] = pos[1] - self.size[1]//2
    
class Drawer():
  def __init__(self, holistic_face_mesh, holistic_hand_connections, img_size):
    self.mp_drawing = mp.solutions.drawing_utils
    self.face_mesh = holistic_face_mesh
    self.hand_connections = holistic_hand_connections
    self.img_size = img_size # [H, W, C]
    
  def draw_mp(self, image, results, face=False, right_hand=True, left_hand=True):
    if face:
      self.mp_drawing.draw_landmarks(
          image,
          results.face_landmarks,
          self.face_mesh,
          landmark_drawing_spec=None,
          connection_drawing_spec=self.mp_drawing.DrawingSpec(
            color=(237, 247, 203),
            thickness=1,
            circle_radius=1
          )
      )
    if right_hand:
      self.mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        self.hand_connections,
        landmark_drawing_spec=self.mp_drawing.DrawingSpec(
          color=(91, 57, 35),
          thickness=2,
          circle_radius=3
        ),
        connection_drawing_spec=self.mp_drawing.DrawingSpec(
          color=(237, 247, 203),
          thickness=1,
          circle_radius=1
        )
      )
    if left_hand:
      self.mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        self.hand_connections,
        landmark_drawing_spec=self.mp_drawing.DrawingSpec(
          color=(91, 57, 35),
          thickness=2,
          circle_radius=3
        ),
        connection_drawing_spec=self.mp_drawing.DrawingSpec(
          color=(237, 247, 203),
          thickness=1,
          circle_radius=1
        )
      )
  def draw_points(self, image, detections, idxs, color):
    if detections:
      for idx in idxs:
        x = int(self.img_size[1]*detections[idx].x)
        y = int(self.img_size[0]*detections[idx].y)
        cv2.circle(image, (x, y) , 5, color, -1)
  
  def draw_line(self, image, detections, idxs, color):
    if detections:
      x0 = int(self.img_size[1]*detections[idxs[0]].x)
      y0 = int(self.img_size[0]*detections[idxs[0]].y)
      x1 = int(self.img_size[1]*detections[idxs[1]].x)
      y1 = int(self.img_size[0]*detections[idxs[1]].y)
      cv2.line(image, (x0,y0), (x1,y1), color, 2)
      
  def draw_debug(self, image, results):
    pass