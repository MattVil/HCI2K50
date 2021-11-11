########## TODO ##########
# - Create Detector class 
# - Create Hand class
# - define gesture and formulate click as gesture
# - simple model for click and hold_click


import math
import numpy as np
import cv2
import mediapipe as mp

from utils import Window, Drawer
from detector import PoseDetector

def main(camera=1):
  mp_drawing = mp.solutions.drawing_utils
  mp_drawing_styles = mp.solutions.drawing_styles
  
  cv2.namedWindow("HCI2K50", cv2.WND_PROP_FULLSCREEN)
  cv2.setWindowProperty("HCI2K50",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
  
  detector = PoseDetector()
  drawer = Drawer(
    detector.mp_holistic.FACEMESH_TESSELATION,
    detector.mp_holistic.HAND_CONNECTIONS,
    (480, 640, 3)
  )
  
  positions = [[50, 200], [150, 200], [300, 200]]  
  rectangles = [Window(pos, [100, 100]) for pos in positions]
  
  cap = cv2.VideoCapture(camera)
  with detector.holistic as holistic:
    while cap.isOpened():
      success, image = cap.read()
      H, W, C = image.shape
      if not success:
        print("Ignoring empty camera frame.")
        continue

      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results, _, hand_r, hand_l = detector.detect(image, holistic)
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      
      drawer.draw_mp(image, results)
      drawer.draw_points(image, hand_r, [4, 8], (0, 0, 255))
      drawer.draw_line(image, hand_r, [4, 8], (255,0,0))
      drawer.draw_points(image, hand_l, [4, 8], (0, 0, 255))
      drawer.draw_line(image, hand_l, [4, 8], (255,0,0))
      
      image = cv2.flip(image, 1)
      cv2.imshow('HCI2K50', image)
      if cv2.waitKey(5) & 0xFF == ord('q'):
        break
  cap.release()
 
def main_old():
  mp_drawing = mp.solutions.drawing_utils
  mp_drawing_styles = mp.solutions.drawing_styles

  cv2.namedWindow("HCI2K50", cv2.WND_PROP_FULLSCREEN)
  cv2.setWindowProperty("HCI2K50",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
  
  positions = [[100, 100], [200, 200], [300, 300]]  
  rectangles = [Window(pos, [100, 100]) for pos in positions]

  # For webcam input:
  cap = cv2.VideoCapture(1)
  detector = PoseDetector()

  with detector.holistic as holistic:
    while cap.isOpened():
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        continue

      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      H, W, C = image.shape
      results, _, _, _ = detector.detect(image, holistic)

      # Draw landmark annotation on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      # mp_drawing.draw_landmarks(
      #     image,
      #     results.face_landmarks,
      #     detector.mp_holistic.FACEMESH_TESSELATION,
      #     landmark_drawing_spec=None,
      #     connection_drawing_spec=mp_drawing.DrawingSpec(
      #       color=(237, 247, 203),
      #       thickness=1,
      #       circle_radius=1
      #     )
      # )
      
      mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        detector.mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(
          color=(91, 57, 35),
          thickness=2,
          circle_radius=3
        ),
        connection_drawing_spec=mp_drawing.DrawingSpec(
          color=(237, 247, 203),
          thickness=1,
          circle_radius=1
        )
      )
      mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        detector.mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(
          color=(91, 57, 35),
          thickness=2,
          circle_radius=3
        ),
        connection_drawing_spec=mp_drawing.DrawingSpec(
          color=(237, 247, 203),
          thickness=1,
          circle_radius=1
        )
      )
      
      distance_r = 100.
      click_pose_r = (0, 0)
      click_r = False
      distance_l = 100.
      click_pose_l = (0, 0)
      click_l = False
      
      click_threshold = 20
      # detect if click
      try:
        x_thumb_r = int(W*results.right_hand_landmarks.landmark[4].x)
        y_thumb_r = int(H*results.right_hand_landmarks.landmark[4].y)
        x_index_r = int(W*results.right_hand_landmarks.landmark[8].x)
        y_index_r = int(H*results.right_hand_landmarks.landmark[8].y)
        cv2.line(image,(x_thumb_r,y_thumb_r),(x_index_r,y_index_r),(255,0,0),2)
        distance_r = math.sqrt(math.pow(x_thumb_r-x_index_r, 2)+math.pow(y_thumb_r-y_index_r, 2))
        if distance_r < click_threshold:
          click_r = True
          click_pos_r = ((x_thumb_r+x_index_r)//2, (y_thumb_r+y_index_r)//2)
        color = (0,255,0) if click_r else (0,0,255)
        cv2.circle(image, (x_thumb_r, y_thumb_r) , 5, color, -1)
        cv2.circle(image, (x_index_r, y_index_r) , 5, color, -1) 
      except:
        pass  
      
      try:
        x_thumb_l = int(W*results.left_hand_landmarks.landmark[4].x)
        y_thumb_l = int(H*results.left_hand_landmarks.landmark[4].y)
        x_index_l = int(W*results.left_hand_landmarks.landmark[8].x)
        y_index_l = int(H*results.left_hand_landmarks.landmark[8].y)
        cv2.line(image,(x_thumb_l,y_thumb_l),(x_index_l,y_index_l),(255,0,0),2)
        distance_l = math.sqrt(math.pow(x_thumb_l-x_index_l, 2)+math.pow(y_thumb_l-y_index_l, 2))
        if distance_l < click_threshold:
          click_l = True
          click_pos_l = ((x_thumb_l+x_index_l)//2, (y_thumb_l+y_index_l)//2)
        color = (0,255,0) if click_l else (0,0,255)
        cv2.circle(image, (x_thumb_l, y_thumb_l) , 5, color, -1)
        cv2.circle(image, (x_index_l, y_index_l) , 5, color, -1) 
      except:
        pass
      
      if click_l or click_r:
        click_pos = click_pos_l if click_l else click_pos_r
        for rectangle in rectangles:
          if rectangle.in_rectangle(click_pos):
            rectangle.set_pos(click_pos)
        
      for rectangle in rectangles:
        rectangle.draw(image)
        
            
      # cv2.circle(image, (447,63) , 63, (0,0,255), -1)
      image = cv2.flip(image, 1)
      cv2.putText(image,"Right : {:.2f}".format(distance_r),(5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),1,cv2.LINE_AA)
      cv2.putText(image,"Left : {:.2f}".format(distance_l),(5, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),1,cv2.LINE_AA)
      cv2.imshow('HCI2K50', image)
      if cv2.waitKey(5) & 0xFF == ord('q'):
        break
  cap.release()
  
  
if __name__ == '__main__':
  main()