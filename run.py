########## TODO ##########
# - Define gesture and formulate click as gesture
# - Simple model for click and hold_click


import math
import numpy as np
import cv2
import mediapipe as mp

from utils import Window, Drawer
from detector import PoseDetector
from model import Hand

IMG_SIZE = (480, 640, 3)

def main(camera=1):
  mp_drawing = mp.solutions.drawing_utils
  mp_drawing_styles = mp.solutions.drawing_styles
  
  cv2.namedWindow("HCI2K50", cv2.WND_PROP_FULLSCREEN)
  cv2.setWindowProperty("HCI2K50",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
  
  detector = PoseDetector()
  drawer = Drawer(
    detector.mp_holistic.FACEMESH_TESSELATION,
    detector.mp_holistic.HAND_CONNECTIONS,
    IMG_SIZE
  )
  
  positions = [[75, 200], [275, 200], [475, 200]]  
  rectangles = [Window(pos, [100, 100]) for pos in positions]
  right_hand, left_hand = Hand(IMG_SIZE), Hand(IMG_SIZE)
  
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
      results, _, hand_r_lm, hand_l_lm = detector.detect(image, holistic)
      right_hand.update(hand_r_lm)
      left_hand.update(hand_l_lm)
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      
      drawer.draw_mp(image, results)
      
      r_click, r_click_pos = right_hand.is_clicking()
      l_click, l_click_pos = left_hand.is_clicking()

      for rectangle in rectangles:
        if r_click:
          drawer.draw_points(image, right_hand.landmarks, [4, 8], (0, 255, 0))
          if rectangle.in_rectangle(r_click_pos):
            rectangle.set_pos(r_click_pos)
        else:
          drawer.draw_points(image, right_hand.landmarks, [4, 8], (0, 0, 255))
        if l_click:
          drawer.draw_points(image, left_hand.landmarks, [4, 8], (0, 255, 0))
          if rectangle.in_rectangle(l_click_pos):
            rectangle.set_pos(l_click_pos)
        else:
          drawer.draw_points(image, left_hand.landmarks, [4, 8], (0, 0, 255))
        drawer.draw_window(image, rectangle)
      
      image = cv2.flip(image, 1)
      cv2.imshow('HCI2K50', image)
      if cv2.waitKey(5) & 0xFF == ord('q'):
        break
  cap.release()
  
  
if __name__ == '__main__':
  main()