import math
import numpy as np
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

cv2.namedWindow("HCI2K50", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("HCI2K50",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

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
  
positions = [[100, 100], [200, 200], [300, 300]]  
rectangles = [Window(pos, [100, 100]) for pos in positions]

# For webcam input:
cap = cv2.VideoCapture(1)
with mp_holistic.Holistic(
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3, 
    model_complexity=1, 
    refine_face_landmarks=True) as holistic:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    H, W, C = image.shape
    results = holistic.process(image)

    # Draw landmark annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # mp_drawing.draw_landmarks(
    #     image,
    #     results.face_landmarks,
    #     mp_holistic.FACEMESH_TESSELATION,
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
      mp_holistic.HAND_CONNECTIONS,
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
      mp_holistic.HAND_CONNECTIONS,
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