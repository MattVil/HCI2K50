import mediapipe as mp

class PoseDetector():
  def __init__(self):
    self.mp_holistic = mp.solutions.holistic
    self.holistic = self.mp_holistic.Holistic(
      min_detection_confidence=0.3,
      min_tracking_confidence=0.3, 
      model_complexity=1, 
      refine_face_landmarks=True)
    
  def detect(self, image, holistic):
    results = holistic.process(image)
    face_landmarks, right_hand_landmarks, left_hand_landmarks = [], [], []
    try:
      face_landmarks = results.face_landmarks.landmark
    except:
      pass
    try:
      right_hand_landmarks = results.right_hand_landmarks.landmark
    except:
      pass
    try:
      left_hand_landmarks = results.left_hand_landmarks.landmark
    except:
      pass
    return results, face_landmarks, right_hand_landmarks, left_hand_landmarks