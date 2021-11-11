import math

class Hand():
  def __init__(self, img_size):
    self.detected = False
    self.landmarks = []
    self.img_size = img_size
    
  def update(self, landmarks):
    self.detected = True if landmarks else False
    self.landmarks = landmarks
    
  def get_distance(self, idxs):
    if not self.detected:
      return -1.
    x0 = int(self.img_size[1]*self.landmarks[idxs[0]].x)
    y0 = int(self.img_size[0]*self.landmarks[idxs[0]].y)
    x1 = int(self.img_size[1]*self.landmarks[idxs[1]].x)
    y1 = int(self.img_size[0]*self.landmarks[idxs[1]].y)
    return math.sqrt(math.pow(x0-x1, 2)+math.pow(y0-y1, 2))
  
  def is_clicking(self, click_threshold=25):
    if not self.detected:
      return False, (-1., -1.)
    x0 = int(self.img_size[1]*self.landmarks[4].x)
    y0 = int(self.img_size[0]*self.landmarks[4].y)
    x1 = int(self.img_size[1]*self.landmarks[8].x)
    y1 = int(self.img_size[0]*self.landmarks[8].y)
    pos = ((x0+x1)//2, (y0+y1)//2)
    click = math.sqrt(math.pow(x0-x1, 2)+math.pow(y0-y1, 2)) < click_threshold
    return click, pos