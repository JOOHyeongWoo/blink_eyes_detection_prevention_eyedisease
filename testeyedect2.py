import cv2, dlib, sys
import numpy as np

scaler = 1


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('blink_eyes_detection_prevention_eyedisease/shape_predictor_68_face_landmarks.dat')


img = cv2.imread('C:/Users/hw/Desktop/trump.png', 1)

faces = detector(img)
face = faces[0]


img = cv2.resize(img, (int(img.shape[1] * scaler), int(img.shape[0] * scaler)))

dlib_shape = predictor(img, face)
shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])

for s in shape_2d:
  cv2.circle(img, center=tuple(s), radius=1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)


  
cv2.imshow('facial landmarks', img)
cv2.waitKey(0)
