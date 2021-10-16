import cv2, dlib
import numpy as np
from imutils import face_utils
from keras.models import load_model
import time

IMG_SIZE = (34, 26)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('blink_eyes_detection_prevention_eyedisease/check_blinkeye_model/shape_predictor_68_face_landmarks.dat')

model = load_model('blink_eyes_detection_prevention_eyedisease/check_blinkeye_model/models/2021_06_02_09_47_46.h5')
model.summary()

def crop_eye(img, eye_points):
  x1, y1 = np.amin(eye_points, axis=0)
  x2, y2 = np.amax(eye_points, axis=0)
  cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

  w = (x2 - x1) * 1.2
  h = w * IMG_SIZE[1] / IMG_SIZE[0]

  margin_x, margin_y = w / 2, h / 2

  min_x, min_y = int(cx - margin_x), int(cy - margin_y)
  max_x, max_y = int(cx + margin_x), int(cy + margin_y)

  eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)

  eye_img = gray[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

  return eye_img, eye_rect



cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)


#cap = cv2.VideoCapture('blink_eyes_detection_prevention_eyedisease/testsample/two_blink_withoutmask.mp4')

settingOK= False
checkOpen = True
closedEyenum = 0
timepoint = 1
max_l = 0
max_r =0
start = time.time()
while cap.isOpened():
  ret, img_ori = cap.read()

  if not ret:
    break

  check_setting = False

  img = img_ori.copy()
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  grayframe = cv2.equalizeHist(gray)

  faces = detector(gray)

  for face in faces:
    shapes = predictor(gray, face)
    shapes = face_utils.shape_to_np(shapes)

    eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42])
    eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48])

    eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
    eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
    eye_img_r = cv2.flip(eye_img_r, flipCode=1)

    
    #eye_l = cv2.equalizeHist(eye_img_l)
    #eye_r = cv2.equalizeHist(eye_img_r)
    #cv2.imshow('l', eye_img_l)
    #cv2.imshow('r', eye_img_r)

    eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.
    eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.
    

    pred_l = model.predict(eye_input_l)
    pred_r = model.predict(eye_input_r)


    check_setting = True

  if settingOK == False :
    if check_setting == True :
        if timepoint == 1 :
            start = time.time()
            timepoint = 0
        elif time.time() - start > 5 :
            settingOK = True 
        settime = time.time()-start
        settime = str(settime)
        cv2.putText(img, settime, (20,400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        max_l = pred_l if pred_l > max_l else max_l
        max_r = pred_r if pred_r > max_r else max_r
    elif check_setting == False:
        cv2.putText(img, "check setting", (20,400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        max_l = 0
        max_r = 0
        timepoint =1
  elif settingOK == True :
      

    repred_l = pred_l #* (1/max_l)
    repred_r = pred_r #* (1/max_r)

    state_l = 'open %.2f' if repred_l > 0.1*max_l else 'closed %.2f'
    state_r = 'open %.2f' if repred_r > 0.1*max_r else 'closed %.2f'
    #print( repred_l, repred_r)
    #state_l = 'open %.2f' if repred_l > 0.02 else 'closed %.2f'
    #state_r = 'open %.2f' if repred_r > 0.02 else 'closed %.2f'

    '''
    if repred_l <0.02 and repred_r <0.02 and checkOpen == True :
        checkOpen = False
        closedEyenum += 1
    elif repred_l >0.02 and repred_r >0.02 and checkOpen == False :
        checkOpen = True
    '''

    if repred_l <0.1*max_l and repred_r <0.1*max_r and checkOpen == True :
        checkOpen = False
        closedEyenum += 1
    elif repred_l >0.1*max_l and repred_r >0.1*max_r and checkOpen == False :
        checkOpen = True

    state_l = state_l % repred_l
    state_r = state_r % repred_r
    print( pred_l, pred_r)
    

    cv2.rectangle(img, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(255,255,255), thickness=2)
    cv2.rectangle(img, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(255,255,255), thickness=2)

    cv2.putText(img, state_l, tuple(eye_rect_l[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(img, state_r, tuple(eye_rect_r[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.putText(img, "closednum:"+ str(closedEyenum) , (20,400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

  #img = cv2.resize(img, dsize=(0, 0), fx=0.3, fy=0.3)
  cv2.imshow('result', img)
  if cv2.waitKey(1) == ord('q'):
    break

print( max_l, max_r)

