#  -*- coding: utf-8 -*-

from kivy.app import App
from kivy.uix.camera import Camera
import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
import os
import sys
from scipy.spatial import distance
import threading
import time
from time import sleep
from plyer import notification


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

I_SIZE = (34, 26)

#model = load_model(resource_path('2021_06_02_09_47_46.h5')) 
model = load_model('2021_06_02_09_47_46.h5')


model.summary()

class Eye_check:
    programstate=True
    max_blinknum_permin= 20
    check_permin=1

    def __init__(self):
        self.VISIBILITY_THRESHOLD = 0.5
        self.PRESENCE_THRESHOLD = 0.5
        self.Landmark_eye = [
            33, 7, 163, 144, 145, 153, 154, 155,
            133, 246, 161, 160, 159, 158, 157, 173,
            263, 249, 390, 373, 374, 380, 381, 382,
            362, 466, 388, 387, 386, 385, 384, 398
        ]
        #self.model = load_model(resource_path('2021_06_02_09_47_46.h5'))
        self.model = load_model('2021_06_02_09_47_46.h5')
        self.I_SIZE = (34, 26)
        self.mp_drawing = mp.solutions.drawing_utils

    def cut_eye(self, img, eye_points):
        x1, y1 = np.amin(eye_points, axis=0)
        x2, y2 = np.amax(eye_points, axis=0)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        w = (x2 - x1) * 1.2
        h = w * self.I_SIZE[1] / self.I_SIZE[0]

        margin_x, margin_y = w / 2, h / 2

        min_x, min_y = int(cx - margin_x), int(cy - margin_y)
        max_x, max_y = int(cx + margin_x), int(cy + margin_y)

        eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)
        eye_img = img[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

        return eye_img, eye_rect


    def n_array(self, dict):
        return np.array([x for i, x in dict.items() if i in self.Landmark_eye])

    

    def landmark_dict(self, results, width, height):
        face_landmark = {}
        for face_landmarks in results.multi_face_landmarks:
            for idx, landmark in enumerate(face_landmarks.landmark):
                if ((landmark.HasField('visibility') and landmark.visibility < self.VISIBILITY_THRESHOLD) or (landmark.HasField('presence') and landmark.presence < self.PRESENCE_THRESHOLD)):
                    continue
                landmark_px = self.mp_drawing._normalized_to_pixel_coordinates(landmark.x, landmark.y, width, height)
                if landmark_px:
                    face_landmark[idx] = landmark_px
        return face_landmark

    def changeLabel_checkface(eyeself,self,num):
        self.get_screen('running').ids.check_eye_num.text= '?????? ?????? ?????? ???????????? ????????? ????????? '
        return

    def changeLabel_settingtime(eyeself,self,time):
        self.get_screen('running').ids.check_eye_num.text= '?????? ??? ?????????: '+ str(time) + '???'
        return

    def changeLabel(eyeself,self,closedEyenum):
        self.get_screen('running').ids.check_eye_num.text= '???????????? ??? : ' + str(closedEyenum)
        return


    def notification_media(self):
        title = '??????'
        message= '?????? ????????? ?????????'
        notification.notify(title= title,
                            message= message,
                            app_icon = None,
                            timeout= 5,
                            toast=False)

    def checkblink(self):
        from main import WindowManager
        from main import Running_window
        eye_cnt = 0
        frame = 0
        eye = 0
        eye_list = []

        cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        check = Eye_check()

        

        settingOK= False
        checkOpen = True
        closedEyenum = 0
        timepoint = 1
        max_l = 0.1
        max_r =0.1
        start = time.time()
        sum_l =0
        sum_r=0
        countnum=0
        min_l =1
        min_r =1

        max_time_point=False

        while cap.isOpened():
            ret, img = cap.read()

            if not ret:
                eye_list.append(int(eye_cnt / 2))
                break

            check_setting = False

        
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(img)

        
            img.flags.writeable = True
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        
            eye_img = img.copy()

            if results.multi_face_landmarks:
            
                idx_to_coordinates = check.landmark_dict(results, width, height)
 
                eye_np = check.n_array(idx_to_coordinates)

                
                eye_img_l, eye_rect_l = check.cut_eye(gray, eye_points=eye_np[3:13])  
                eye_img_r, eye_rect_r = check.cut_eye(gray, eye_points=eye_np[19:29])  
                
                eye_img_l = cv2.resize(eye_img_l, dsize=I_SIZE)
                eye_img_r = cv2.resize(eye_img_r, dsize=I_SIZE)
                eye_img_r = cv2.flip(eye_img_r, flipCode=1)

                eye_input_l = eye_img_l.copy().reshape((1, I_SIZE[1], I_SIZE[0], 1)).astype(np.float32) / 255.
                eye_input_r = eye_img_r.copy().reshape((1, I_SIZE[1], I_SIZE[0], 1)).astype(np.float32) / 255.
                
                pred_l = model.predict(eye_input_l)
                pred_r = model.predict(eye_input_r)

                check_setting = True
            else:
                check_setting =False
                settingOK= False
                closedEyenum = 0
  
                check.changeLabel_checkface(self,True)

            if settingOK == False :
                if check_setting == True :
                    if timepoint == 1 :
                        start = time.time()
                        timepoint = 0
                    elif time.time() - start > 5 :
                        settingOK = True
                        max_time_point= True 
                    settime = time.time()-start
                    settime = round(settime,1)
                    settime = str(settime)
                
                    check.changeLabel_settingtime(self,settime)

                    max_l = pred_l if pred_l > max_l else max_l
                    max_r = pred_r if pred_r > max_r else max_r
                    min_l = pred_l if pred_l < min_l else min_l
                    min_r = pred_r if pred_r < min_r else min_r
                    sum_l += pred_l
                    sum_r += pred_r
                    countnum +=1
                    
                elif check_setting == False:
                    max_l = 0.1
                    max_r = 0.1
                    timepoint =1
                    min_l=1.0
                    min_r =1.0
            elif settingOK == True :
                
                max_l = sum_l/countnum
                max_r = sum_r/countnum

                repred_l = pred_l 
                repred_r = pred_r 
                predsize_l = 0.1 
                predsize_r = 0.1 

                state_l = 'open %.2f' if repred_l > predsize_l*max_l  else 'close %.2f'
                state_r = 'open %.2f' if repred_r > predsize_r*max_r  else 'close %.2f'
                

                if repred_l <predsize_l*max_l and repred_r <predsize_r*max_r and checkOpen == True :
                    checkOpen = False
                    closedEyenum += 1
                    check.changeLabel(self,closedEyenum)
                    print(closedEyenum)
                    

                elif repred_l >predsize_l*max_l and repred_r >predsize_r*max_r and checkOpen == False :
                    checkOpen = True
                    

                state_l = state_l % repred_l
                state_r = state_r % repred_r

                if max_time_point== True:
                    start=time.time()
                    max_time_point= False
                if time.time() -start> (check.check_permin*60) :
                    if closedEyenum < check.max_blinknum_permin :
                        check.notification_media()
                        start=time.time()
                        closedEyenum = 0
                        max_time_point= True
                        
            
               
            
            if check.programstate == False:
                break
        

        face_mesh.close()
        cap.release()
       
        return
     

