from kivy.app import App
from kivy.uix.camera import Camera
import cv2, dlib
import numpy as np
from imutils import face_utils

import numpy as np
from keras.models import load_model
#import os
from scipy.spatial import distance
import threading
import time
from time import sleep





detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('blink_eyes_detection_prevention_eyedisease/kivy/shape_predictor_68_face_landmarks.dat')


model = load_model('blink_eyes_detection_prevention_eyedisease/kivy/2021_06_02_09_47_46.h5')
model.summary()


IMG_SIZE = (34, 26)
class Eye_check:
    programstate=True
    max_blinknum_permin= 20
    check_permin=1
    IMG_SIZE = (34, 26)

    def crop_eye(self, img, eye_points):
        x1, y1 = np.amin(eye_points, axis=0)
        x2, y2 = np.amax(eye_points, axis=0)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        w = (x2 - x1) * 1.2
        h = w * self.IMG_SIZE[1] / self.IMG_SIZE[0]

        margin_x, margin_y = w / 2, h / 2

        min_x, min_y = int(cx - margin_x), int(cy - margin_y)
        max_x, max_y = int(cx + margin_x), int(cy + margin_y)

        eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)
        eye_img = img[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

        return eye_img, eye_rect



    def changeLabel(eyeself,self,closedEyenum):
        self.get_screen('running').ids.check_eye_num.text= str(closedEyenum)
        return

    def checkblink(self):
        from main import WindowManager
        from main import Running_window
        eye_cnt = 0
        frame = 0
        eye = 0
        eye_list = []

        cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
       
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
            ret, image = cap.read()

            if not ret:
                print("End frame")
                eye_list.append(int(eye_cnt / 2))
                break

            check_setting = False

        
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            
            eye_image = image.copy()

            faces = detector(gray)

            for face in faces:
            
            
                shapes = predictor(gray, face)
                shapes = face_utils.shape_to_np(shapes)

                
                eye_img_l, eye_rect_l = check.crop_eye(gray, eye_points=shapes[36:42])  
                eye_img_r, eye_rect_r = check.crop_eye(gray, eye_points=shapes[42:48])  
                
                eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
                eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
                eye_img_r = cv2.flip(eye_img_r, flipCode=1)

                #cv2.imshow('l', eye_img_l)
                #cv2.imshow('r', eye_img_r)

                eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.
                eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.
                #eye_input_l = eye_img_l.reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.
                #eye_input_r = eye_img_r.reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.
                

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
                        max_time_point= True 
                    settime = time.time()-start
                    settime = str(settime)
                    #cv2.putText(eye_image, settime, (20,400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    #self.get_screen('running').ids.check_eye_num.text=  str(float(settime))
                    max_l = pred_l if pred_l > max_l else max_l
                    max_r = pred_r if pred_r > max_r else max_r
                    min_l = pred_l if pred_l < min_l else min_l
                    min_r = pred_r if pred_r < min_r else min_r
                    sum_l += pred_l
                    sum_r += pred_r
                    countnum +=1
                elif check_setting == False:
                    #cv2.putText(eye_image, "check setting", (20,400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    max_l = 0.1
                    max_r = 0.1
                    timepoint =1
                    min_l=1.0
                    min_r =1.0
            elif settingOK == True :
                
                max_l = sum_l/countnum
                max_r = sum_r/countnum

                repred_l = pred_l #* (1/max_l)
                repred_r = pred_r #* (1/max_r)
                predsize_l = 0.1 #(1 -max_l) *0.9
                predsize_r = 0.1 #(1 -max_r) *0.9

                state_l = 'open %.2f' if repred_l > predsize_l*max_l  else 'close %.2f'
                state_r = 'open %.2f' if repred_r > predsize_r*max_r  else 'close %.2f'
                

                if repred_l <predsize_l*max_l and repred_r <predsize_r*max_r and checkOpen == True :
                    checkOpen = False
                    closedEyenum += 1
                    changelabel_thread = threading.Thread(target=check.changeLabel, args=(self,closedEyenum)) 
                    changelabel_thread.setDaemon(True)
                    changelabel_thread.start()
                    print(closedEyenum)
                    #self.get_screen('running').ids.check_eye_num.text= str(int(closedEyenum))


                    #print(closedEyenum)
                elif repred_l >predsize_l*max_l and repred_r >predsize_r*max_r and checkOpen == False :
                    checkOpen = True

                state_l = state_l % repred_l
                state_r = state_r % repred_r

                if max_time_point== True:
                    start=time.time()
                    max_time_point= False
                if time.time() -start> check.check_permin :
                    if closedEyenum < check.max_blinknum_permin :
                        notification_thread = threading.Thread(target=WindowManager.notification, args=(self,)) 
                        notification_thread.setDaemon(True)
                        notification_thread.start()
                        start=time.time()
                        closedEyenum = 0
                        max_time_point= True
                        print("ppppppppppppp")
            
                #cv2.rectangle(eye_image, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(255, 255, 255), thickness=2)
                #cv2.rectangle(eye_image, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(255, 255, 255), thickness=2)

            
                #cv2.putText(eye_image, str(state_l), tuple(eye_rect_l[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                #cv2.putText(eye_image, str(state_r), tuple(eye_rect_r[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                #cv2.putText(eye_image, "closednum:"+ str(closedEyenum) , (20,400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

            #cv2.imshow('MediaPipe EyeMesh', eye_image)  # 눈 인식 표시

            #if cv2.waitKey(1) == ord('q'):
                #eye_list.append(int(eye_cnt / 2))
                #break
            #self.get_screen('running').ids.check_eye_num.text= str(int(closedEyenum))
            
            
            
            if check.programstate == False:
                break
        
        
       

        
        cap.release()
        #print( max_l, max_r)
        #print( predsize_l*max_l, predsize_r*max_r)
        #cv2.destroyAllWindows()
        return
 
        
        

