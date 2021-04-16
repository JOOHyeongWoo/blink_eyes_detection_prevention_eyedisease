import cv2 
import numpy as np
from imutils import face_utils

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

while(True):
    ret, cam = cap.read()

    if(ret) :
        cv2.imshow('camera', cam)
        
        
        if cv2.waitKey(1) & 0xFF == 27: 
            break
                     
cap.release()
cv2.destroyAllWindows()