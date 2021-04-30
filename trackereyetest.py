import cv2, dlib
import numpy as np
from imutils import face_utils
import sys

IMG_SIZE = (34, 26)
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

  eye_img = grayframe[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

  return eye_img, eye_rect, w, h


#카메라 번호
# /dev/video0 = 0
# 이곳에 동영상 파일명의 위치를 넣어주면 동영상 재생으로 동작함
CAM_ID = 0

# 추적기능 상태
#얼굴 인식
TRACKING_STATE_CHECK = 0
#얼굴인식 위치를 기반으로 추적 기능 초기화
TRACKING_STATE_INIT  = 1
#추적 동작
TRACKING_STATE_ON    = 2

#OpenCV 버전 확인
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

if __name__ == '__main__' :
    #버전 출력
    print((cv2.__version__).split('.'))

    # 트레킹 함수 선택
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    # 기본 KCF(Kernelized Correlation Filters)가 속도가 빠르다.
    tracker_type = tracker_types[2]

    # OpenCV 서브 버전별 함수 호출명이 다르다.
    if int(minor_ver) < 3:
        #3.2 이하
        tracker = cv2.Tracker_create(tracker_type)
    else:
        #3.3 이상
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()

    #카메라 열기
    video = cv2.VideoCapture(0,cv2.CAP_DSHOW)

    #카메라가 정상적으로 열리지 않았다면 프로그램 종료
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    #얼굴인식 함수 생성
    #face_cascade = cv2.CascadeClassifier()
    #얼굴인식용 haar 불러오기
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('blink_eyes_detection_prevention_eyedisease/shape_predictor_68_face_landmarks.dat')

    #추적 상태 저장용 변수
    TrackingState = 0
    #추적 영역 저장용 변수
    TrackingROI = (0,0,0,0)

    #프로그램 시작
    while True:
        #카메라에서 1 frame 읽어오기
        ok, frame = video.read()
        #영상이 없다면 프로그램 종료를 위해 while 문 빠져나옴
        if not ok:
            break

        #추적 상태가 얼굴 인식이면 얼굴 인식 기능 동작
        #처음에 무조건 여기부터 들어옴
        print("starttttt")
        if TrackingState == TRACKING_STATE_CHECK:
            #흑백 변경
            grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #히스토그램 평활화(재분할)
            grayframe = cv2.equalizeHist(grayframe)
            #얼굴 인식
            faces = detector(grayframe)

            #얼굴이 1개라도 잡혔다면
            if len(faces) > 0:
                shapes = predictor(grayframe, faces[0])
                shapes = face_utils.shape_to_np(shapes)
                eye_img_l, eye_rect_l ,w1, h1 = crop_eye(grayframe, eye_points=shapes[36:42])
                eye_img_r, eye_rect_r , w2 , h2= crop_eye(grayframe, eye_points=shapes[42:48])
                #얼굴 인식된 위치 및 크기 얻기
                x= eye_rect_l[0]
                y = eye_rect_l[1]
                w = eye_rect_l[2] -eye_rect_l[0]
                h = eye_rect_l[3] - eye_rect_l[1]
                #인식된 위치및 크기를 TrackingROI에 저장
                TrackingROI = (x,y,w,h)
                #인식된 얼굴 표시 순식간에 지나가서 거의 볼수 없음(녹색)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3, 4, 0)
                #추적 상태를 추적 초기화로 변경
                TrackingState = TRACKING_STATE_ON
                print('det w : %d ' % w + 'h : %d ' % h)
                ok = tracker.init(frame, TrackingROI)

        #추적 초기화
        #얼굴이 인식되면 동작함
      
    
        elif TrackingState == TRACKING_STATE_INIT:
            #추적 함수 초기화
            #얼굴인식으로 가져온 위치와 크기를 함께 넣어준다.
            ok = tracker.init(frame, TrackingROI)
            if ok:
                #성공하였다면 추적 동작상태로 변경
                TrackingState = TRACKING_STATE_ON
                print('tracking init succeeded')
            else:
                #실패하였다면 얼굴 인식상태로 다시 돌아감
                TrackingState = TRACKING_STATE_CHECK
                print('tracking init faileddddd')


        #추적 동작
        elif TrackingState == TRACKING_STATE_ON:
            #추적
            ok, TrackingROI = tracker.update(frame)
            if ok:
                #추적 성공했다면
                p1 = (int(TrackingROI[0]), int(TrackingROI[1]))
                p2 = (int(TrackingROI[0] + TrackingROI[2]), int(TrackingROI[1] + TrackingROI[3]))
                #화면에 박스로 표시 (파랑)
                cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
                print('success x %d ' % (int(TrackingROI[0])) + 'y %d ' % (int(TrackingROI[1])) +
                        'w %d ' % (int(TrackingROI[2])) + 'h %d ' % (int(TrackingROI[3])))
            else:
                print('Tracking failed')

                TrackingState = TRACKING_STATE_CHECK

        #화면에 카메라 영상 표시
        #추적된 박스가 있으면 같이 표시됨
        cv2.imshow("Tracking", frame)

        #ESC키를 누르면 break로 종료
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break