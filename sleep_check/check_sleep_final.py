#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import imutils
import time
import timeit
import dlib
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
from threading import Timer
from check_cam_fps import check_fps
import make_train_data as mtd
import light_remover as lr
import ringing_alarm as alarm

def eye_aspect_ratio(eye) :
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear
    
def init_open_ear() :
    time.sleep(5)
    print("open init time sleep")
    ear_list = []
    th_message1 = Thread(target = init_message)
    th_message1.deamon = True
    th_message1.start()
    for i in range(7) :
        ear_list.append(both_ear)
        time.sleep(1)
    global OPEN_EAR
    OPEN_EAR = sum(ear_list) / len(ear_list)
    #print("open list =", ear_list)
    print("OPEN_EAR =", OPEN_EAR, "\n")

def init_close_ear() : 
    time.sleep(2)
    th_open.join()
    time.sleep(5)
    print("close init time sleep")
    ear_list = []
    th_message2 = Thread(target = init_message)
    th_message2.deamon = True
    th_message2.start()
    time.sleep(1)
    for i in range(7) :
        ear_list.append(both_ear)
        time.sleep(1)
    CLOSE_EAR = sum(ear_list) / len(ear_list)
    global EAR_THRESH
    EAR_THRESH = (((OPEN_EAR - CLOSE_EAR) / 2) + CLOSE_EAR) #EAR_THRESH means 50% of the being opened eyes state
    #print("close list =", ear_list)
    print("CLOSE_EAR =", CLOSE_EAR, "\n")
    print("The last EAR_THRESH's value :",EAR_THRESH, "\n")

def init_message() :
    print("init_message")
    alarm.sound_alarm("init_sound.mp3")

#####################################################################################################################
#1. Variables for checking EAR.
#2. Variables for detecting if user is asleep.
#3. When the alarm rings, measure the time eyes are being closed.
#4. When the alarm is rang, count the number of times it is rang, and prevent the alarm from ringing continuously.
#5. We should count the time eyes are being opened for data labeling.
#6. Variables for trained data generation and calculation fps.
#7. Detect face & eyes.
#8. Run the cam.
#9. Threads to run the functions in which determine the EAR_THRESH. 

#1.
OPEN_EAR = 0 #For init_open_ear()
EAR_THRESH = 0 #Threashold value

#2.
#It doesn't matter what you use instead of a consecutive frame to check out drowsiness state. (ex. timer)
FPS = 45  # 초당 프레임수가 일정하지 않아서 45로 고정값을 둠(컴퓨터 사양마다 상이할듯)
CHECK_FRAME = 0
COUNTER = 0  #Frames counter.
COUNTER_ONE = 0
prevTime = 0
SLEEP = False

#3.
closed_eyes_time = [] #The time eyes were being offed.
TIMER_FLAG = False #Flag to activate 'start_closing' variable, which measures the eyes closing time.
ALARM_FLAG = False #Flag to check if alarm has ever been triggered.

#4. make trained data
np.random.seed(30)
power, nomal, short = mtd.start(25) #actually this three values aren't used now. (if you use this, you can do the plotting)
#The array the actual test data is placed.
test_data = []
#The array the actual labeld data of test data is placed.
result_data = []
#For calculate fps
prev_time = 0

#5.
print("loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

#6.
print("starting video stream thread...")
vs = VideoStream(src=0).start()
time.sleep(1.0)

#7.
th_open = Thread(target = init_open_ear)
th_open.deamon = True
th_open.start()
th_close = Thread(target = init_close_ear)
th_close.deamon = True
th_close.start()

#####################################################################################################################

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width = 400)

    L, gray = lr.light_removing(frame)
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray,0)
    #FPS 확인
    # curTime = time.time()
    # sec = curTime - prevTime
    # prevTime = curTime
    # FPS = 1/(sec)

    #checking fps. If you want to check fps, just uncomment below two lines.
    #prev_time, fps = check_fps(prev_time)
    #cv2.putText(frame, "fps : {:.2f}".format(fps), (10,130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,30,20), 2)

    for rect in rects: # 얼굴 탐지
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        #(leftEAR + rightEAR) / 2 => both_ear. # EAR 측정, 양쪽눈 평균
        both_ear = (leftEAR + rightEAR) * 500  #I multiplied by 1000 to enlarge the scope.

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0,255,0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0,255,0), 1)

        CHECK_FRAME += 1 # while 문 한번 돌때마다 1프레임씩 증가

        if both_ear < EAR_THRESH :  # EAR이 임계치 이하이면 한 frame당 counter 증가
            COUNTER += 1

            # print('COUNTER is :', COUNTER)
            if COUNTER >= FPS * 3:  # FPS = 1초당 Frame 수 = 45, counter가 90이상이면 3초이상 눈 감는다.
                if COUNTER == FPS * 3:  # 3초 이상 눈감았을 때
                    COUNTER_ONE += 1

            # print('counter_one is : ',COUNTER_ONE)
            # print('check_frame is :',CHECK_FRAME)
            if CHECK_FRAME == FPS * 60:  # 60초동안 프레임 수
                if COUNTER_ONE >= 3:  # 눈감은 횟수가 3회 이상이면 졸음 판단
                    SLEEP = True
                    # print(SLEEP)
                    if SLEEP:
                        alarm.select_alarm(0)  # 교수님 멘트
            if CHECK_FRAME > FPS * 60:  # 60초가 지나면 새로운 프레임 시작(60초 얼추 비슷)
                CHECK_FRAME = 0
                COUNTER_ONE = 0

    # 교수님 목소리 가져오기,
        # 눈을 탐지하지 못하면 프레임 증가X,



        else :
            COUNTER = 0


        cv2.putText(frame, "Concentration idx : {}".format(both_ear), (300,130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,30,20), 2)

    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()

