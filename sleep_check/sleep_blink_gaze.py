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
from gaze_tracking import GazeTracking
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear
def init_open_ear():
    time.sleep(5)
    print("open init time sleep")
    ear_list = []
    th_message1 = Thread(target=init_message('o'))
    th_message1.deamon = True
    th_message1.start()
    for i in range(7):
        ear_list.append(both_ear)
        time.sleep(1)
    global OPEN_EAR
    OPEN_EAR = sum(ear_list) / len(ear_list)
    # print("open list =", ear_list)
    print("OPEN_EAR =", OPEN_EAR, "\n")
def init_close_ear():
    time.sleep(2)
    th_open.join()
    time.sleep(5)
    print("close init time sleep")
    ear_list = []
    th_message2 = Thread(target=init_message('c'))
    th_message2.deamon = True
    th_message2.start()
    time.sleep(1)
    for i in range(7):
        ear_list.append(both_ear)
        time.sleep(1)
    CLOSE_EAR = sum(ear_list) / len(ear_list)
    global EAR_THRESH
    EAR_THRESH = (((OPEN_EAR - CLOSE_EAR) / 2) + CLOSE_EAR)  # EAR_THRESH means 50% of the being opened eyes state
    # print("close list =", ear_list)
    print("CLOSE_EAR =", CLOSE_EAR, "\n")
    print("The last EAR_THRESH's value :", EAR_THRESH, "\n")
def init_message(what):
    print("init_message")
    if what == 'o':
        alarm.sound_alarm("openeye.mp3")
    else:
        alarm.sound_alarm("closeeye.mp3")
#####################################################################################################################
# 1. Variables for checking EAR.
# 2. Variables for detecting if user is asleep.
# 3. When the alarm rings, measure the time eyes are being closed.
# 4. When the alarm is rang, count the number of times it is rang, and prevent the alarm from ringing continuously.
# 5. We should count the time eyes are being opened for data labeling.
# 6. Variables for trained data generation and calculation fps.
# 7. Detect face & eyes.
# 8. Run the cam.
# 9. Threads to run the functions in which determine the EAR_THRESH.
# 1.
OPEN_EAR = 0  # For init_open_ear()
EAR_THRESH = 0  # Threashold value
# 5.
print("loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
# 6.
print("starting video stream thread...")
vs = VideoStream(src=0).start()
time.sleep(1.0)
# 7.
th_open = Thread(target=init_open_ear)
th_open.deamon = True
th_open.start()
th_close = Thread(target=init_close_ear)
th_close.deamon = True
th_close.start()
# 8.
COUNTER = 0

""" FPS 지정 !!!!!!!!"""


FPS = 9  # 초당 프레임수가 일정하지 않아서 45로 고정값을 둠(컴퓨터 사양마다 상이할듯)

# FPS 확인
# curTime = time.time()
# sec = curTime - prevTime
# prevTime = curTime
# FPS = 1/(sec)
# print("FPS is : ",FPS)

# ----------------------------------"



CHECK_FRAME = 0
# sleep_blink = 21.42
sleep_criterion = 0
close_state = 0
close_time = 0
# 자고 있는 상태 체크
COUNTER_ONE = 0
prevTime = 0
SLEEP = False
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
# 동공 움직임
curr_location = [0, 0]
prev_location = [0, 0]
prev_second = -1
prev_second2 = -1
prev_min = -1
prev_min2 = -1
distraction_point = 0
distance = []
photo_block = []
arr = []
gaze = GazeTracking()
#####################################################################################################################
while True:
    frame = vs.read()
    gaze.refresh(frame)
    frame = imutils.resize(frame, width=400)
    tm = time.localtime()
    L, gray = lr.light_removing(frame)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    text = ''
    rects = detector(gray, 0)

    # checking fps. If you want to check fps, just uncomment below two lines.
    # prev_time, fps = check_fps(prev_time)
    # cv2.putText(frame, "fps : {:.2f}".format(fps), (10,130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,30,20), 2)
    """눈 깜박임 횟수와 아예 자고있는 상태 체크 !!!"""
    for rect in rects:  # 얼굴 탐지
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        # (leftEAR + rightEAR) / 2 => both_ear. # EAR 측정, 양쪽눈 평균
        both_ear = (leftEAR + rightEAR) * 500  # I multiplied by 1000 to enlarge the scope.
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        CHECK_FRAME += 1  # while 문 한번 돌때마다 1프레임씩 증가


        # print(FPS)
        print('Frame is :', CHECK_FRAME)
        print('Counter is :', COUNTER)
        print('Counterone is :', COUNTER_ONE)
        print('criterion is :', sleep_criterion)
        print('close_time is ', close_time)
        print("--------------------------")





        # 감은 상태가 지속되고 있는지? 0=뜬상태, 1=감은상태
        if both_ear <= EAR_THRESH:  # EAR이 임계치 이하이면 한 frame당 counter 증가
            close_state = 1
            close_time += 1
            if close_time >= FPS*3:
                if close_time == FPS * 3:
                    COUNTER_ONE += 1

            if CHECK_FRAME == FPS * 60:  # 60초동안 프레임 수
                if COUNTER_ONE >= 3:  # 눈감은 횟수가 3회 이상이면 졸음 판단
                    SLEEP = True
                    if SLEEP:
                        alarm.select_alarm(0)


        else:
            # 직전까지 눈을 감은 상태였으면(=방금 눈뜬거면)
            if close_state == 1:
                COUNTER += 1
                close_state = 0
            close_time = 0

        if CHECK_FRAME == FPS * 60:
            if COUNTER <= 19:
                if sleep_criterion <= 0:
                    pass
                else:
                    sleep_criterion -= 1
            elif COUNTER >=20 and COUNTER <40:
                # if sleep_criterion >= 0:
                sleep_criterion += 1
            else:
                if sleep_criterion > 2:
                    pass
                else:
                    sleep_criterion += 1
        if sleep_criterion == 0 :
            text = 'good'
        elif sleep_criterion == 1 :
            text = 'normal'
        elif sleep_criterion == 2 :
            text = 'bad'
        else:
            text = 'sleep'
        if CHECK_FRAME > FPS * 60:  # 60초가 지나면 새로운 프레임 시작(60초 얼추 비슷)
            CHECK_FRAME = 0
            COUNTER = 0
            COUNTER_ONE = 0

        # frame 단위로 돌아서 눈 깜빡일때 Counter가 여러개 올라가는 상황 발생
        # 눈 깜빡일때 마다 counter += 1 되게 해야함
        # --> 해결 완료
        cv2.putText(frame, text, (200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.imshow("Sleep management", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    """ 시선 움직임 빈도에 따른 집중도 체크 부분 """
    hori_ratio = gaze.horizontal_ratio()
    verti_ratio = gaze.vertical_ratio()
    try:
        if curr_location == [0,0]:
            curr_location = [hori_ratio, verti_ratio]
        else:
            prev_location = curr_location
            curr_location = [hori_ratio, verti_ratio]
            hori_diff = curr_location[0] - prev_location[0]
            verti_diff = curr_location[1] - prev_location[1]
            if prev_second2 == -1:
                prev_second2 = tm.tm_sec
            else:
                curr_second2 = tm.tm_sec
                if curr_second2 - prev_second2 == 1 or curr_second2 - prev_second2 < 0:
                    distance.append((hori_diff ** 2) + (verti_diff ** 2))
                    prev_second2 = curr_second2
                    # print("distance is : ", distance)
                    if len(photo_block) < 3:
                        photo_block.append((hori_diff ** 2))
                        # print("photoblock is : ", photo_block)
            # len(distance), sum(distance)임의 값 설정
            if len(distance) > 59:
                if sum(distance) > 1: # 수정
                    print('주의 산만')
                    distraction_point += 1
                    distance = distance[1:]
    except:
        curr_location = [0.5, 0.5]
cv2.destroyAllWindows()
vs.stop()
