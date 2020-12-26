"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.

"""

import cv2
from gaze_tracking import GazeTracking
import time
curr_location = [0, 0]
prev_location = [0, 0]


def gaze():

    gaze = GazeTracking()
    # webcam = cv2.VideoCapture(0)
    distraction_point = 0
    tm = time.localtime()
    photo_block = []

    while True:
        # We get a new frame from the webcam
        # _, frame = webcam.read()

        # We send this frame to GazeTracking to analyze it
        # gaze.refresh(frame)


        # frame = gaze.annotated_frame()
        text = ""

        if gaze.is_blinking():
            text = "Blinking"
            # print("blinking")

        if gaze.is_right():
            text = "Looking right"
            # print("right")

        elif gaze.is_left():
            text = "Looking left"
            # print("left")

        elif gaze.is_center():
            text = "Looking center"
            # print("center")


        ## 수 정 시 작

        hori_ratio = gaze.horizontal_ratio()
        verti_ratio = gaze.vertical_ratio()

        try:

            if curr_location == [0,0]:
                curr_location = [hori_ratio, verti_ratio]
                print(curr_location)
            else:
                prev_location = curr_location
                curr_location = [hori_ratio, verti_ratio]
                hori_diff = curr_location[0] - prev_location[0]
                verti_diff = curr_location[1] - prev_location[1]

                if prev_second2 == -1:
                    prev_second2 = tm.tm_sec
                    print(prev_second2)
                else:
                    curr_second2 = tm.tm_sec
                    if curr_second2 - prev_second2 == 1 or curr_second2 - prev_second2 < 0:
                        distance.append((hori_diff ** 2) + (verti_diff ** 2))
                        prev_second2 = curr_second2

                        if len(photo_block) < 3:
                            photo_block.append((hori_diff ** 2))

                # len(distance), sum(distance)임의 값 설정
                if len(distance) > 59:
                    if sum(distance) > 1:
                        print('주의 산만')
                        distraction_point += 1
                        distance = distance[1:]

        except:
            curr_location = [0.5, 0.5]

        ## 수 정 끝


        # cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

        # left_pupil = gaze.pupil_left_coords()
        # right_pupil = gaze.pupil_right_coords()
        # cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        # cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        #
        # cv2.imshow("Demo", frame)

# esc(27번키) 를 누르면 프로그램 종료
#         if cv2.waitKey(1) == 27:
#             break

