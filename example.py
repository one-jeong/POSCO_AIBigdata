"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
from gaze_tracking import GazeTracking
import phonix_tracking

def gaze():

    gaze = GazeTracking()
    webcam = cv2.VideoCapture(0)
    face_detect = 1

    while True:
        # We get a new frame from the webcam
        _, frame = webcam.read()

        # We send this frame to GazeTracking to analyze it
        gaze.refresh(frame)

        if face_detect:
            phonix_tracking.detect_video()

        frame = gaze.annotated_frame()
        text = ""

        # if gaze.is_blinking():
        #     text = "Blinking"
        #     print("blinking")

        if gaze.is_right():
            text = "Looking right"
            print("right")

        elif gaze.is_left():
            text = "Looking left"
            print("left")

        # elif gaze.is_center():
        #     text = "Looking center"
        #     print("center")


        # cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

        left_pupil = gaze.pupil_left_coords()
        right_pupil = gaze.pupil_right_coords()
        # cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        # cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

        # cv2.imshow("Demo", frame)

# esc(27번키) 를 누르면 프로그램 종료
        if cv2.waitKey(1) == 27:
            break
