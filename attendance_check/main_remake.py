#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 03:32:37 2020

@author: 권용진
"""

import check_ponix
from time import sleep
from gaze_tracking import gaze_tracking
import cv2

def main():
    
    ponix = check_ponix.moving_ponix()
    ponix.set_ponix()
    gaze = gaze_tracking.GazeTracking()
    webcam = cv2.VideoCapture(0)
    face_detect = 1
    
    while True:
        _, frame = webcam.read()

        gaze.refresh(frame)

        if face_detect:
            gaze

        frame = gaze.annotated_frame()

        pos = ponix.start_game()
        if pos > 500:
            if gaze.is_right() == True: # 포닉스 중심기준 오른쪽을 본다면 right
                ponix.sign('Right', True)
            else:
                ponix.sign('Right', False)

            
        if pos < 500:
            if gaze.is_left() == True:  # 왼쪽을 본다면 left
                ponix.sign('Left', True)

            else:
                ponix.sign('Left', False)
                             
            
        if pos < -100 or pos > 1100:
            if ponix.count > 100 :   # count 비교해서 100 이상이면 True, 틀리면 False
                ponix.check_eye(True) 
            else:
                ponix.check_eye(False)
            ponix.end_game()
            break


        if cv2.waitKey(1) == 27:
            break

if __name__ == '__main__':
    main()


# In[ ]:




