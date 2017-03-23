import numpy as np
import sys
import cv2
import time

videoArg = sys.argv[1]

lower = np.array([0, 48, 80], np.uint8)
upper = np.array([20, 255, 255], np.uint8)

video = cv2.VideoCapture(videoArg)

fps = video.get(cv2.CAP_PROP_FPS)

while True:
    start = time.time()

    grabbed, frame = video.read()
    if not grabbed:
        break

    if cv2.waitKey(int(fps)) & 0xFF == ord("q"):
        break
    
    HSV_Converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(HSV_Converted, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skinMask = cv2.morphologyEx(skinMask, cv2.MORPH_OPEN, kernel, iterations=1)

    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)

    im2, cnts, hierarchy = cv2.findContours(skinMask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.contourArea(c) < 1000:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        #cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 0), 2)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(frame,[box],0,(0,0,255),2)
        
    cv2.imshow('frame',frame)

    end=time.time()
    print(end-start)

video.release()
cv2.destroyAllWindows()
