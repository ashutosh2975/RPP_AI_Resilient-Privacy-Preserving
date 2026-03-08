import cv2
import numpy as np


def detect_tumor(image_path):

    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5),0)

    # threshold
    _, thresh = cv2.threshold(blur, 45,255,cv2.THRESH_BINARY)

    # find contours
    contours,_ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area > 500:   # remove noise

            (x,y),radius = cv2.minEnclosingCircle(cnt)

            center = (int(x),int(y))

            radius = int(radius)

            cv2.circle(image,center,radius,(0,0,255),3)

    cv2.imshow("Tumor Detection", image)

    cv2.waitKey(0)

    cv2.destroyAllWindows()