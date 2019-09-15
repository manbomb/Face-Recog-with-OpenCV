import cv2
import numpy as np

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:

        _, foto = cap.read()
        gray = cv2.cvtColor(foto, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
                cv2.rectangle(foto, (x,y),(x+w,y+h), (0,0,255), 2)
                
        cv2.imshow("Video", foto)
        key = cv2.waitKey(1)

        if key == 27:
                break


cap.release()
cv2.destroyAllWindows()
