import cv2 as cv
import numpy as np

path1 = 'D:/Anaconda_all/11__hour_couse_image_processing/Data/a.jpg'
path2 = 'D:/Anaconda_all/11__hour_couse_image_processing/Data/cascades/haarcascade_frontalface_default.xml'
path3 = 'D:/Anaconda_all/11__hour_couse_image_processing/Data/cascades/haarcascade_eye.xml'

#image = cv.imread(path1)
face = cv.CascadeClassifier(path2)
eye  = cv.CascadeClassifier(path3)

def detector(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    face_rec = face.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in face_rec:
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        g_img = gray[y:y + h, x:x + w]
        c_img = image[y:y + h, x:x + w]
        eye_rec = eye.detectMultiScale(g_img, 1.1, 1)
        for (x1, y1, w1, h1) in eye_rec:
            cv.rectangle(c_img, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 255), 3)


    return image



cap = cv.VideoCapture(0, cv.CAP_DSHOW)
while True:
    _, img = cap.read()
    img = cv.flip(img, 2)
    img = cv.resize(img, (800, 600))
    cv.imshow("Face", detector(img))
    if cv.waitKey(30)==27:
        break

cap.release()
cv.destroyAllWindows()


