#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: leidelong
@license: Apache Licence 
@contact: leidl8907@gmail.com
@site: https://github.com/JoneNash
@software: PyCharm Community Edition
@time: 2016/12/27 下午4:24
"""

import cv2

face_haar = cv2.CascadeClassifier("/Users/leidelong/data/opencv/data/haarcascades/haarcascade_frontalface_default.xml")
eye_haar = cv2.CascadeClassifier("/Users/leidelong/data/opencv/data/haarcascades/haarcascade_eye.xml")

cam = cv2.VideoCapture(0)

while True:
    _, img = cam.read()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_haar.detectMultiScale(gray_img, 1.3, 5)
    for face_x, face_y, face_w, face_h in faces:
        cv2.rectangle(img, (face_x, face_y), (face_x + face_w, face_y + face_h), (0, 255, 0), 2)

        roi_gray_img = gray_img[face_y:face_y + face_h, face_x:face_x + face_w]
        roi_img = img[face_y:face_y + face_h, face_x:face_x + face_w]
        eyes = eye_haar.detectMultiScale(roi_gray_img, 1.3, 5)
        for eye_x, eye_y, eye_w, eye_h in eyes:
            cv2.rectangle(roi_img, (eye_x, eye_y), (eye_x + eye_w, eye_y + eye_h), (255, 0, 0), 2)

    cv2.imshow('img', img)
    key = cv2.waitKey(30) & 0xff
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()