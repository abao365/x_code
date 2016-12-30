#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: leidelong
@license: Apache Licence 
@contact: leidl8907@gmail.com
@site: https://github.com/JoneNash
@software: PyCharm Community Edition
@time: 2016/12/27 下午5:45
"""

import cv2

mouse_haar = cv2.CascadeClassifier("data/cascade.xml")

cam = cv2.VideoCapture(0)

while True:
    _, img = cam.read()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # http://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html
    mouse = mouse_haar.detectMultiScale(gray_img, 1.1, 2)  # 调整参数

    for mouse_x, mouse_y, mouse_w, mouse_h in mouse:
        cv2.rectangle(img, (mouse_x, mouse_y), (mouse_x + mouse_w, mouse_y + mouse_h), (0, 255, 0), 2)

    cv2.imshow('img', img)
    key = cv2.waitKey(30) & 0xff
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()