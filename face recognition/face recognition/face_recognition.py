import cv2
from PIL import Image, ImageDraw
import face_recognition
import time
import numpy as np
import math

start_time = time.time()

#图片加载
image = face_recognition.load_image_file("D:/face_sticker/faces/3.jpg")
sticker = cv2.imread("D:/face_sticker/stickers/rabbit.png", -1)


#获得人脸的面部特征和位置
face_landmarks_list = face_recognition.face_landmarks(image)
face_locations = face_recognition.face_locations(image)

src = Image.fromarray(image)
d = ImageDraw.Draw(src)
for face_landmarks,(top, right, bottom, left) in zip(
    face_landmarks_list, face_locations):
    for feature in face_landmarks.keys():
        d.line(face_landmarks[feature])


src.show()
