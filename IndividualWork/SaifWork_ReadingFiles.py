import os
from datetime import datetime

import cv2
import face_recognition
import numpy as np

path = '../ImagesBasic'
images = []
classNames = []

mylist = os.listdir(path)
print(mylist)
for cls in mylist:
    currentImage = cv2.imread(f'{path}/{cls}')
    images.append(currentImage)
    classNames.append(os.path.splitext(cls)[0])
print(classNames)