import os
from datetime import datetime

import cv2
import face_recognition
import numpy as np

path = 'ImagesBasic'
images = []
classNames = []

mylist = os.listdir(path)
print(mylist)
for cls in mylist:
    currentImage = cv2.imread(f'{path}/{cls}')
    images.append(currentImage)
    classNames.append(os.path.splitext(cls)[0])
print(classNames)


def findEncoding(images):
    encodinglist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodinglist.append(encode)
    return encodinglist


def markAttendece(name):
    with open('Attendence', 'r+') as f:
        myDatalist = f.readline()
        nameList = []
        for line in myDatalist:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            datestr = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{datestr}')


encodeListknown = findEncoding(images)
print('Encoding Complete')

capture = cv2.VideoCapture(0)

while True:
    success, img = capture.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceInFrame = face_recognition.face_locations(imgS)
    encodeFrame = face_recognition.face_encodings(imgS, faceInFrame)

    for encodeFace, FaceLoc in zip(encodeFrame, faceInFrame):
        matches = face_recognition.compare_faces(encodeListknown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListknown, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = FaceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 25), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (255, 255, 25), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255, 0), 2)
            markAttendece(name)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)
    
# imgSaif = face_recognition.load_image_file('ImagesBasic/Saif.jpeg')
# imgSaif = cv2.cvtColor(imgSaif, cv2.COLOR_BGR2RGB)
# imgTest = face_recognition.load_image_file('ImagesBasic/Saif Test.jpeg')
# imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)
