from datetime import datetime

import cv2
import face_recognition


def findEncoding(images):
    encodinglist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodinglist.append(encode)
    return encodinglist

def markAttendece(name):

    with open('../Attendence', 'r+') as f:
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