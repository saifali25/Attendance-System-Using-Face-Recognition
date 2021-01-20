import cv2
import face_recognition
import numpy as np

imgSaif = face_recognition.load_image_file('ImagesBasic/Saif.jpeg')
imgSaif = cv2.cvtColor(imgSaif, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('ImagesBasic/Saif Test.jpeg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgSaif)[0]
encodeFace = face_recognition.face_encodings(imgSaif)[0]
cv2.rectangle(imgSaif,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,255,25),2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,255,25),2)

results = face_recognition.compare_faces([encodeFace],encodeTest)
faceDis = face_recognition.face_distance([encodeFace],encodeTest)
cv2.putText(imgSaif,f'{results} {round(faceDis[0],2)}',(30,50),cv2.FONT_HERSHEY_PLAIN,1,(255,255,25),1)


cv2.imshow('Saif Image', imgSaif)
cv2.imshow('Saif Test', imgTest)
cv2.waitKey(0)