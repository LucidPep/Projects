import cv2
import os
import numpy as np

#Подключаемся к камере
cap = cv2.VideoCapture(1)
cap.set(3, 640) #Высота
cap.set(4, 480) #Ширина

#Классификатор каскадов Хаар
faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')

#
face_id = input('\n Enter user id and press ==>  ')
print("\n [INFO] Initializing face capture. Look the camera and wait ...")
count = 0

#Запись с камеры
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0,  (640, 480))

#Луп для считывания данных с камеры
while True:
    #Собственно, считывание
    ret, frame = cap.read()

    #Переворот камеры
    img = cv2.flip(frame, -1)

    #Серый
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    out.write(frame)

    #Классификатор
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20)
    )

    #Маркировка лица прямоугольником
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        count += 1
        #Сохранение захваченного фото
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h, x:x+w])

        cv2.imshow('video', frame)

    cv2.imshow('test', frame)
    #Останавливаем луп
    k = cv2.waitKey(1) & 0xFF == ord('q')
    if k == 27:
        break
    elif count >= 30:
        break

cap.release()
out.release()
cv2.destroyAllWindows()