from cgitb import reset
from a import shape_to_np
from a import FACIAL_LANDMARKS_IDXS
from a import visualize_facial_landmarks
import os
import b
import c
import d
import numpy as np
import argparse
import imutils
import dlib
import cv2


"""
os.chdir("C:/Users/user/OneDrive/Masaüstü/karisikKapali/verisetiSol---/saDonukSolGoz")
print(os.listdir())
ikiAcikResimler = os.listdir()
num = 0
for x in ikiAcikResimler: 
"""

def GozKesmeFonk(gelenGoruntu):

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    #try:
    #os.chdir("C:/Users/irem/OneDrive/Masaüstü/karisikKapali/verisetiSol---/saDonukSolGoz")
    #num = num + 1
    image = cv2.imread(gelenGoruntu)
    #image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # gri tonlamalı görüntüdeki yüzleri algıla
    rects = detector(gray, 1)

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)   
        shape = shape_to_np(shape)
        for (name, (i, j)) in FACIAL_LANDMARKS_IDXS.items():
            clone = image.copy()
            cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2)
            for (x, y) in shape[i:j]:
                #cv2.circle() yöntemi, herhangi bir görüntü üzerinde bir daire çizmek için kullanılır.
                cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
                
            # (x, y, w, h) --> X koordinatı, Y koordinatı, Genişlik, Yükseklik
            (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
            roi = image[y:y + h, x:x + w]
            roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
            #cv2.imshow("ROI", roi)
            #cv2.imshow("Image", clone)
            cv2.waitKey(0)

    print(x)
    print(type(x))    

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = b.shape_to_np(shape)
        for (name, (i, j)) in b.FACIAL_LANDMARKS_IDXS.items():
            clone = image.copy()
            cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2)
            for (xx, yy) in shape[i:j]:
                #cv2.circle() yöntemi, herhangi bir görüntü üzerinde bir daire çizmek için kullanılır.
                cv2.circle(clone, (xx, yy), 1, (0, 0, 255), -1)
                
            # (x, y, w, h) --> X koordinatı, Y koordinatı, Genişlik, Yükseklik
            (x1, y1, w1, h1) = cv2.boundingRect(np.array([shape[i:j]]))
            roi = image[y1:y1 + h1, x1:x1 + w1]
            roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
            #cv2.imshow("ROI", roi)
            #cv2.imshow("Image", clone)
            cv2.waitKey(0)

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = c.shape_to_np(shape)
        for (name, (i, j)) in c.FACIAL_LANDMARKS_IDXS.items():
            clone = image.copy()
            cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2)
            for (xxx, yyy) in shape[i:j]:
                #cv2.circle() yöntemi, herhangi bir görüntü üzerinde bir daire çizmek için kullanılır.
                cv2.circle(clone, (xxx, yyy), 1, (0, 0, 255), -1)
                
            # (x, y, w, h) --> X koordinatı, Y koordinatı, Genişlik, Yükseklik
            (x2, y2, w2, h2) = cv2.boundingRect(np.array([shape[i:j]]))
            roi = image[y2:y2 + h2, x2:x2 + w2]
            roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
            #cv2.imshow("ROI", roi)
            #cv2.imshow("Image", clone)
            cv2.waitKey(0)

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = d.shape_to_np(shape)
        for (name, (i, j)) in d.FACIAL_LANDMARKS_IDXS.items():
            clone = image.copy()
            cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2)
            for (xxxx, yyyy) in shape[i:j]:
                #cv2.circle() yöntemi, herhangi bir görüntü üzerinde bir daire çizmek için kullanılır.
                cv2.circle(clone, (xxxx, yyyy), 1, (0, 0, 255), -1)
                
            # (x, y, w, h) --> X koordinatı, Y koordinatı, Genişlik, Yükseklik
            (x3, y3, w3, h3) = cv2.boundingRect(np.array([shape[i:j]]))
            roi = image[y3:y3 + h3, x3:x3 + w3]
            roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
            #cv2.imshow("ROI", roi)
            #cv2.imshow("Image", clone)
            cv2.waitKey(0)


    y = y - 5
    w = w + 20
    h = h + 12

    x1 = x1 - 5
    y1 = y1 - 5
    w1 = w1 + 20
    h1 = h1 + 12

    x2 = x2 - 5
    y2 = y2 - 5
    w2 = w2 + 20
    h2 = h2 + 12

    x3 = x3 - 5
    y3 = y3 - 5
    w3 = w3 + 20
    h3 = h3 + 12

    if(y < y1):
        if(y2 + h2 > y3 + h3):
            if(x < x2):
                if(x1 + w1 > x3 + w3):
                    roi = image[y:y2 + h2 ,x:x1 + w1]
                roi = image[y:y2 + h2,x:x3 + w3]
            else:
                if(x1 + w1 > x3 + w3):
                    roi = image[y:y2 + h2,x2:x1 + w1]
                roi = image[y:y2 + h2,x2:x3 + w3]
        else:
            if(x < x2):
                if(x1 + w1 > x3 + w3):
                    roi = image[y:y3 + h3,x:x1 + w1]
                roi = image[y:y3 + h3,x:x3 + w3]
            else:
                if(x1 + w1 > x3 + w3):
                    roi = image[y:y3 + h3,x2:x1 + w1]
                roi = image[y:y3 + h3,x2:x3 + w3]
    else:
        if(y2 + h2 > y3 + h3):
            if(x < x2):
                if(x1 + w1 > x3 + w3):
                    roi = image[y1:y2 + h2,x:x1 + w1]
                roi = image[y1:y2 + h2,x:x3 + w3]
            else:
                if(x1 + w1 > x3 + w3):
                    roi = image[y1:y2 + h2,x2:x1 + w1]
                roi = image[y1:y2 + h2,x2:x3 + w3]
        else:
            if(x < x2):
                if(x1 + w1 > x3 + w3):
                    roi = image[y1:y3 + h3,x:x1 + w1]
                roi = image[y1:y3 + h3,x:x3 + w3]
            else:
                if(x1 + w1 > x3 + w3):
                    roi = image[y1:y3 + h3,x2:x1 + w1]
                roi = image[y1:y3 + h3,x2:x3 + w3]


    roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
    #cv2.imshow("ROI", roi)
    #os.chdir("C:/Users/irem/OneDrive/Masaüstü/imageProcessing")
    cv2.imwrite("TestResmi.jpg", roi)
#except:
#   print("Kesme hatasi (önceki resmi tekrarlardi)")

#output = visualize_facial_landmarks(image, shape)
#cv2.imshow("Image", output)

cv2.waitKey(0)