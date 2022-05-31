import cv2
import numpy as np
import os
import imutils

os.getcwd()
os.chdir("C:/Users/user/OneDrive/Masaüstü/veriseti/solgoz")
print(os.getcwd())
print(os.listdir())
ikiAcikResimler = os.listdir()
print(ikiAcikResimler)

num = 1

for x in ikiAcikResimler:
  os.chdir("C:/Users/user/OneDrive/Masaüstü/veriseti/solgoz")
  image = cv2.imread(x)

  scale_percent = 250 # percent of original size
  width = int(image.shape[1] * scale_percent / 100)
  height = int(image.shape[0] * scale_percent / 100)
  dim = (width, height)
    
  # resize image
  resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

  #cv2.imshow('resized', resized)

  im1 = cv2.blur(resized, (10, 10))
  #cv2.imshow('blur', im1)  

  (h, w) = resized.shape[:2]
  center = (w / 2, h / 2)
  angle = 30
  scale = 1
  M = cv2.getRotationMatrix2D(center, angle, scale)
  im2 = cv2.warpAffine(resized, M, (w, h))
  #cv2.imshow('Rotated', im2) 

  (h, w) = resized.shape[:2]
  center = (w / 2, h / 2)
  angle = -30
  scale = 1
  M = cv2.getRotationMatrix2D(center, angle, scale)
  im3 = cv2.warpAffine(resized, M, (w, h))
  #cv2.imshow('Rotated Image', im3)  

  os.chdir("C:/Users/user/OneDrive/Masaüstü/imageProcessing")
  cv2.imwrite("solBulanik/solBulanik" + str(num) + ".jpg", im1)
  

  num = num + 1


cv2.waitKey(0)
cv2.destroyAllWindows()