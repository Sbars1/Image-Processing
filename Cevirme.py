from skimage.util import random_noise
import cv2
from skimage.exposure import adjust_gamma
import os

os.getcwd()
os.chdir("C:/Users/user/OneDrive/Masaüstü/wp/wpKapali")
print(os.getcwd())
print(os.listdir())
ikiAcikResimler = os.listdir()
print(ikiAcikResimler)

num = 1

for x in ikiAcikResimler:
    os.chdir("C:/Users/user/OneDrive/Masaüstü/wp/wpKapali")
    image = cv2.imread(x)
    noisy_image= random_noise(image)

    image_bright = adjust_gamma(image, gamma=0.3,gain=1)
    image_dark = adjust_gamma(image, gamma=3,gain=1)

    """
    cv2.imshow('Rotated Image', noisy_image)
    cv2.imshow('acik', image_bright)
    cv2.imshow('koyu', image_dark)
    """

    os.chdir("C:/Users/user/OneDrive/Masaüstü/imageProcessing")
    cv2.imwrite("wpAydinlikKapali/wpAydinlikKapali" + str(num) + ".jpg", image_bright)
    cv2.imwrite("wpKoyuKapali/wpKoyuKapali" + str(num) + ".jpg", image_dark)
    num = num + 1

cv2.waitKey(0)
cv2.destroyAllWindows()