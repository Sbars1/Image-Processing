import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from keras.layers import Layer
import os
import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from keras.utils.vis_utils import plot_model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import cv2 
import GozKesme
import time
from selenium import webdriver
from tkinter import *
from selenium.webdriver.common.keys import Keys
from keyboard import press
from selenium import webdriver
import time

# EGITIM 

train_dir='C:/Users/user/OneDrive/Masaüstü/Egitim'
test_dir='C:/Users/user/OneDrive/Masaüstü/Test'

print(os.listdir(train_dir))

liste_kapali = glob.glob(train_dir+'/Kapali/*')
liste_sagAcik = glob.glob(train_dir+'/sagAcik/*')
liste_solAcik = glob.glob(train_dir+'/solAcik/*')

len(liste_solAcik)


# load dataset
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2, )

train_gen = datagen.flow_from_directory(train_dir,
                                        target_size=(128, 128),
                                        batch_size=64,
                                        class_mode='categorical',
                                        color_mode='grayscale',
                                        shuffle=True,
                                        subset="training"
                                        )

valid_gen = datagen.flow_from_directory(train_dir,
                                        target_size=(128, 128),
                                        batch_size=64,
                                        class_mode='categorical',
                                        color_mode='grayscale',
                                        shuffle=True,
                                        subset="validation"
                                        )

test_gen = datagen.flow_from_directory(test_dir,
                                       target_size=(128, 128),
                                       batch_size=64,
                                       class_mode='categorical',
                                       color_mode='grayscale'
                                       )


first_model = Sequential()
first_model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=(128, 128, 1)))
first_model.add(MaxPooling2D(pool_size=(2, 2)))
first_model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
first_model.add(MaxPooling2D(pool_size=(2, 2)))
first_model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
first_model.add(MaxPooling2D(pool_size=(2, 2)))
first_model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
first_model.add(MaxPooling2D(pool_size=(2, 2)))
first_model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
first_model.add(MaxPooling2D(pool_size=(2, 2)))
first_model.add(Flatten())
first_model.add(Dense(256, activation='relu'))
first_model.add(Dropout(0.5))
first_model.add(Dense(256, activation='relu'))
first_model.add(Dropout(0.5))
first_model.add(Dense(3, activation='softmax'))

first_model.summary()

first_model.compile(loss=categorical_crossentropy,
                    optimizer=Adam(lr=0.001),
                    metrics=['accuracy'])

history = first_model.fit(train_gen,
                          batch_size=64,
                          epochs=15,
                          verbose=1,
                          validation_data=valid_gen
                          )

first_model.save('my_first_xray_model.h5')

first_model.save('/content/drive/My Drive/Teknotip_Vision/my_first_xray_model.h5')

loss, accuracy = first_model.evaluate(test_gen)
print("Test: accuracy = %f  ;  loss = %f " % (accuracy, loss))


class Google:
    def __init__(self):
        self.browserProfile = webdriver.ChromeOptions()
        self.browserProfile.add_experimental_option('prefs', {'intl.accept_languages':'en,en_US'})
        self.browser = webdriver.Chrome('chromedriver.exe', chrome_options=self.browserProfile)
        self.browser.get("https://www.google.com/search?q=translate&oq=trans&aqs=chrome.1.69i57j69i59l2.3729j0j15&sourceid=chrome&ie=UTF-8")

    def getFollowers(self):      

        action = webdriver.ActionChains(self.browser)
        time.sleep(2)
        self.browser.find_elements_by_class_name("tw-ta tw-text-large q8U8x goog-textarea").click() 
        action.key_down(Keys.ENTER).key_up(Keys.ENTER).perform()

    def solAcik(self):
        self.browser.execute_script('window.scrollTo(0, document.body.scrollHeight);')
    
    def sagAcik(self):
        self.browser.execute_script('window.scrollTo(0, document.body.scrollHeight - 1000000);')

google = Google()


"""
#VIDEO DENEME

kamera=cv2.VideoCapture(0)
ret, goruntu=kamera.read()
i = 0
while True:
    i = i + 1
    ret, goruntu=kamera.read()
    cv2.imwrite("kameraGoruntuleri/kameraGoruntusu.jpg",goruntu)
    cv2.imshow("Ozengineer", goruntu)

    if cv2.waitKey(30) & 0xFF==('q'):
        break

    first_model = load_model('my_first_xray_model.h5')

    classes = ['kapali', 'sagAcik', 'solAcik']

    kameraGoruntusu='kameraGoruntuleri/kameraGoruntusu.jpg'
    if(i == 1):
        GozKesme.GozKesmeFonk("indir2.jpg")
    if(i != 1):
        GozKesme.GozKesmeFonk(kameraGoruntusu)
    testResmi = 'TestResmi.jpg'

    image = load_img(testResmi, target_size=(128, 128), color_mode="grayscale")
    image = img_to_array(image) / 255
    image = np.expand_dims(image, axis=0)
    print('shape:', image.shape)

    preds = first_model.predict(image)

    print(preds)

    print(preds.argmax())
    print(classes[preds.argmax()])

    if(classes[preds.argmax()] == "solAcik"):
        time.sleep(1)
        google.solAcik()
    if(classes[preds.argmax()] == "sagAcik"):
        time.sleep(1)
        google.sagAcik()
    if(classes[preds.argmax()] == "kapali"):  
        time.sleep(1)
        google.getFollowers()
"""


#TEK RESIM DENEME

GozKesme.GozKesmeFonk("indir2.jpg")
testResmi = 'TestResmi.jpg'

first_model = load_model('my_first_xray_model.h5')
classes = ['kapali', 'sagAcik', 'solAcik']

image = load_img(testResmi, target_size=(128, 128), color_mode="grayscale")
image = img_to_array(image) / 255
image = np.expand_dims(image, axis=0)
print('shape:', image.shape)

preds = first_model.predict(image)

print(preds)

print(preds.argmax())
print(classes[preds.argmax()])

if(classes[preds.argmax()] == "solAcik"):
    time.sleep(1)
    google.solAcik()
if(classes[preds.argmax()] == "sagAcik"):
    time.sleep(1)
    google.sagAcik()
if(classes[preds.argmax()] == "kapali"):  
    time.sleep(1)
    google.getFollowers()



#GRAFIK GOSTERIMI

# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()
