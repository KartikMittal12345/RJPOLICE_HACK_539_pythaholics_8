
# DEEP FAKE DETECTION CODE USING MACHINE LEARNING AND DEEP LEARNING with Jarvis***




# FIRST WE IMPORT PYTHON LIBRARY RELATED TO DEEP LEARNING AND DEPP LEARNING

import os
import cv2
import pyttsx3  # THIS LIBRARY IS MAINLY USED TO RUN VOICE ENGINE FOR JARVIS
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf # MOST IMPORTANT LIBRARY FOR DEEP LEARNING FOR DETECTION
import sklearn
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential # FOR GENERATE A  DEEP LEARNING SEQUENTIAL MODEL WE USE THIS LIBRARY
from keras.layers import Dense,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam,Adagrad,Adadelta,RMSprop
from playsound import playsound  # THE PLAYSOUND LIBARY IS BASICALLY USED FOR GENERATE A SIREN VOICE FOR DEEP FAKE


# GIVEN THE PATH OF TRAIN DIRECTORY REAL AND FAKE
train_data_dir = "C:\\Users\\sony\\Downloads\\DeepFakeimages\\real_and_fake_face"

# GIVEN THE PATH OF TEST DIRECTORY REAL AND FAKE
test_data_dir = "C:\\Users\\sony\\Downloads\\DeepFakeimages\\real_and_fake_face_detection"

# DATA GENERATOR FOR IMAGES IN WHICH DATA IS SPLIT IN VALIDATION AND RESCALE THE IMAGE
datagen  = ImageDataGenerator(
    validation_split=0.40,
    rescale = 1./255
)

# GENERATE THE TRAIN GENERATOR FOR OUR DEEP LEARNING  MODEL
train_generator = datagen.flow_from_directory(
    train_data_dir,
    color_mode = "grayscale",
    target_size=(48,48),
    batch_size=32,# LOWER THE BATCH SIZE HIGHER THE ACCURACY WE USE THIS PARAMETER AFTER HYPERPARAMETER TUNNING
    class_mode="categorical",
    subset="training"
)

# GENERATE THE VALIDATION GENERATOR FOR OUR DEEP LEARNING MODEL FOR VALIDATE IMAGES
validation_generator = datagen.flow_from_directory(
    test_data_dir,
    color_mode = "grayscale",
    target_size=(48,48),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

# THERE IS BASICALLY TO CLASS LABELS : DEEP FAKE IAMGE,REAL FACE IMAGE
class_label = ['Deep Fake Image','Real Face Iamge']

# PRINTING THE LABELS 
print(f'Total Number of Class Label {(class_label)}')

# PRINTING THE LENGTH OF OUR LABELS
print(f'Total Length of Label {len(class_label)}')


# GENERATING THE IMAGE IN MATRIX FORMAT USING TRAIN GENERATOR

img,label = train_generator.__next__()

# PRINTING THE IMAGE IN MATRIX ACCORDING TO SIZE OF IMAGE
print(f'Images In Mtarix Format{(img)}')

# CREATING A DEEP LEARNING CNN SEQUENTIAL MODEL

model = Sequential()

# ADDING CONVULATION LAYER IS CORE BUILDING OF CNN WITH SOME NEURON LAYERS , DEFINE THE INPUT SHAPE OF IMAGE AND USE ONE OF THE MOST IMPORTANT ACTIVATION FUNCTION RELU

model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(48,48,1)))

# ADDING OTHER NEURON LAYER WITH SOME KERNEL SIZE WITH ACTIVATION FUNCTION RELU
model.add(Conv2D(64,kernel_size=(3,3),activation="relu"))
# THE MAXPOOLING LAYER IS UDE FOR DOWNSAMPLE ALONG ITS HEIGHT AND WIDTH
model.add(MaxPooling2D(pool_size=(2,2)))

# RANDOMLY DEACTIVATING A PORTION OF INPUT UNITS AN UPDATE TRAINING
model.add(Dropout(0.1))

model.add(Conv2D(128,kernel_size=(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(256,kernel_size=(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# CONVERTING THE MULTIDIMENSIONAL INPUT INTO ONE DIMENSION INPUT
model.add(Flatten())

# TO CLASSIFY IMAGE BASED ON OUTPUT FROM CONVOLUTIONAL LAYERS
model.add(Dense(512,activation='relu'))

model.add(Dense(2,activation="softmax"))

# GENRATE THE SUMMARY OF MODEL
model.summary()

import pyttsx3

# THE SAPI5 VOICE ARE USED FOR GENERATING AI VOICE OF MALE AND FEAMALE
engine = pyttsx3.init('sapi5')

# USING THIS LINE OF CODE WE GET PROPERTY OF ALL VOICES
voices = engine.getProperty('voices')

# PRINTING THE VOICE ID 
print(voices[1].id)

# WE ARE SET THE PROPERTY OF VOICES INSIDE INIT
engine.setProperty('voices',voices[1].id)


# SPEAK AUDIO FUNCTION IS USE FOR JARVIS SPEAK***
def speak(audio):

    engine.say(audio)
    engine.runAndWait()



if __name__ == '__main__':


    speak("HI Myself Jarvis! What Can i Help You")


    speak('Hi  Everyone Welcome To Rajasthan Police Heckathon')

    speak('In This Hecathon The Team Pythaholics is working on Deep Fake Images Identification')

    speak('I am Very Thankful To Rajasthan Police For Conducting This Hackathon')


# Compilation of model by using optimizer adam and categorical cross entropy
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# fit our model for training
history=model.fit(train_generator,steps_per_epoch=25,epochs=30,validation_data=validation_generator,batch_size=32)


# Testing our Model
frame = cv2.imread(r"C:\Users\sony\Downloads\DeepFakeimages\real_and_fake_face\training_fake\easy_46_1100.jpg")

gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
labels_dict = {0:'Deep_Fake_Image',1:'Real_Image'}
gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
resized = cv2.resize(gray,(48,48))
normalize = resized/255.0
reshaped = np.reshape(normalize,(1,48,48,1))
# prediction of model
result = model.predict(reshaped)
label = np.argmax(result,axis=1)[0]

print(label)

while (True):
   

  if __name__ == "__main__":
    # speak the label that are predict by deep learning model
    if (label==0):
      # playsound siren
      playsound("C:\\Users\\sony\\Downloads\\emergency-alarm-with-reverb-29431.mp3")

      print('Deep fake Image Detect')
      speak("Deep Fake Image Detect!")
      

    elif (label==1):
    
       print('real face Image detect')
       speak("Real Face Image Detect")



## model losses and accuracy for val_loss and val_accc using plot

history_df = pd.DataFrame(history.history)
history_df.loc[:,['loss','val_loss']].plot()

history_df.loc[:,['accuracy','val_accuracy']].plot()


