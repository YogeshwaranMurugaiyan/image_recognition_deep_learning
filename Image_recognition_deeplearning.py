import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from google.colab import drive
drive.mount('/content/gdrive')

train1 = "/content/gdrive/MyDrive/Colab Notebooks/car_logo_detection/Trian"
validate1 = "/content/gdrive/MyDrive/Colab Notebooks/car_logo_detection/validation"
train_datagen=ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory(train1,target_size=(300,300),batch_size=128,class_mode='binary')
validate_generator=train_datagen.flow_from_directory(validate1,target_size=(300,300),batch_size=128,class_mode='binary')

train_datagen=ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory(train1,target_size=(300,300),batch_size=128,class_mode='binary')
validate_generator=train_datagen.flow_from_directory(validate1,target_size=(300,300),batch_size=128,class_mode='binary')

train_generator.class_indices

validate_generator.class_indices

model= tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(300, 300, 3)),
                                   tf.keras.layers.MaxPool2D(2,2),
                                   tf.keras.layers.Conv2D(34,(3,3),activation='relu'),
                                   tf.keras.layers.MaxPool2D(2,2),
                                   tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
                                   tf.keras.layers.MaxPool2D(2,2),
                                   tf.keras.layers.Flatten(),
                                   tf.keras.layers.Dense(1000,activation='relu'),
                                   tf.keras.layers.Dense(1,activation='sigmoid')])


from tensorflow.keras.optimizers import RMSprop

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])

history = model.fit(
      train_generator,
      steps_per_epoch=1,  
      epochs=18)

dir_path= "/content/gdrive/MyDrive/Colab Notebooks/car_logo_detection/validation/benz"
dir_path2= "/content/gdrive/MyDrive/Colab Notebooks/car_logo_detection/validation/BMW"

for i in os.listdir(dir_path):
  img=image.load_img(os.path.join(dir_path , i),target_size=(300,300))
  plt.imshow(img)
  plt.show()

  X=image.img_to_array(img)
  X= np.expand_dims(X,axis=0)
  #print(np.shape(X))
  #print (X)
  images= np.vstack([X])

  value=model.predict(images)
  if value== 0:
    print ('BMW')
  else:
    print ('BENZ')