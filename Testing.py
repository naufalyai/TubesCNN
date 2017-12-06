from keras.models import load_model
import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization,GlobalAveragePooling2D
from keras import backend as K
import keras
from PIL import Image
import glob
import _compat_pickle as pickle
import pandas



def create_model():
    img_width, img_height = 48, 48
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)
    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=input_shape, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), input_shape=input_shape, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
    model.add(
        BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                           gamma_initializer='ones', moving_mean_initializer='zeros',
                           moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                           beta_constraint=None, gamma_constraint=None))

    model.add(Conv2D(128, (3, 3), input_shape=input_shape, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), input_shape=input_shape, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
    model.add(
        BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                           gamma_initializer='ones', moving_mean_initializer='zeros',
                           moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                           beta_constraint=None, gamma_constraint=None))

    model.add(Conv2D(256, (3, 3), input_shape=input_shape, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), input_shape=input_shape, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), input_shape=input_shape, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
    model.add(
        BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                           gamma_initializer='ones', moving_mean_initializer='zeros',
                           moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                           beta_constraint=None, gamma_constraint=None))

    model.add(Conv2D(256, (3, 3), input_shape=input_shape, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), input_shape=input_shape, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), input_shape=input_shape, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
    model.add(
        BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                           gamma_initializer='ones', moving_mean_initializer='zeros',
                           moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                           beta_constraint=None, gamma_constraint=None))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(7))
    model.add(Activation('softmax'))
    return model

def loadLabel(filename):
    with open(filename,'r') as f:
        for line in f:
            line = line.split(', ')
            target = [int(i) for i in line]
    return target


def load_trained_model(weights_path):
   model = create_model()
   model.load_weights(weights_path)
   return model

model = load_trained_model('5DesCNNvgg16v4.h5')
kelas = ['Angry','Disgust','Fear','Happy','Sad','Surprised','Neutral']
opt = keras.optimizers.nadam(lr = 0.01)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
predicted = []
hasil = []
predictedLabel = 1
target = loadLabel('target.txt')
predictedTrue = np.zeros(6)
path = './data/test'
for img_path in glob.glob(path+'\*.jpg'):
    img = Image.open(img_path)
    temp = np.zeros((1,48,48,3))
    img = np.array(img)
    temp[0,:,:,0] = img
    temp[0,:,:,1] = img
    temp[0,:,:,2] = img
    classes =model.predict(temp)
    label = np.argmax(classes)
    predicted.append(label)
    hasil.append([label,kelas[label]])

temp = np.array(hasil)

accuracy = np.sum(np.array(predicted) == np.array(target))/len(target)

print("accuracy : ",accuracy)

df = pandas.DataFrame(hasil,columns=["Label","Emotion"])
df.to_excel('HasilLabel.xlsx')
