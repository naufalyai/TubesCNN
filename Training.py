from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization, GlobalAveragePooling2D
from keras import backend as K
import keras

img_width, img_height = 48, 48
train_data_dir = './data/train'
validation_data_dir = './data/validation'
nb_train_samples = 28709
nb_validation_samples = 3589
epochs = 1
batch_size = 30

# check image format in keras backend
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
# define VGG16 Architecture , we add batch normalization after pooling layer and we change 2 layer of 4096 dense into Global Average Pooling
model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=input_shape,padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), input_shape=input_shape,padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3),strides=2))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))

model.add(Conv2D(128, (3, 3), input_shape=input_shape,padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), input_shape=input_shape,padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3),strides=2))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))

model.add(Conv2D(256, (3, 3), input_shape=input_shape,padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3), input_shape=input_shape,padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3), input_shape=input_shape,padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3),strides=2))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))

model.add(Conv2D(256, (3, 3), input_shape=input_shape,padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3), input_shape=input_shape,padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3), input_shape=input_shape,padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3),strides=2))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))

model.add(GlobalAveragePooling2D())
model.add(Dense(7))
model.add(Activation('softmax'))
model.load_weights('5DesCNNvgg16v4.h5')
opt = keras.optimizers.nadam(lr = 0.0001)


model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest',
    horizontal_flip='true'
)

# this is the augmentation configuration we will use for validating:
test_datagen = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest',
    horizontal_flip='true'

)
# Get summary from architecture
model.summary()

# Generate train data from folder train
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

# Generate validation data from folder validation
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

# train
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save_weights('5DesCNNvgg16v5.h5')
model.save('5DesCNNmodelvgg16v5.h5')