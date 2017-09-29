from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.preprocessing import image
import numpy as np
import os





def cls():
    os.system('cls' if os.name=='nt' else 'clear')                 # Функция очистки экрана. Информационные сообщения
                                                                   # CUDA захламляют экран, после инициализации их желательно
img_width, img_height = 150, 150                                   # убрать и работать в чистой консоли

train_data_dir = 'D:/data/train'
validation_data_dir = 'D:/data/validation'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 50
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
#model.add(Dropout(0.5))                                      # При обучении раскоментировать. На этапе проверки dropout
model.add(Dense(1))                                           # не нужен, т.к.  каждый раз для одного изображения
model.add(Activation('sigmoid'))                              # возвращаются различные вероятности


model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training

'''
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save_weights('weights.h5')
'''

model.load_weights('weights.h5')

cls()

while True:
    try:
        img_path = str(raw_input("jpg file path: "))
        img = image.load_img(img_path, target_size=(150, 150))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        classes = model.predict(x, 16, 0)
        ts = classes.tostring()
        tx = np.fromstring(ts, dtype=int)
        #print (tx)
        #print(str(classes))

        if str(tx) == '[0]':
            print("Chekaem " + str(img_path))
            print('')
            print("Koteika")
            print('')

        elif str(tx) == '[1065353216]':
            print("Chekaem " + str(img_path))
            print('')
            print("Sobaka spidozniy")
            print('')
        else:
            print("X3 4e eto takoe")
    except:
        print("4eto keknulos'")
