import os
import cv2
import keras
from keras_preprocessing.image import ImageDataGenerator
from numpy import array, ndarray
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K, layers
# from keras.utils import print_summary
from typing import List
from config import Config
import matplotlib.pyplot as plt


class CNN:
    def train(self, save_name: str = None):
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

        batch_size = 128
        self.num_classes = 10
        epochs = 10

        img_rows, img_cols = 28, 28

        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            self.input_shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            self.input_shape = (img_rows, img_cols, 1)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        x_train /= 255
        x_test /= 255

        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')

        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        model = self.__create_model()

        # print_summary(model)

        aug = ImageDataGenerator(
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode="nearest"
        )

        h = model.fit_generator(aug.flow(x_train, y_train, batch_size=batch_size),
                                epochs=epochs,
                                steps_per_epoch=len(x_train) // batch_size)

        score = model.evaluate(x_test, y_test, verbose=0)

        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        # if save_name is not None:
        model.save("/Users/a1/univ/S1/diploma/cnn/models/1.h5")

        return

        # print_summary(model)

        plt.plot(h.history['acc'])
        plt.plot(h.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

        # Plot training & validation loss values
        plt.plot(h.history['loss'])
        plt.plot(h.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

    def load(self, model_name: str):
        self.model = load_model(Config.BASE_DIR + '/cnn/models/' + model_name)

    def predict(self, file_path: str) -> List[float]:
        img_data = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (64, 64))

        img_data = img_data.reshape(1, 64, 64, 1)

        img_data = img_data.astype('float32')

        img_data /= 255

        f = self.model.predict(img_data)

        return f

    def __create_model(self):
        model = keras.Sequential(
            [
                keras.Input(shape=self.input_shape),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(self.num_classes, activation="softmax"),
            ]
        )
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model

        model.add(Conv2D(8, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=self.input_shape))

        # model.add(Conv2D(16, (3, 3),
        #                  activation='relu'
        #                  ))
        # model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3),
                         activation='relu'
                         ))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3),
                         activation='relu'
                         ))

        # model.add(Conv2D(128, (3, 3),
        #                  activation='relu'
        #                  ))
        #
        # model.add(Conv2D(128, (3, 3),
        #                  activation='relu'
        #                  ))
        # model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())

        # model.add(Dense(1024,
        #                 activation='relu'
        #                 ))
        # model.add(Dense(512,
        #                 activation='relu'
        #                 ))
        model.add(Dense(128,
                        activation='relu'
                        ))
        model.add(Dropout(0.5))
        model.add(Dense(32,
                        activation='relu'
                        ))
        model.add(Dense(self.num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                      metrics=['acc'])

        return model

    def __load_data(self, path):
        train_pic = os.listdir(path)

        data = ndarray(shape=(len(train_pic), 64, 64), dtype=float)
        labels = array(train_pic)

        for i, x in enumerate(train_pic):
            img_data = cv2.imread(path + '\\' + x, cv2.IMREAD_GRAYSCALE)
            img_data = cv2.resize(img_data, (64, 64))
            data[i] = array(img_data)

            if 'rect' in x:
                labels[i] = 0
            else:
                labels[i] = 1

        return data, labels
