import json
import os
import random

import cv2
import keras
from keras_preprocessing.image import ImageDataGenerator
import numpy as np

from cnn.cnn_class import CNN
from config import Config
import matplotlib.pyplot as plt


def augment():
    path = Config.BASE_DIR + "/dataset_2/"

    datagen = ImageDataGenerator(
        rotation_range=15,
        zoom_range=(0.9, 1.1),
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.2,
        fill_mode='nearest'
    )

    for n in range(0, 10):
        os.system(f"mkdir dataset_augmented_2/{n}")

        train_pic = os.listdir(path + f"{n}")

        k = len(train_pic)

        train_x = np.ndarray(shape=(k, 120, 120, 1), dtype=float)

        m = 0
        for i, x in enumerate(train_pic):
            if not x.endswith(".png"):
                continue
            p = path + f"{n}/" + x
            img_data = cv2.imread(p, cv2.IMREAD_GRAYSCALE)

            img_data = img_data[:120, :]

            img_data = img_data.reshape(120, 120, 1)

            train_x[m] = np.array(img_data)

            m += 1

        i = 0
        for batch in datagen.flow(train_x, batch_size=11,
                                  save_to_dir=f'{Config.BASE_DIR}/dataset_augmented_2/{n}',
                                  save_format='PNG'):
            i += 1
            if i > k:
                break


def visualise_keras_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    for x in x_train:
        img_data = x
        img_data = np.invert(img_data)
        plt.imshow(img_data, interpolation='nearest')
        plt.show()


if __name__ == '__main__':
    # visualise_keras_mnist()

    # img_data = cv2.imread(Config.BASE_DIR + f"/dataset_augmented_2/3/_5_3410.PNG", cv2.IMREAD_GRAYSCALE)
    # # img_data = cv2.imread(Config.BASE_DIR + f"/test_data/9_351905.png", cv2.IMREAD_GRAYSCALE)
    #
    # img_data = img_data[:120, :]
    #
    # img_data = cv2.resize(img_data, (28, 28))
    # # # img_data = np.invert(img_data)
    #
    # plt.imshow(img_data, interpolation='nearest')
    # plt.show()

    # augment()

    cnn = CNN()
    cnn.train()

    # cnn.load('4.h5')
    # cnn.test_model()
    # prediction = cnn.predict('test_data/6_781211.png')
    #
    # a = {x: 0 for x in range(0, 10)}
    # for i, x in enumerate(prediction[0]):
    #     a[i] = float(x)
    # print(json.dumps(a, indent=4))
