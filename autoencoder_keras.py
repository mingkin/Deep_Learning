# -*- coding: utf-8 -*-
"""
# Author  : Ming
# File    : {NAME}.py
# Time    : 2019/4/19 0019 下午 5:42
"""

from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, load_model

'图片去噪'

'加载数据'
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))


'加噪音数据'
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)


def display(x_test_noisy):
    n = 10
    plt.figure(figsize=(20, 2))
    for i in range(n):
        ax = plt.subplot(1, n, i + 1)
        plt.imshow(x_test_noisy[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

def autoencoder_model(x_train_noisy, x_train):
    input_img = Input(shape=(28, 28, 1,)) # N * 28 * 28 * 1
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(input_img) # 28 * 28 * 32
    x = MaxPooling2D((2, 2), padding='same')(x) # 14 * 14 * 32
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x) # 14 * 14 * 32
    encoded = MaxPooling2D((2, 2), padding='same')(x) # 7 * 7 * 32


    x = Conv2D(32, (3, 3), padding='same', activation='relu')(encoded) # 7 * 7 * 32
    x = UpSampling2D((2, 2))(x) # 14 * 14 * 32
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x) # 14 * 14 * 32
    x = UpSampling2D((2, 2))(x) # 28 * 28 * 32
    decoded = Conv2D(1, (3, 3), padding='same', activation='sigmoid')(x) # 28

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    autoencoder.fit(x_train_noisy, x_train,
                    epochs=10,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(x_test_noisy, x_test))
    autoencoder.save('autoencoder.h5')




def test_autoencoder(x_test_noisy):
    autoencoder = load_model('autoencoder.h5')

    decoded_imgs = autoencoder.predict(x_test_noisy)

    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test_noisy[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


if __name__ == '__main__':
    autoencoder_model(x_train_noisy, x_train)
    test_autoencoder(x_test_noisy)
