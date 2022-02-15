import numpy as np
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD

from tensorflow.keras.layers import Conv2DTranspose, LeakyReLU, BatchNormalization, Activation, Reshape, MaxPool2D


class CGAN:
    def generator(width, height, depth, channels=3, inputDim=100, outputDim=128):
        model = Sequential()
        inputShape = (width, height, depth)
        chanDim = 3
        model = Sequential([
            Dense(input_dim=inputDim, units=outputDim),
            Activation("relu"),
            BatchNormalization(),
            Dense(width * height * depth),
            Activation("relu"),
            BatchNormalization(),
            Reshape(inputShape),
            Conv2DTranspose(32, (5, 5), strides=(8, 2), padding="valid"),
            Activation("relu"),
            BatchNormalization(axis=chanDim),
            Conv2DTranspose(channels, (5, 5), strides=(4, 4), padding="same"),
            Activation("tanh"),
        ])
        return model

    def discriminator(x, y, z, alpha=0.2):
        inputShape = (x, y, z)
        model = Sequential([
            Conv2D(64, (3, 3), padding="same", strides=(2, 2), input_shape=inputShape),
            LeakyReLU(alpha=alpha),
            Conv2D(128, (3, 3), padding="same", strides=(2, 2)),
            LeakyReLU(alpha=alpha),
            Conv2D(128, (3, 3), padding="same", strides=(2, 2)),
            LeakyReLU(alpha=alpha),
            Conv2D(256, (5, 5), padding="same", strides=(2, 2)),
            LeakyReLU(alpha=alpha),
            Flatten(),
            Dropout(0.4),
            Dense(1, activation="sigmoid"),
        ])
        return model

    def gen_real(dataset, n_samples):
        seed = np.randint(0, dataset.shape[0], n_samples)
        x = dataset[seed]
        y = np.ones((n_samples, 1))
        return x, y
    
    def gen_noise(x, y, z, n_samples):
        seed = np.rand(x * y * z * n_samples)
        seed = -1 + seed * 2
        X = seed.reshape((n_samples, x, y, z))
        y = np.zeros((n_samples, 1))
        return X, y
