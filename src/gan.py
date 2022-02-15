import numpy as np
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Conv2DTranspose, LeakyReLU, BatchNormalization, Activation, Reshape, MaxPool2D


class CGAN:
    def build_generator(width, height, depth, channels=3, inputDim=100, outputDim=128):
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
            Conv2DTranspose(32, (5, 5), strides=(2, 2), padding="same"),
            Activation("relu"),
            BatchNormalization(axis=chanDim),
            Conv2DTranspose(channels, (5, 5), strides=(2, 2), padding="same"),
            MaxPool2D(pool_size=(4,4)),
            Activation("tanh"),
        ])
        return model

    def build_discriminator(width, height, depth, alpha=0.2):
        inputShape = (height, width, depth)
        model = Sequential([
            Conv2D(32, (5, 5), padding="same", strides=(2, 2), input_shape=inputShape),
            LeakyReLU(alpha=alpha),
            Conv2D(64, (5, 5), padding="same", strides=(2, 2)),
            LeakyReLU(alpha=alpha),
            Flatten(),
            Dense(128),
            LeakyReLU(alpha=alpha),
            Dense(1),
            Activation("sigmoid"),
        ])
        return model

