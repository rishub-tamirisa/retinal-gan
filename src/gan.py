import numpy as np
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Conv2DTranspose, LeakyReLU, BatchNormalization, Activation, Reshape


class CGAN:
    def build_generator(width, height, depth, channels=1, inputDim=100, outputDim=512):
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
            Conv2DTranspose(20, (5, 5), strides=(2, 2), padding="same"),
            Activation("relu"),
            BatchNormalization(axis=chanDim),
            Conv2DTranspose(channels, (5, 5), strides=(2, 2), padding="same"),
            Activation("tanh"),
        ])
        return model

    def build_discriminator(width, height, depth, alpha=0.2):
        inputShape = (height, width, depth)
        model = Sequential([
            Conv2D(10, (5, 5), padding="same", strides=(2, 2), input_shape=inputShape),
            LeakyReLU(alpha=alpha),
            Conv2D(20, (5, 5), padding="same", strides=(2, 2)),
            LeakyReLU(alpha=alpha),
            Flatten(),
            Dense(512),
            LeakyReLU(alpha=alpha),
            Dense(1),
            Activation("sigmoid"),
        ])
        return model

