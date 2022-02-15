import numpy as np
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from numpy.random import rand
from numpy.random import randint
from tensorflow.keras.layers import Conv2DTranspose, LeakyReLU, BatchNormalization, Activation, Reshape, MaxPool2D
import mapper
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

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

    def discriminator(inputShape, alpha=0.2):
        
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
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])

        return model

    def gen_real(dataset : mapper.RetinalImages.retinal_data, n_samples):
        # print(dataset.samples)

        seed = randint(0, high=int(dataset.samples), size=n_samples)
        print(seed)
        # dataset.index_array = seed
        # x = np.zeros((len(seed),) + dataset.image_shape, dtype=dataset.dtype)
        x = dataset._get_batches_of_transformed_samples(index_array=seed)
        x = tf.unstack(x[0], axis=0)
        # print(x)
        for i in x:
            print(i.shape)
        x = np.reshape(1,-1)
        y = np.ones((n_samples, 1))
        print(y,shape)
        # print(x.shape)
        return x, y
    
    def gen_noise(x, y, z, n_samples):
        seed = rand(x * y * z * n_samples)
        seed = -1 + seed * 2
        X = seed.reshape((n_samples, x, y, z))
        y = np.zeros((n_samples, 1))
        return X, y
