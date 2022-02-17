from xml.sax.xmlreader import XMLReader
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
from mapper import RetinalImages
import tensorflow as tf
from tensorflow.keras.optimizers import Adam


class CGAN:
    def generator(res, inputShape, latent=100, outputDim=128, alpha=0.2):
        print(inputShape[0] * inputShape[1] * outputDim)
        model = Sequential([
            Dense(units=int(inputShape[0]* res * inputShape[1] * res * outputDim), input_dim=latent),
            LeakyReLU(alpha=alpha),
            Reshape((int(inputShape[0] * res), int(inputShape[1] * res), outputDim)),
            BatchNormalization(),
            Conv2DTranspose(64, (5, 5), padding="same", strides=(5, 5)),
            LeakyReLU(alpha=alpha),
            BatchNormalization(),
            Conv2DTranspose(32, (5, 5), padding="same", strides=(5, 5)),
            LeakyReLU(alpha=alpha),
            BatchNormalization(),
            Conv2DTranspose(32, (4, 4), padding="same", strides=(2, 2)),
            LeakyReLU(alpha=alpha),
            Conv2DTranspose(16, (4, 4), padding="same", strides=(2, 2)),
            LeakyReLU(alpha=alpha),
            Conv2D(3, (int(inputShape[0] * res), int(inputShape[1] * res)), activation='sigmoid', padding="same")
        ])

        return model

    def discriminator(inputShape, alpha=0.2):
        
        model = Sequential([
            Conv2D(16, (3, 3), padding="same", strides=(2, 2), input_shape=inputShape),
            LeakyReLU(alpha=alpha),
            Conv2D(64, (3, 3), padding="same", strides=(2, 2)),
            LeakyReLU(alpha=alpha),
            Conv2D(64, (3, 3), padding="same", strides=(2, 2)),
            LeakyReLU(alpha=alpha),
            Conv2D(128, (5, 5), padding="same", strides=(2, 2)),
            LeakyReLU(alpha=alpha),
            Flatten(),
            Dropout(0.4),
            Dense(1, activation="sigmoid"),
        ])
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5), metrics=['accuracy'])

        return model

    def gan(g, d):
        d.trainable = False
        model = Sequential([
            g,
            d
        ])
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))
        return model


    def gen_real(dataset : mapper.RetinalImages.retinal_data, n_samples):
        seed = randint(0, high=int(RetinalImages.retinal_data.samples), size=n_samples)
        # print(seed)
        x = RetinalImages.retinal_data._get_batches_of_transformed_samples(index_array=seed)
        return x[0], x[1]
    
    def gen_gauss(latent_dim, n):
        vector = (rand(latent_dim*n)).reshape(n, latent_dim)
        return vector

    def gen_synthetic(self, generator, n_samples):
        input = self.gen_gauss(100, n_samples)
        prediction = generator.predict(input)
        label = np.zeros((n_samples, 1))
        return prediction, label

    