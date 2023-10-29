from keras.layers import Dense
from keras import (
    Input,
    Model
)
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

def generate_auto_encoder(dimensions: int = 32, image_shape: Tuple[int] = (3072, )) -> Model:
    """Builds an Autoencoder object given the encoded image dimensions
    and the original image shapes

    Arguments:
        dimensions (int): The size of our encoded image representations
        image_shape (tuple(int)): The size and shape of our input images
    
    Returns:
        autoencoder (Model): A full autoencoder representation
    """
    # Original Input Image, will be subsequently fed into our encoding layer
    inputs = Input(shape=image_shape)

    # The single encoding layer which will output the encoded (compressed)
    # representation of our input image
    encoded_layer = Dense(dimensions, activation='relu')(inputs)

    # The single decoding layer which will output a reconstructed image
    # from the compressed output of the encoded_layer
    decoded_layer = Dense(image_shape[0], activation='sigmoid')(encoded_layer)

    # The full autoencoder
    autoencoder = Model(inputs=inputs, outputs=decoded_layer)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder

def main():

    # Some standard settings
    epochs     = 50  # number of epochs to run
    batch_size = 256 # training batch size
    n_digits   = 5   # number of images to show on output graph

    # loading in our mnist dataset - https://keras.io/api/datasets/mnist/
    (x_train, _), (x_test, _) = mnist.load_data()

    # save original shape of image before reshaping array
    x_train_shape_orig = x_train.shape

    # normalize the image values between 0 and 1 and then flatten them
    # into an input vector
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    # generate our untrained autoencoder
    image_shape = (x_train.shape[1],)
    autoencoder = generate_auto_encoder(image_shape=image_shape)

    # fit/train our autoencoder to the training data
    autoencoder.fit(
        x_train, x_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(x_test, x_test)
    )
    
    # Using our autoencoder, reconstruct the test data from
    # the mnist dataset
    decoded_imgs = autoencoder.predict(x_test)

    # display the outupts so we can see the original vs reconstructed
    # images
    plt.figure(figsize=(20, 4))
    counter = 0
    for i in range(n_digits):
        # Original
        ax = plt.subplot(n_digits, 2, counter + 1)
        plt.imshow(x_test[i].reshape(x_train_shape_orig[1], x_train_shape_orig[2]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if (counter == 0):
            plt.title("Original")

        # Reconstructed
        ax = plt.subplot(n_digits, 2, counter + 2)
        plt.imshow(decoded_imgs[i].reshape(x_train_shape_orig[1], x_train_shape_orig[2]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if (counter == 0):
            plt.title("Reconstructed")
        counter += 2
    plt.show()

if __name__ == "__main__":
    main()