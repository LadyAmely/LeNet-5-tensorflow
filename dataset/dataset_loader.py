import tensorflow as tf
from tensorflow.keras.datasets import mnist

def load_and_preprocess_data():

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train[..., tf.newaxis] / 255.0
    x_test = x_test[..., tf.newaxis] / 255.0

    x_train = tf.image.resize_with_pad(x_train, 32, 32).numpy()
    x_test = tf.image.resize_with_pad(x_test, 32, 32).numpy()

    return x_train, y_train, x_test, y_test
