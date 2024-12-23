import tensorflow as tf

class LeNet5(tf.keras.Model):
    def __init__(self, input_shape=(32, 32, 1)):
        super(LeNet5, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=6, kernel_size=5, padding='same', activation=tf.nn.tanh, input_shape=input_shape)
        self.pool1 = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)
        self.conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=5, padding='valid', activation=tf.nn.tanh)
        self.pool2 = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)
        self.conv3 = tf.keras.layers.Conv2D(filters=120, kernel_size=5, activation=tf.nn.tanh)

        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(units=84, activation=tf.nn.tanh)
        self.fc2 = tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return self.fc2(x)