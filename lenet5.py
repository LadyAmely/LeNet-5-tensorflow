import tensorflow as tf

class LeNet5:

    def __init__(self, input_shape=(32, 32, 1)):
        self.input_shape = input_shape

        self.conv1 = tf.keras.layers.Conv2D(filters=6, kernel_size=5, padding='same', activation=tf.nn.tanh)
        self.pool1 = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)
        self.conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=5, padding='same', activation=tf.nn.tanh)
        self.pool2 = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)
        self.conv3 = tf.keras.layers.Conv2D(filters=120, kernel_size=5, padding='same', activation=tf.nn.tanh)

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
        outputs = self.fc2(x)

        return outputs

input_tensor = tf.keras.Input(shape=(32, 32, 1))
model_instance = LeNet5()
outputs = model_instance.call(input_tensor)
model = tf.keras.Model(inputs=input_tensor, outputs=outputs)

model.summary()
