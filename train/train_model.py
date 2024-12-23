import os
import sys
models_dir = os.path.join(os.path.dirname(__file__), '../models')
sys.path.append(models_dir)
from models.lenet5 import LeNet5
from dataset.dataset_loader import load_and_preprocess_data

def train_model():

    x_train, y_train, x_test, y_test = load_and_preprocess_data()
    model = LeNet5(input_shape=(32, 32, 1))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))
    model.save('../saved_models/lenet5.h5')

if __name__ == "__main__":
    train_model()
