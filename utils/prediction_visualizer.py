import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def load_saved_model(saved_model_path):

    return tf.keras.layers.TFSMLayer(saved_model_path, call_endpoint='serving_default')

def load_test_data():

    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.astype('float32') / 255.0
    x_test = np.expand_dims(x_test, axis=-1)
    return x_test, y_test

def predict_with_model(model_layer, x_test):

    predictions = model_layer(x_test)
    if isinstance(predictions, dict):
        predictions = predictions[list(predictions.keys())[0]]
    predictions = predictions.numpy()

    if predictions.ndim > 1:
        return np.argmax(predictions, axis=1)
    else:
        return predictions.astype(int)


def visualize_predictions(x_test, y_test, predicted_labels):

    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(x_test[i].squeeze(), cmap='gray')
        plt.title(f"True: {y_test[i]}, Pred: {predicted_labels[i]}")
        plt.axis('off')
    plt.show()


if __name__ == "__main__":
    saved_model_path = '../saved_models/lenet5'

    print("Ładowanie modelu...")
    model_layer = load_saved_model(saved_model_path)

    print("Ładowanie danych testowych...")
    x_test, y_test = load_test_data()

    print("Wykonywanie przewidywań...")
    predicted_labels = predict_with_model(model_layer, x_test)

    print("Wizualizacja wyników...")
    visualize_predictions(x_test, y_test, predicted_labels)
