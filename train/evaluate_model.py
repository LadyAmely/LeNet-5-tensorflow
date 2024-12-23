import tensorflow as tf
import numpy as np
from dataset.dataset_loader import load_and_preprocess_data
from utils.metrics import calculate_metrics
def evaluate_model():
    _, _, x_test, y_test = load_and_preprocess_data()
    model = tf.keras.models.load_model('../saved_models/lenet5.h5')
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    y_pred = np.argmax(model.predict(x_test), axis=1)
    precision, recall, f1 = calculate_metrics(y_test, y_pred)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")


if __name__ == "__main__":
    evaluate_model()
