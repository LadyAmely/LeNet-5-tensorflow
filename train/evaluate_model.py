import tensorflow as tf
import numpy as np
from dataset.dataset_loader import load_and_preprocess_data
from utils.metrics import calculate_metrics
def evaluate_model():
    _, _, x_test, y_test = load_and_preprocess_data()

    model_layer = tf.keras.layers.TFSMLayer('../saved_models/lenet5/', call_endpoint='serving_default')

    predictions = model_layer(x_test)
    if isinstance(predictions, dict):
        predictions = predictions[list(predictions.keys())[0]]
    predictions = predictions.numpy()

    y_pred = np.argmax(predictions, axis=1)
    precision, recall, f1 = calculate_metrics(y_test, y_pred)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

if __name__ == "__main__":
    evaluate_model()
