import tensorflow as tf
from dataset.dataset_loader import load_and_preprocess_data

def evaluate_model():
    _, _, x_test, y_test = load_and_preprocess_data()
    model = tf.keras.models.load_model('../saved_models/lenet5.h5')
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    evaluate_model()
