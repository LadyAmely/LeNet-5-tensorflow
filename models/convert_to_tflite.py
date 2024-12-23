import tensorflow as tf

saved_model_dir = '../saved_models/lenet5'
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

tflite_model_path = '../saved_models/lenet5.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f"Model TFLite zapisany pod ścieżką: {tflite_model_path}")
