import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet


# # Construct a basic model.
# root = tf.train.Checkpoint()
# root.v1 = tf.Variable(3.)
# root.v2 = tf.Variable(2.)
# root.f = tf.function(lambda x: root.v1 * root.v2 * x)
#
# # Save the model.
# export_dir = "/tmp/test_saved_model"
# input_data = tf.constant(1., shape=[1, 1])
# to_save = root.f.get_concrete_function(input_data)
# tf.saved_model.save(root, export_dir, to_save)
#
# # Convert the model.
# converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
# tflite_model = converter.convert()
# graph_def_file = "/models/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03/tflite_graph.pb"
# input_arrays = ["image_tensor"]
# output_arrays = ["detection_boxes","detection_scores","num_detections","detection_classes"]
#
# converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays)
# converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays, input_shapes={'image_tensor': [1, 300, 300, 3]})
# tflite_model = converter.convert()

# export_dir = 'models/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03/'
# converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
# tflite_model = converter.convert()

# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# tflite_quant_model = converter.convert()

# print(tf.__version__)


# model dir
saved_model_dir = 'models/ssd_mobilenet_v2_coco_2018_03_29/saved_model'

# 1 >> None is only supported in the 1st dimension. Tensor 'image_tensor' has invalid shape '[None, None, None, 3]'.
# converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
# tflite_model = converter.convert()


# 2 >>
# Found StridedSlice as non-selected output from Switch, but only Merge supported.
# Control flow ops like Switch and Merge are not generally supported. We are working on fixing this,
# please see the Github issue at https://github.com/tensorflow/tensorflow/issues/28485.
model = tf.saved_model.load(saved_model_dir)
concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
concrete_func.inputs[0].set_shape([None, 256, 256, 3])
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()


#  write tflite
open("converted_model.tflite", "wb").write(tflite_model)





