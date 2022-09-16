from pathlib import Path
import random
import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
import cv2
import core.utils as utils

flags.DEFINE_string('weights', './checkpoints/yolov4-416', 'path to weights file')
flags.DEFINE_string('output', './checkpoints/yolov4-416-fp32.tflite', 'path to output')
flags.DEFINE_integer('input_height', 480, 'image height')
flags.DEFINE_integer('input_width', 640, 'image width')
flags.DEFINE_string('quantize_mode', 'float32', 'quantize mode (int8, full-int8, float16, float32)')
flags.DEFINE_string('dataset', None, 'path to dataset for calibration')
flags.DEFINE_integer('representative_number', 50, 'number of representative images for calibration.')


def representative_data_gen():
  with open(FLAGS.dataset, "r") as image_list_file:
    image_paths = [Path(line.split("\n")[0]) for line in image_list_file.readlines()]
    for image_path in random.sample(image_paths, FLAGS.representative_number):
      if image_path.exists():
        image = utils.image_preprocess(np.copy(cv2.imread(image_path.as_posix())), [FLAGS.input_height, FLAGS.input_width])
        yield [image[np.newaxis, ...].astype(np.float32)]

def save_tflite():
  converter = tf.lite.TFLiteConverter.from_saved_model(FLAGS.weights)
  converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
  if FLAGS.quantize_mode != 'float32':
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.allow_custom_ops = True
  if FLAGS.quantize_mode == 'float16':
    converter.target_spec.supported_types = [tf.float16]
  elif FLAGS.quantize_mode == 'full-int8':
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
  if FLAGS.dataset is not None:
      converter.representative_dataset = representative_data_gen
  tflite_model = converter.convert()
  with open(FLAGS.output, 'wb') as tflite_file:
    tflite_file.write(tflite_model)
  logging.info("model saved to: {}".format(FLAGS.output))


def main(_argv):
  with tf.device("cpu"):
    _ = _argv
    save_tflite()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass


