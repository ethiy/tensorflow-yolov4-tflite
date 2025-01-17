import time
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

flags.DEFINE_string("framework", "tf", "(tf, tflite, trt")
flags.DEFINE_string("weights", "./checkpoints/yolov4-416", "path to weights file")
flags.DEFINE_integer("height", 480, "resize images height to")
flags.DEFINE_integer("width", 640, "resize images width to")
flags.DEFINE_boolean("tiny", False, "yolo or yolo-tiny")
flags.DEFINE_string("model", "yolov4", "yolov3 or yolov4")
flags.DEFINE_string("image", "./data/kite.jpg", "path to input image")
flags.DEFINE_string("output", "result.png", "path to output image")
flags.DEFINE_float("iou", 0.45, "iou threshold")
flags.DEFINE_float("score", 0.25, "score threshold")
flags.DEFINE_string("classes", "./data/classes/coco.names", "File of class names")


def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_height = FLAGS.height
    input_width = FLAGS.width
    image_path = FLAGS.image

    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    image_data = cv2.resize(original_image, (input_height, input_width))
    image_data = image_data / 255.0

    images_data = np.asarray([image_data]).astype(np.float32)

    start_time = time.time()
    if FLAGS.framework == "tflite":
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
        interpreter.set_tensor(input_details[0]["index"], images_data)
        interpreter.invoke()
        pred = [
            interpreter.get_tensor(output_details[i]["index"])
            for i in range(len(output_details))
        ]
        if FLAGS.model == "yolov3" and FLAGS.tiny == True:
            boxes, pred_conf = filter_boxes(
                pred[1],
                pred[0],
                score_threshold=0.25,
                input_shape=tf.constant([input_height, input_width]),
            )
        else:
            boxes, pred_conf = filter_boxes(
                pred[0],
                pred[1],
                score_threshold=0.25,
                input_shape=tf.constant([input_height, input_width]),
            )
    else:
        saved_model_loaded = tf.saved_model.load(
            FLAGS.weights, tags=[tag_constants.SERVING]
        )
        infer = saved_model_loaded.signatures["serving_default"]
        batch_data = tf.constant(images_data)
        pred_bbox = infer(batch_data)
        for _, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])
        ),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=FLAGS.iou,
        score_threshold=FLAGS.score,
    )
    print("Runtime:", time.time() - start_time)
    pred_bbox = [
        boxes.numpy(),
        scores.numpy(),
        classes.numpy(),
        valid_detections.numpy(),
    ]
    image = utils.draw_bbox(
        original_image, pred_bbox, classes=utils.read_class_names(FLAGS.classes)
    )
    image = Image.fromarray(image.astype(np.uint8))
    image.show()
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    cv2.imwrite(FLAGS.output, image)


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
