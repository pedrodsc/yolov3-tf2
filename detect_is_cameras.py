import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs
from is_wire.core import Channel, Subscription
from is_msgs.image_pb2 import Image
from pprint import pprint

def get_np_image(input_image):
    if isinstance(input_image, np.ndarray):
        output_image = input_image
    elif isinstance(input_image, Image):
        buffer = np.frombuffer(input_image.data, dtype=np.uint8)
        output_image = cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)
    else:
        output_image = np.array([], dtype=np.uint8)
    return output_image

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 608, 'resize images to')
flags.DEFINE_string('camera', '1', 'camera id')
flags.DEFINE_string('output', './output.jpg', 'path to output image')

def main(_argv):
    # Connect to the broker
    broker = "ampq://guest:guest@10.10.2.1:30000"
    channel = Channel(broker)
    
    # Subscribe to the desired topic
    subscription = Subscription(channel)
    camera_id = "CameraGateway."+FLAGS.camera+".Frame"
    subscription.subscribe(topic=camera_id)
    
    if FLAGS.tiny:
        yolo = YoloV3Tiny()
    else:
        yolo = YoloV3()

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')
    
    msg = channel.consume()
    img = msg.unpack(Image)
    img = get_np_image(img)
    img_to_draw = img
    
    #img = tf.image.decode_image(img, channels=3)
    img = tf.expand_dims(img, 0)
    img = transform_images(img, FLAGS.size)
 
    t1 = time.time()
    boxes, scores, classes, nums = yolo(img)
    t2 = time.time()
    logging.info('time: {}'.format(t2 - t1))

    logging.info('detections:')
    for i in range(nums[0]):
        logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                           np.array(scores[0][i]),
                                           np.array(boxes[0][i])))

    img_to_draw = draw_outputs(img_to_draw, (boxes, scores, classes, nums), class_names)
    cv2.imwrite(FLAGS.output, img_to_draw)
    logging.info('output saved to: {}'.format(FLAGS.output))

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
