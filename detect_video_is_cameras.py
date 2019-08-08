import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import tensorflow as tf
import numpy as np
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs
from is_wire.core import Channel, Subscription
from is_msgs.image_pb2 import Image
from pyimagesearch.centroidtracker import CentroidTracker
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

def get_rects(img, outputs):
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    wh = np.append(wh,wh)
    rect = []
    for i in range(nums):
        rect.append(tuple((np.array(boxes[i][0:4]) * wh).astype(np.int32)))
    return rect

config = tf.compat.v1.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf','path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('camera', '1', 'camera id')
flags.DEFINE_integer('nframes', 100,'camera id')
flags.DEFINE_string('output', './output.mp4', 'path to output video')

centroidTracker = CentroidTracker(20)

def main(_argv):
    if FLAGS.tiny:
        yolo = YoloV3Tiny()
    else:
        yolo = YoloV3()

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    times = []

    # Connect to the broker
    broker = "ampq://guest:guest@10.10.2.1:30000"
    channel = Channel(broker)
    
    # Subscribe to the desired topic
    subscription = Subscription(channel)
    camera_id = "CameraGateway."+FLAGS.camera+".Frame"
    subscription.subscribe(topic=camera_id)
    
    #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(FLAGS.output,fourcc, 5.0, (1288,728))
    for i in range(FLAGS.nframes):
        
        msg = channel.consume()
        img = msg.unpack(Image)
        img = get_np_image(img)
        img_to_draw = img
        
        #img = tf.image.decode_image(img, channels=3)
        img = tf.expand_dims(img, 0)
        img = transform_images(img, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img)
        t2 = time.time()
        times.append(t2-t1)
        times = times[-20:]

        for i in range(nums[0]):
            logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])], np.array(scores[0][i]),np.array(boxes[0][i])))
        rects = get_rects(img_to_draw, (boxes, scores, classes, nums))

        img_to_draw = draw_outputs(img_to_draw, (boxes, scores, classes, nums), class_names)
        
        objects = centroidTracker.update(rects)

        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "{}".format(objectID)
            cv2.putText(img_to_draw, text, (centroid[0], centroid[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 240, 0), 4)
            #cv2.circle(frame, (centroid[0], centroid[1]), 3, (0, 255, 0), -1)

        out.write(img_to_draw)
        
    out.release()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
