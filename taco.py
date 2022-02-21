import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt
from tensorflow.python.util import compat
from tensorflow.core.protobuf import saved_model_pb2
from google.protobuf import text_format
import pprint
import json
import time
import cv2
import threading

from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import dataset_util, label_map_util
from object_detection.protos import string_int_label_map_pb2


USE_LIVE_DETECTION = False
IMAGE_PATH = os.path.join(os.getcwd(), 'input.png')

# reconstruct frozen graph
def reconstruct(pb_path):
    if not os.path.isfile(pb_path):
        print("Error: %s not found" % pb_path)

    print("Reconstructing Tensorflow model")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(pb_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    print("Success!")
    return detection_graph


def image2np(image):
    (w, h) = image.size
    return np.array(image.getdata()).reshape((h, w, 3)).astype(np.uint8)

def image2tensor(image):
    npim = image2np(image)
    return np.expand_dims(npim, axis=0)

def detect(detection_graph, image):
    with detection_graph.as_default():
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.01)
        with tf.compat.v1.Session(graph=detection_graph,config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image2tensor(image)}
            )

            npim = image2np(image)
            vis_util.visualize_boxes_and_labels_on_image_array(
                npim,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=15)
            plt.figure(figsize=(12, 8))
            plt.imshow(npim)
            plt.savefig("output.png")
            if USE_LIVE_DETECTION is False:
                print("Finished. Image saved to file")
            return npim


def resize_image(image):
    width, height = image.size
    max_w = 600
    if width < max_w:
        return image
    return image.resize((max_w, int(max_w/width * height)))

WORKING_DIR = os.getcwd()
ANNOTATIONS_FILE = os.path.join(WORKING_DIR, 'annotations.json')
NCLASSES = 60

with open(ANNOTATIONS_FILE) as json_file:
    data = json.load(json_file)
    
categories = data['categories']

print('\nBuilding label map from examples')

labelmap = string_int_label_map_pb2.StringIntLabelMap()
for idx,category in enumerate(categories):
    item = labelmap.item.add()
    # label map id 0 is reserved for the background label
    item.id = int(category['id'])+1
    item.name = category['name']

with open('./labelmap.pbtxt', 'w') as f:
    f.write(text_format.MessageToString(labelmap))

print('Label map witten to labelmap.pbtxt')

label_map = label_map_util.load_labelmap('labelmap.pbtxt')
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NCLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = reconstruct("taco_pb.pb")

if USE_LIVE_DETECTION is True:
    print("\nUsing OpenCV live detection")
    print('Starting up webcam')

    cam = cv2.VideoCapture(2, cv2.CAP_DSHOW)
    cv2.startWindowThread()
    cv2.namedWindow("Input")
    cv2.namedWindow("Output")

    blank = 1
    while True:
        ret, frame = cam.read()
        if blank == 1:
            im_np = np.zeros((300, 300, 3), np.uint8)
        if not ret:
            print("Failed to open webcam")
            break
        cv2.imshow("Input", frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            break
        if k == ord(' '):
            # Take the frame and convert it to PIL image, then feed it to the object detection api
            temp_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = resize_image(Image.fromarray(temp_frame))

            output = detect(detection_graph, pil_frame)
            im_np = np.asarray(output)
            im_np = cv2.cvtColor(im_np, cv2.COLOR_RGB2BGR)
            blank = 0

        cv2.imshow("Output", im_np)

else:
    print("Using single image detection. Processing...")
    st = time.time()
    detect(detection_graph, resize_image(Image.open(IMAGE_PATH)))
    print("Elapsed: " + str(round(time.time() - st, 2)) + " seconds")