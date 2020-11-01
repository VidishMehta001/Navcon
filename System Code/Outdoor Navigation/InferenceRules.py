# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 08:34:31 2020

@author: vidis
"""

import numpy as np
import cv2
import time

import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile


from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image, ImageGrab
from IPython.display import display


sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util
import pyttsx3

import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'

from datetime import datetime


MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

## Loading the model from API
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if "frozen_inference_graph.pb" in file_name:
        tar_file.extract(file, os.getcwd())


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
        

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes = NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

IMAGE_SIZE = (12, 8)

def speech_to_text(text):
    speechEngine = pyttsx3.init()
    speechEngine.say(text)
    speechEngine.runAndWait()
    return

update_time = time.time()

with detection_graph.as_default():
  with tf.compat.v1.Session(graph=detection_graph) as sess:
    last_update_time = datetime.now()
    while True:
      print("Time taken for each frame:" + str(datetime.now()-last_update_time))
      last_update_time = datetime.now()
      screen = np.array(ImageGrab.grab(bbox=(0,40,750,1000)))
      cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
      #cv2.imshow('window', screen, cv2.COLOR_BGR2RGB)
      #ret, image_np = cap.read()
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(screen, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          screen,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
      
      for i, b in enumerate(boxes[0]):
        if classes[0][i] == 1:
            if scores[0][i] > 0.5:
                mid_x = (boxes[0][i][3] + boxes[0][i][1])/2
                mid_y = (boxes[0][i][0] + boxes[0][i][2])/2
                distance = round((1-(boxes[0][i][3] - boxes[0][i][1]))**4,1)
                cv2.putText(screen, '{}'.format(distance), (int(mid_x*750),int(mid_y*1000)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
                if distance <= 0.3:
                    if mid_x > 0.3 and mid_x < 0.6:
                        cv2.putText(screen, 'Person Ahead!!', (int(mid_x*750)-50,int(mid_y*1000)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
                        speechEngine = pyttsx3.init()
                        speechEngine.say("Person ahead, please slow down!")
                        speechEngine.runAndWait()
        
        
      for i, b in enumerate(boxes[0]):
          if (classes[0][i]==3) or (classes[0][i]==4) or (classes[0][i]==6) or (classes[0][i]==8):
              if scores[0][i]>0.5:
                  mid_x = (boxes[0][i][3] + boxes[0][i][1])/2
                  mid_y = (boxes[0][i][0] + boxes[0][i][2])/2
                  distance = round((1-(boxes[0][i][3] - boxes[0][i][1]))**4,1)
                  if distance <=0.4:
                      if mid_x > 0.7:
                          if classes[0][i]==3:
                              current_time = time.time()
                              delta_time_update = time.time() - update_time
                              cv2.putText(screen, "Car to your right! Please stay back",  (int(mid_x*750)-50,int(mid_y*1000)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
                              if delta_time_update > 20:
                                  speech_to_text("Car to your right! Please stay back")
                                  update_time = time.time()
                              elif classes[0][i]==4:
                                  cv2.putText(screen, "Motorcycle to your right! Please stay back",  (int(mid_x*750)-50,int(mid_y*1000)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
                                  speech_to_text("Motorcycle to your right! Please stay back")
                              elif classes[0][i]==8:
                                  cv2.putText(screen, "Bus to your right! Please stay back",  (int(mid_x*750)-50,int(mid_y*1000)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
                                  speech_to_text("Bus to your right! Please stay back")
                      elif mid_x < 0.3:
                          if classes[0][i]==3:
                                  current_time = time.time()
                                  delta_time_update = time.time() - update_time
                                  cv2.putText(screen, "Car to your left! Please stay back",  (int(mid_x*750)-50,int(mid_y*1000)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
                            #if delta_time_update > 15:
                            #time.sleep(5)
                                  speech_to_text("Car to your left! Please stay back")
                                  update_time = time.time()
                          elif classes[0][i]==4:
                              cv2.putText(screen, "Motorcycle to your left! Please stay back",  (int(mid_x*750)-50,int(mid_y*1000)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
                              speech_to_text("Motorcycle to your left! Please stay back")
                          elif classes[0][i]==8:
                              cv2.putText(screen, "Bus to your left! Please stay back",  (int(mid_x*750)-50,int(mid_y*1000)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
                              speech_to_text("Bus to your left! Please stay back")
      if classes[0][i]==10:
            if scores[0][i] > 0.4:
                # mid_x = (boxes[0][i][3] + boxes[0][i][1])/2
                # mid_y = (boxes[0][i][0] + boxes[0][i][2])/2
                # distance = round((1-(boxes[0][i][3]-boxes[0][i][1]))**4,1)
                # if distance <= 0.5:
                #     if mid_x > 0.3 and mid_x < 0.7:
                cv2.putText(screen, 'Traffic Light ahead', (int(mid_x*750)-50,int(mid_y*1000)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
                speechEngine = pyttsx3.init()
                speechEngine.say("Junction ahead")
                speechEngine.runAndWait()
    
    
    
    
      ####################################
      # TEXT RECOGNITION WITH PYTESERRACT#
      ####################################
      
      hImg, wImg = 1000,750
      boxes = pytesseract.image_to_data(screen)
      for index, b in enumerate(boxes.splitlines()):
          if index!= 0:
              b = b.split()
              if len(b) == 12:
                  x,y,w,h = int(b[6]),int(b[7]),int(b[8]),int(b[9])
                  cv2.rectangle(screen,(x,y),(w+x,h+y),(0,0,255),3)
                   #cv2.putText(screen,b[11],(x,hImg-y+25),(w,hImg-h),cv2.FONT_HERSHEY_COMPLEX,(50,50,255),2)
                  
              
           
      cv2.imshow('object detection', cv2.resize(screen, (800,600)))
      if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break


