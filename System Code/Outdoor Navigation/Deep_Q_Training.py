# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 22:55:50 2020

@author: vidis
"""

import os 
# import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from collections import deque
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Activation, Flatten, MaxPool2D
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import time
import random
# from PIL import Image
import urllib
import tarfile
from datetime import datetime
from tqdm import tqdm

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


#####################
# DEEP Q CONSTANTS  # 
#####################

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50000
# MIN_REPLAY_MEMORY_SIZE = 1000
MIN_REPLAY_MEMORY_SIZE = 128
MINIBATCH_SIZE = 64
ACTION_CHOICES = 3
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MIN_REWARD = -900

LAST_AVE_REWARD = -900

MODEL_NAME = '2x256'

# EPISODES = 1000
EPISODES = 6

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False

# For stats
ep_rewards = [-200] * AGGREGATE_STATS_EVERY

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')
    
tf.compat.v1.disable_v2_behavior()


class BlockEnv(object):
    
    """
    Block Environment is the class to create a frame of reference for the person walking

    """
    
    def __init__(self, frame, x1, x2, y1, y2, frame_width, frame_height):
        self.frame = frame
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.frame_width = frame_width
        self.frame_height = frame_height
        
    def BuildBlock(self):
        return cv2.rectangle(self.frame, (self.x1, self.y1), (self.x2, self.y2), (0,0,0), cv2.FILLED)
    
    
    def movement(self):
        self.move_right = (self.x1+100, self.x2+100)
        self.move_left = (self.x1-100, self.x2-100)
        
    def action(self, choice):
        self.choice = choice
        
        ## If choice is 0, stay, choice is 1, move to the left, choice is 2, move to the right
        ## choice 3 is to move forward, and choice 4 is to move back.
        
        if self.choice == 2:
            if self.x2+10 > self.frame_width:
                self.x2 = self.frame_width
                self.x1 = self.x1
            else:
                self.x2 = self.x2+10
                self.x1 = self.x1+10
            return self.x1, self.x2, self.y1, self.y2
                
        if self.choice == 1:
            if self.x1 - 10 < 0:
                self.x1 = 0
                self.x2 = self.x2
            else:
                self.x1 = self.x1 - 10
                self.x2 = self.x2 - 10
            return self.x1, self.x2, self.y1, self.y2
    
        if self.choice == 0:
            return self.x1, self.x2, self.y1, self.y2
        



# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        
        self.writer = tf.summary.create_file_writer(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass
    

    def _write_logs(self, logs, index):
        # with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)



WIDTH = 120
HEIGHT = 150

class DQNAgent(object):
    
    def __init__(self):
                
        self.model = self.create_model()
        self.model.summary()
        
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        
        self.replay_memory = deque(maxlen = REPLAY_MEMORY_SIZE)
        
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))
        
        self.target_update_counter = 0
        
    def create_model(self):
        model = Sequential()
        
        # model.add(Conv2D(256, (3, 3), input_shape=(300, 250, 3)))
        # model.add(Conv2D(256, (5, 5), input_shape=(300, 250, 3)))
        model.add(Conv2D(256, (5, 5), input_shape=(HEIGHT, WIDTH, 3)))
        model.add(Activation('relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        
        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
    
        model.add(Flatten())
        model.add(Dense(64))
        
        model.add(Dense(ACTION_CHOICES, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.01), metrics=['accuracy', 'mse'])
        return model
    
    # Add the steps data to a memory replay array
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
        
    
    #Train the model
    
    def train(self, terminal_state, episode, step):
        
        # Start training process if the replay memory size is greater than the min replay memory size
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch])/255
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch])/255
        future_qs_list = self.target_model.predict(new_current_states)
        
        X = []
        Y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward
            
            current_qs = current_qs_list[index]
            new_qs = current_qs[action]

            current_qs[action] = new_q
            
            X.append(current_state)
            Y.append(current_qs)

        self.model.fit(np.array(X)/255, np.array(Y),
                       epochs=1,
                       batch_size=MINIBATCH_SIZE,
                       shuffle=False,
                       callbacks=[self.tensorboard] if terminal_state else None)

        if terminal_state:
            self.target_update_counter = self.target_update_counter + 1
            
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

        # save models
        # self.model.save('models/ep_{}_step_{}.model'.format(episode, step))  # save all the models
        self.model.save('models/best.model')# only save current model
    
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]
    
    
agent_graph = tf.Graph()
with agent_graph.as_default():
    agent = DQNAgent()

import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

## Loading the model from API
#opener = urllib.request.URLopener()
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
#tar_file = tarfile.open(MODEL_FILE)
#for file in tar_file.getmembers():
#    file_name = os.path.basename(file.name)
#    if "frozen_inference_graph.pb" in file_name:
#        tar_file.extract(file, os.getcwd())


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

last_update_time = datetime.now()
REWARD = 0
MOVE_PENALTY = 1
OBSTRUCTION_PENALTY = 300

def calculateIntersection(a0, a1, b0, b1):
    if a0 >= b0 and a1 <= b1: # Contained
        intersection = a1 - a0
    elif a0 < b0 and a1 > b1: # Contains
        intersection = b1 - b0
    elif a0 < b0 and a1 > b0: # Intersects right
        intersection = a1 - b0
    elif a1 > b1 and a0 < b1: # Intersects left
        intersection = b1 - a0
    else: # No intersection (either side)
        intersection = 0

    return intersection


def extract_frame(file_name, frame_count=80):
    frame_list = []
    cap = cv2.VideoCapture(file_name)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame = cv2.resize(frame, (250, 300))
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        frame_list.append(frame)
    return frame_list[:frame_count]


# extract frame before training procedure
video_frames = extract_frame("C:/Users/vidis/Videos/Movavi Library/TrainingInSg.mp4", frame_count=80)

i = 0
def detect_objects(video_frames):
    detection_result_list = []
    for index, frame in enumerate(video_frames):
        with detection_graph.as_default():
            with tf.compat.v1.Session(graph=detection_graph) as sess:
                image_np_expanded = np.expand_dims(frame, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],
                                                                    feed_dict={image_tensor: image_np_expanded})
                detection_result_list.append({
                    'boxes': boxes,
                    'scores': scores,
                    'classes': classes,
                    'num_detections': num_detections,
                })
                
  
                cv2.imwrite("C:/Users/vidis/OneDrive/Desktop/Navcon/Navcon/Object Detection/dqn_images/"+str(index)+".jpg", frame)

                
    return detection_result_list


# detect objects in frame before training procedure
detection_result_list = detect_objects(video_frames)

#%%
BlkCoordinates = [[[60, 80],[80, 150]]] #Size of the object

episode_positions = []

for episode in tqdm(range(1, EPISODES+1), ascii=True, unit='episodes'):
    print("start training episode: ", episode)
    
    # Update tensorboard step every episode
    with agent_graph.as_default():
        agent.tensorboard.step = episode
    
    # Restarting the episode - reset episode reward and the step
    # episode_reward = 0
    episode_reward = -200
    step = 1

    # cap = cv2.VideoCapture("WalkingInSg.mp4")

    # update ep_rewards
    ep_rewards = ep_rewards[-AGGREGATE_STATS_EVERY:]

    print("Time taken for each frame:" + str(datetime.now() - last_update_time))
    last_update_time = datetime.now()
    
    image_counter = 0
    
    position = []
    
    for cnt, frame, detection_result in zip(range(1, len(video_frames) + 1), video_frames, detection_result_list):
        print("Still within while loop!!!")
        choices = [0, 1, 2]

        current_state = frame

        if np.random.random() > epsilon:
            # Get action from Q table
            with agent_graph.as_default():
                action = np.argmax(agent.get_qs(current_state))
        else:
            # Get Random Action
            action = np.random.randint(0, len(choices))
            # print(action)

        print("--------------------------")
        print("Action taken" + str(action))
        
        image_counter = image_counter+1
        
        #cv2.imwrite("C:/Users/vidis/OneDrive/Desktop/Navcon/Navcon/Object Detection/dqn_images/"+str(image_counter)+".jpg",frame)

        # with detection_graph.as_default():
        #     with tf.compat.v1.Session(graph=detection_graph) as sess:
        #         image_np_expanded = np.expand_dims(frame, axis=0)
        #         image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        #         # Each box represents a part of the image where a particular object was detected.
        #         boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        #          # Each score represent how level of confidence for each of the objects.
        #         # Score is shown on the result image, together with the class label.
        #         scores = detection_graph.get_tensor_by_name('detection_scores:0')
        #         classes = detection_graph.get_tensor_by_name('detection_classes:0')
        #         num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        #         # Actual detection.
        #         (boxes, scores, classes, num_detections) = sess.run(
        #             [boxes, scores, classes, num_detections],
        #             feed_dict={image_tensor: image_np_expanded})
        #         # Visualization of the results of a detection.
        #         vis_util.visualize_boxes_and_labels_on_image_array(
        #             frame,
        #             np.squeeze(boxes),
        #             np.squeeze(classes).astype(np.int32),
        #             np.squeeze(scores),
        #             category_index,
        #             use_normalized_coordinates=True,
        #             line_thickness=8)

        boxes = detection_result['boxes']
        classes = detection_result['classes']
        scores = detection_result['scores']
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)

        height, width, channels = frame.shape

        x1 = BlkCoordinates[-1][0][0]
        x2 = BlkCoordinates[-1][1][0]
        y1 = BlkCoordinates[-1][0][1]
        y2 = BlkCoordinates[-1][1][1]

        Blk = BlockEnv(frame=frame, x1=x1, x2=x2, y1=y1, y2=y2, frame_width=width, frame_height=height)

        # print("--------------")
        # print(x1, x2, y1, y2)
        # print(action)
        x1, x2, y1, y2 = Blk.action(choice=action)
        
        position.append([x1,x2,y1,y2])

        if action == 1 or action == 2:
            reward = -MOVE_PENALTY
            episode_reward = episode_reward + reward
        else:
            reward = 0
            episode_reward = episode_reward + reward

        AREA = float((x2 - x1) * (y2 - y1))

        # print(x1, x2, y1, y2)
        # print("---------------")

        BlkCoordinates = [[[x1, y1], [x2, y2]]]

        Blk = BlockEnv(frame=frame, x1=x1, x2=x2, y1=y1, y2=y2, frame_width=width, frame_height=height)

        Agent = Blk.BuildBlock()
        
        new_state = frame

        for i, b in enumerate(boxes[0]):
            if classes[0][i] == 1:
                if scores[0][i] > 0.5:
                    width = calculateIntersection(boxes[0][i][1] * 800, boxes[0][i][3] * 800, x1, x2)
                    height = calculateIntersection(boxes[0][i][0] * 600, boxes[0][i][2] * 600, y1, y2)
                    area = width * height
                    percent = area / AREA
                    print(percent)
                    if percent >= 0.5:
                        reward = -OBSTRUCTION_PENALTY
                        episode_reward = episode_reward + reward

        done = True
        with agent_graph.as_default():
            agent.update_replay_memory((current_state, action, reward, new_state, done))
            agent.train(done, episode, step)

        print("--------------------")
        print("Action taken is " + str(action) + " and reward is " + str(reward))
        print("======================")

        current_state = new_state
        step += 1

        print("Current Reward is " + str(episode_reward))

        cv2.imshow('window', frame)

        ep_rewards.append(episode_reward)
        # if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        if cnt % 10 == 0:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            with agent_graph.as_default():
                agent.tensorboard.update_stats(reward_avg=average_reward,
                                               reward_min=min_reward,
                                               reward_max=max_reward,
                                               epsilon=epsilon)

            # Save model, but only when min reward is greater or equal a set value
            if min_reward >= MIN_REWARD:
                with agent_graph.as_default():
                    print('current ave reward is: ', average_reward)
                    if average_reward > LAST_AVE_REWARD:
                        LAST_AVE_REWARD = average_reward
                        print('model is saved with average reward: ', average_reward)
                        # agent.model.save(
                            # f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
                        #     f'models/best.model')

        print("Check if model is saving")
        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            print('break')
            break
    
    episode_positions.append(position)

print('Done!')


#%%


# for index, episode_position in enumerate(episode_positions):
    
#     print("Episode: " + str(index+1))
    
for index2, (position, frame) in enumerate(zip(episode_positions[5],video_frames)):
    
    
    # boxes = detection_result['boxes']
    # classes = detection_result['classes']
    # scores = detection_result['scores']
    # vis_util.visualize_boxes_and_labels_on_image_array(
    #     frame,
    #     np.squeeze(boxes),
    #     np.squeeze(classes).astype(np.int32),
    #     np.squeeze(scores),
    #     category_index,
    #     use_normalized_coordinates=True,
    #     line_thickness=8)

    cv2.rectangle(frame, (position[0], position[2]), (position[1], position[3]), (0,0,0), cv2.FILLED)
    
    cv2.imwrite("C:/Users/vidis/OneDrive/Desktop/Navcon/Navcon/Object Detection/dqn_images/"+str(index2)+"episode_.jpg",frame)
        


#%%
def extract_frame(file_name, frame_count=80):
    frame_list = []
    cap = cv2.VideoCapture(file_name)
    i = 0
    while True:
        i = i + 1
        ret, frame = cap.read()
        if not ret:
            break

        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame = cv2.resize(frame, (250, 300))
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        frame_list.append(frame)
    return frame_list[:frame_count]


video_frames = extract_frame("C:/Users/vidis/Videos/Movavi Library/TrainingInSg.mp4", frame_count=80)

for index, frame in enumerate(video_frames):
    cv2.imwrite("C:/Users/vidis/OneDrive/Desktop/Navcon/Navcon/Object Detection/dqn_images/test/"+str(index)+".jpg", frame)

