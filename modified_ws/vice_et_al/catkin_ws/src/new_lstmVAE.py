#!/usr/bin/env python
"""
This script runs the ROS node for Convolutional LSTM based exploration.

Gazebo simulation with ROS should be running first.
Usage: python lstmAE.py
"""

import os
import tensorflow as tf
import rospy
import cv2
import numpy as np
import threading
import time
from std_msgs.msg import Float32, Float32MultiArray, String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from skimage.metrics import structural_similarity as ssim

from tensorflow.keras.layers import Conv2D, ConvLSTM2D, TimeDistributed, LayerNormalization
from tensorflow.keras.models import Sequential

# Global variables for the knowledge graph (KG)
knowledge_graph = {
    'nodes': [],
    'edges': []
}

def main(args):
    rospy.init_node('lstm_AE', anonymous=True)
    q = build_q()
    ic = image_converter(q)  # pass tf q
    evaluate(q)  # sets up publisher and ConvLSTM model and then calls main loop
    cv2.destroyAllWindows()
    q.close()

class inference_obj(object):
    def __init__(self, model, model_inf):
        self.pub_ssim = rospy.Publisher('ssim', Float32, queue_size=1)
        self.pub_ae_image = rospy.Publisher('ae_image', Float32MultiArray, queue_size=1)
        self.debug = rospy.Publisher('debug', String, queue_size=1)
        self.bridge = CvBridge()
        self.fifo_set = np.zeros((1, 15, 256, 256, 1), dtype=float)  # initialize fifo buffer of 15 frames
        self.model = model  # to copy weights
        self.model_inf = model_inf
        self.i = 0
        self.image_subscribe = rospy.Subscriber("/camera/image", Image, self.callback)

    def callback(self, data):
        if self.i % 90 == 0:  # about every three seconds
            self.model_inf.set_weights(self.model.get_weights())
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
            return
        
        cv_image = cv2.resize(cv_image, (256, 256))
        cv_image_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        cv_image_norm = np.array(cv_image_gray, dtype=np.float32) / 256.0
        cv_image_norm = np.reshape(cv_image_norm, (1, 1, 256, 256, 1))

        self.fifo_set = np.concatenate((cv_image_norm, self.fifo_set[:, :-1, :, :, :]), axis=1)

        predicted_frames = self.model_inf.predict(self.fifo_set[:, :10, :, :, :], batch_size=1)

        # Calculate Structural Similarity Index (SSIM)
        ssim_value = ssim(np.reshape(self.fifo_set[0, 14], (256, 256)),
                          np.reshape(predicted_frames[0, 4], (256, 256)),
                          data_range=1)

        # Publish SSIM value
        self.pub_ssim.publish(ssim_value.astype(np.float32))

        # Log anomaly if SSIM is below threshold (example threshold = 0.9)
        if ssim_value < 0.9:
            log_anomaly(cv_image_gray)

        self.i += 1

def log_anomaly(image_gray):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    anomaly_data = {
        'timestamp': timestamp,
        'image': image_gray.tolist(),
        'ssim': ssim_value
    }
    
    knowledge_graph['nodes'].append(anomaly_data)

    # For simplicity, we assume an edge to connect the last added node to the previous one
    if len(knowledge_graph['nodes']) > 1:
        knowledge_graph['edges'].append({
            'from': len(knowledge_graph['nodes']) - 2,
            'to': len(knowledge_graph['nodes']) - 1,
            'type': 'occurred'
        })

def inference_thread_f(model, model_inf):
    ic = inference_obj(model, model_inf)
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        print("ROS Interrupted")

def evaluate(q):
    model, model_inf = get_func_model()
    exec_main_loop(model, model_inf, q)

def exec_main_loop(model, model_inf, q):
    saveDot9 = False  # Example: save model when SSIM > 0.9
    inference_thread = threading.Thread(target=inference_thread_f, args=(model, model_inf,))
    inference_thread.start()

    # Main loop example
    while not rospy.is_shutdown():
        # Perform some operations here
        time.sleep(1)

    # Optionally save model weights
    if saveDot9:
        model.save_weights('models/dot9Model')

def build_q():
    q_size = 100
    shape = (256, 256, 1)
    q = tf.queue.FIFOQueue(q_size, [tf.float32], shapes=shape)
    return q

class image_converter(object):
    def __init__(self, q):
        self.bridge = CvBridge()
        self.q = q
        self.image_sub = rospy.Subscriber("/camera/image", Image, self.callback)

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

        q_image = cv2.resize(cv_image, (256, 256))
        q_image = cv2.cvtColor(q_image, cv2.COLOR_BGR2GRAY)
        q_image = np.array(q_image, dtype=np.float32) / 256.0
        self.q.enqueue(np.reshape(q_image, (256, 256, 1)))

def get_func_model():
    # Example function to create and compile ConvLSTM model
    model_inputs, encode_only = encoder_model()
    model_all_layers = decoder_model(encode_only)

    model = tf.keras.Model(inputs=[model_inputs], outputs=[model_all_layers], name="FullConvLSTM_AE")
    model.compile(loss=['mse', 'mse'], optimizer=tf.keras.optimizers.Adam(lr=1.5e-4),
                  metrics=["mae"], loss_weights=[1.0, 0.0])

    model_inf = tf.keras.Model(inputs=[model_inputs], outputs=[model_all_layers], name="encoder_only")
    model_inf.compile(loss=['mse', 'mse'], optimizer=tf.keras.optimizers.Adam(lr=1.5e-4),
                      metrics=["mae"], loss_weights=[1.0, 0.0])

    return model, model_inf

def encoder_model():
    # Example encoder model
    model_input = tf.keras.Input(shape=(10, 256, 256, 1))
    conv_layer = Conv2D(64, (3, 3), padding='same')
    lstm_layer = ConvLSTM2D(32, (3, 3), padding='same', return_sequences=True)
    norm_layer = LayerNormalization()

    x = TimeDistributed(conv_layer)(model_input)
    x = TimeDistributed(norm_layer)(x)
    x = lstm_layer(x)

    return model_input, x

def decoder_model(encoder_model):
    # Example decoder model
    conv_transpose_layer = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')
    time_dist_layer = TimeDistributed(conv_transpose_layer)

    x = time_dist_layer(encoder_model)

    return x

if __name__ == '__main__':
    main(sys.argv)

