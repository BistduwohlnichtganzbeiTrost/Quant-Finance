import tensorflow as tf
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Flatten, Lambda, Dense
from keras.models import Sequential
from keras.layers import BatchNormalization, Activation

import numpy as np
import pandas as pd


class DDPG(object):

    def __init__(self, config):
        '''
        initialized approximate value function
        
        config should have the following attributes
        
        Args:
            device: the device to use computation, e.g. '/gpu:0'
            gamma(float): the decay rate for value at RL
            history_length(int): input_length for each scale at CNN
            n_feature(int): the number of type of input 
                (e.g. the number of company to use at stock trading)
            trade_stock_idx(int): trading stock index
            gam (float): discount factor
            n_history(int): the nubmer of history that will be used as input
            n_smooth, n_down(int): the number of smoothed and down sampling input at CNN
            k_w(int): the size of filter at CNN
            n_hidden(int): the size of fully connected layer
            n_batch(int): the size of mini batch
            n_epochs(int): the training epoch for each time
            update_rate (0, 1): parameter for soft update
            learning_rate(float): learning rate for SGD
            memory_length(int): the length of Replay Memory
            n_memory(int): the number of different Replay Memories
            alpha, beta: [0, 1] parameters for Prioritized Replay Memories
            action_scale(float): the scale of initialized ation
        '''
        self.device = config.device
        self.save_path = config.save_path
        self.is_load = config.is_load
        self.gamma = config.gamma
        self.history_length = config.history_length
        self.n_stock = config.n_stock
        self.n_smooth = config.n_smooth
        self.n_down = config.n_down
        self.n_batch = config.n_batch
        self.n_epoch = config.n_epoch
        self.update_rate = config.update_rate
        self.alpha = config.alpha
        self.beta = config.beta
        self.lr = config.learning_rate
        self.memory_length = config.memory_length
        self.n_memory = config.n_memory
        self.noise_scale = config.noise_scale
        self.model_config = config.model_config
        # the length of the data as input
        self.n_history = max(self.n_smooth + self.history_length, (self.n_down + 1) * self.history_length)
        print ("building model....")
        # have compatibility with new tensorflow
        tf.python.control_flow_ops = tf
        # avoid creating _LEARNING_PHASE outside the network
        K.clear_session()
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        K.set_session(self.sess)
        with self.sess.as_default():
            with tf.device(self.device):
                self.build_model()
        print('finished building model!')

    def train(self, input_data):
        self.max_action = 100
        '''
        training DDPG, where action is confined to integer space

        Args:
            data (DataFrame): stock price for self.n_feature companies
        '''
         stock_data = input_data.values
        date = input_data.index
        T = len(stock_data)
        
        # frequency for output
        print_freq = int(T / 10)
        if print_freq == 0:
            print_freq = 1
            
        print ("training....")
        st = time.time()
        # prioritizomg parameter
        db = (1 - self.beta) / 1000
        
        # result for return value
        values = []
        date_label = []
        value = 0
        values.append(value)
        date_label.append(date[0])
        # keep half an year data 
        t0 = self.n_history + self.n_batch
        self.initialize_memory(stock_data[:t0])
        plot_freq = 10
        save_freq = 100000
        count = 0
        for t in range(t0, T - 1):
            self.update_memory(stock_data[t], stock_data[t+1])
            reward = self.take_action(stock_data[t], stock_data[t+1])
            value += reward
            date_label.append(date[t+1])
            values.append(value)
            count += 1
            for epoch in range(self.n_epoch):    
                # select transition from pool
                self.update_weight()
                # update prioritizing paramter untill it goes over 1
                # self.beta  += db
                if self.beta >= 1.0:
                    self.beta = 1.0
                 
            if t % print_freq == 0:
                print ("time:",  date[t + 1])
                action = self.predict_action(stock_data[t+1])
                print("portfolio:", action)
                print("reward:", reward)
                print("value:", value)
                print ("elapsed time", time.time() - st)
                print("********************************************************************")
                
            if count % plot_freq == 0:
                result = pd.DataFrame(values, index=pd.DatetimeIndex(date_label))
                result.to_csv("training_result.csv")
                
            if count % save_freq == 0:
                save_path = self.saver.save(self.sess, self.save_path)
                print("Model saved in file: %s" % self.save_path)

        save_path = self.saver.save(self.sess, self.save_path)
        print("Model saved in file: %s" % self.save_path)
        print ("finished training")
           
        return pd.DataFrame(values, index=pd.DatetimeIndex(date_label))
    
    def norm_action(self, action):
        max_action = np.max(np.abs(action))
        if max_action > 1:
            return action / max_action
        else:
            return action