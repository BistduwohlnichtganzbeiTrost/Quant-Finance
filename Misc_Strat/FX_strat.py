import tensorflow as tf

import numpy as np
import pandas as pd
import random

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout
from keras.utils import to_categorical
from keras import optimizers, metrics
from keras import backend as K

from datetime import datetime, timedelta

seed = 123
random.seed(seed)
np.random.seed(seed)

class BasicTemplateAlgorithm(QCAlgorithm):

    def Initialize(self):

        self.session = K.get_session()
        self.graph = tf.get_default_graph()

        self.SetStartDate(2018, 8, 1)
        self.SetEndDate(2018, 11, 21)
        self.SetCash(1e5)

        # set the currency pair, and the correlated currency pair
        self.currency = "AUDUSD"
        self.AddForex(self.currency, Resolution.Daily)

        self.corr_currency = "USDCHF"
        self.AddForex(self.corr_currency, Resolution.Daily)

        # define a long-short portfolio
        self.long_list = []
        self.short_list = []

        # initialize the indicators
        self.rsi = RelativeStrengthIndex(9)
        self.bb = BollingerBonds(14, 2, 2)
        self.macd = MovingAverageConvergenceDivergence(12, 26, 9)
        self.stochastic = Stochastic(14, 3, 3)
        self.ema = ExponentialMovingAverage(9)

        # arrays to store historical indicator values