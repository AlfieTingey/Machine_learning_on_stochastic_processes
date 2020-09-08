import keras
import keras.backend as k_back
from keras.constraints import *
import os
from keras.models import Model
import tensorflow as tf
from scipy.io import loadmat, savemat
from tensorflow.python.framework import tensor_util
import sys
from keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback
import time
from collections import defaultdict
import datetime
from get_best_model import GetBestModel
from keras.initializers import RandomUniform, Constant
import numpy as np

""" This details the code that we use for building our RNN and the process that
we use for learning the parameters using the RNN.

This code has been inspired and adapted from the following source:
Garbi, G et al. (2020). Learning Queueing Networks by Recurrent Neural Networks.
Accessible at:
https://pdfs.semanticscholar.org/7f7c/12bcc23ba098ad5a4a0ad251bd92e9b9c27a.pdf."""


class RNNCell(keras.layers.Layer):
    """ In this class we define some variables and some methods that we will use
    to build the crucial 'explainable' rnn_cell as we detailed in the project."""
    def __init__(self, trace, **kwargs):
        """ Input the trace, number of stations M, state-size, job placements """
        M = trace.size
        self.trace = k_back.constant(trace)
        self.index_jobs = slice(1, M+1)
        self.state_size = (M+1,)

        super().__init__(*kwargs)

    def build(self, shape_input):
        """ Build the RNN Cell using certain variables. We have the number of stations M;
        we set the initial conditions for mu and pfd as randomely chosen following a uniform
        distrobution. We add a weight that tracks the mu values. We set a constaint that is
        is non-negative. We set a weight that tracks the matrix P, and we set constraints
        as detailed in the report. We define the 'odot' function as shown in the paper. """

        M = shape_input[1] - 1
        self.I = k_back.eye(M)
        init_mu = RandomUniform(minval=0.01, maxval=10)
        init_pfd = RandomUniform(minval=0.01, maxval=10)
        self.mu = self.add_weight('mu', shape=(M, 1), initializer=init_mu, constraint=NonNeg())
        data_p = self.add_weight('data_p', shape=(M, M-1), initializer=init_pfd, constraint=NonNeg())
        data_p_scaled = data_p/k_back.sum(data_p, axis=1, keepdims=True)
        self.P = k_back.reshape(k_back.flatten(data_p_scaled)[None, :] @ k_back.one_hot([j for j in range(M*M) if j % (M+1) != 0], M*M), (M, M))
        self.odot = (self.P - self.I)*self.mu
        self.is_built = True

    def init_xh(self, inputs):
        """ Initialize x_h with input values """
        return_input = inputs
        return return_input, return_input

    def predict_xh(self, inputs, state):
        """ Define a method to predict the next x_h as detailed by the
        fluid dynamic equations """
        current_t = inputs[:, 0]
        old_t = state[:, 0]
        diff_t = current_t - old_t
        pred = state[:, self.index_jobs] + (diff_t[:, None]*k_back.minimum(state[:, self.index_jobs], self.trace)) @ self.odot
        pred_out = k_back.concatenate([current_t[:, None], pred], axis=1)
        return pred_out, pred_out

    def call(self, inputs, states):
        """ Define next values for x_h """
        xh_first, st_first = self.init_xh(inputs)
        xh_pred, st_pred = self.predict_xh(inputs, states[0])
        decide_first = k_back.equal(inputs[:, 1], -1)
        xh = k_back.switch(decide_first, xh_pred, xh_first)
        st = k_back.switch(decide_first, st_pred, st_first)
        return xh, [st]

    def max_abs_percent_error(self, y_true, y_pred):
        """ Create max absolute percentage error method to calculate error.
        We then use this value for the loss function """

        y_true = k_back.mean(y_true, axis=0, keepdims=True)
        y_pred = k_back.mean(y_pred, axis=0, keepdims=True)
        self.y_true = y_true
        self.y_pred = y_pred
        ones_matrix = k_back.ones_like(y_true[:, :, self.index_jobs])
        zero_matrix = k_back.zeros_like(y_true[:, :, self.index_jobs])
        err = k_back.abs(y_true[:, :, self.index_jobs] - y_pred[:, :, self.index_jobs])*k_back.switch(
            k_back.equal(y_true[:, :, self.index_jobs], - ones_matrix), zero_matrix, ones_matrix)
        N = k_back.sum(y_true[:, :, self.index_jobs], axis=2)
        percentage_error = k_back.sum(err, axis=2)/(2*N)
        max_trace_error = k_back.max(percentage_error, axis=1)
        average_error = k_back.mean(max_trace_error, axis=0)
        return 100*average_error

    def loss(self, y_true, y_pred):
        """ Define the loss function that we will use while learning """
        return self.max_abs_percent_error(y_true, y_pred)

    def get_mu(self):
        """ Evaluate predicted parameter mu """
        return k_back.eval(self.mu)

    def get_p(self):
        """ Evaluate predicted parameter P"""
        return k_back.eval(self.P)

    def print(mu_P):
        """ Define a method to print our predicted parameters"""
        print(self.get_mu())
        print(self.get_P())
        print(K.eval(self.odot))


class RNNCellTraining:
    """ In this class we build the full RNN model and we load, read and input the
    trace data from the original queueing network model into the RNN model. We then
    define the learning process (using val loss function), and save the results
    of the learned parameters """

    def __init__(self, directory, rnn_cell):
        """ Define the directory that we use and the cell we use to build the RNN """
        self.directory = directory
        self.rnn_cell = rnn_cell

    def read_files(self, fname):
        """ Method that we will use in conjunction with load_file. This method
        takes each file in the directory that contains the traces and reads each
        average queue length value in each station (and the s.c.v for the .srv file) """
        read_trace = list()
        with open(fname) as trace_files:
            for tr in trace_files:
                av_q_len = tr.strip().split()
                read_trace += [float(i) for i in av_q_len]
        return np.array(read_trace)

    def load_file(self, maxR=None, Hmax=None):
        """ This method loads the traces from a file specified in 'main.py'. We have
        that in the directory that holds the trace data there will be .mat files
        which contain the traces of each station and one .srv file that contains
        the server concurrency values that we use for training. Hence, we have to
        use different code on each file."""

        list_traces = []
        list_inputs = []

        Hmin = Hmax
        for file in os.listdir(self.directory):
            if os.path.isfile(os.path.join(self.directory, file)) and file.endswith('.srv'):
                print(f'Loading definition of server concurrency')
                self.init_s = self.read_files(os.path.join(self.directory, file))
            if os.path.isfile(os.path.join(self.directory, file)) and file.endswith('.mat') and len(list_traces) != maxR:
                print(f'Loading the trace {file}')
                ml = loadmat(os.path.join(self.directory, file))
                trace_in = ml['average_queue_length_trace']
                trace_seq = trace_in[:, 0:1]
                H = trace_in.shape[0]
                # print(H)
                if Hmin is None or Hmin > H:
                    Hmin = H
                input0 = trace_in[0, 1:]
                inputH = -np.ones((H-1, input0.shape[0]))
                input0H = np.concatenate([input0[np.newaxis], inputH])
                input_trace = np.hstack([trace_seq, input0H])
                list_traces.append(trace_in)
                list_inputs.append(input_trace)

        for i in range(len(list_traces)):
            # print(Hmin)
            list_inputs[i] = list_inputs[i][:Hmin, :]
            list_traces[i] = list_traces[i][:Hmin, :]
        self.traces = np.stack(list_traces)
        self.inputs = np.stack(list_inputs)

        print('Loading the traces has ended')

    def makeNN(self, lr):
        """ This methods details how we build the rnn using a cell created by us
        in the 'RNNCell' class """

        print('RNN is Building')
        self.cell = self.rnn_cell(self.init_s)
        tests_in = keras.Input((None, self.traces.shape[2]))
        rnn_layer = keras.layers.RNN(self.cell, return_sequences=True)
        rec = rnn_layer(tests_in)
        optimizer = keras.optimizers.Adam(lr=lr)
        self.model = Model(inputs=[tests_in], outputs=[rec])
        self.model.compile(optimizer=optimizer, metrics=[], loss=self.cell.loss)

    def learn(self):
        """ This method details how we learn the parameters """

        print('We have began learning')
        early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, min_delta=0.01)
        h = keras.callbacks.History()
        gb = GetBestModel(monitor='val_loss', verbose=1, mode='min', period=1)
        self.learn_begin = time.time()
        hist = self.model.fit(self.inputs, self.traces, epochs=500, batch_size=1, validation_split=0.5, callbacks=[gb, early_stop, h])
        self.learn_end = time.time()
        self.val_loss = hist.history['val_loss'][-1]
        print('We have finished learning')

    def saveResults(self, fname):
        """ Save the results to a text file as chosen by user in 'main.py' """
        # now = datetime.datetime.now()
        print(f"Saving results on {fname}")
        mu = self.cell.get_mu()
        p = self.cell.get_p()
        with open(fname, "w") as f:
            print(f"tbeg {self.learn_begin}", file=f)
            print(f"tend {self.learn_end}", file=f)
            print(f"elapsed {self.learn_end-self.learn_begin}", file=f)
            print(f"val_loss {self.val_loss}", file=f)
            print(f"mu {' '.join([str(i[0]) for i in mu])}", file=f)
            print(f"P {' '.join([str(j) for i in p for j in i ])}", file=f)
