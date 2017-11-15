"""
MGCNN : Multi-Graph Convolutional Neural Networks

The code contained in this repository represents a TensorFlow implementation of the Recurrent Muli-Graph Convolutional Neural Network depicted in:

Geometric Matrix Completion with Recurrent Multi-Graph Neural Networks (in Proc. NIPS, 2017)
Federico Monti, Michael M. Bronstein, Xavier Bresson

https://arxiv.org/abs/1704.06803

License : GNU General Public License v3.0
by @fmonti (Frederico Monti)

Modifications : @dtsbourg (Dylan Bourgeois)
This code is an attempt to package the code presented in
https://github.com/fmonti/mgcnn for the Netflix challenge.

---

model.py : the neural network model

"""

import os,sys,inspect
import os
import joblib
import tensorflow as tf
import numpy as np
import h5py
import scipy.sparse.linalg as la
from scipy.sparse import csgraph
import scipy
import time
import pandas as pd


class Train_test_matrix_completion:
    """
    The neural network model.
    """
    def frobenius_norm(self, tensor):
        square_tensor = tf.square(tensor)
        tensor_sum = tf.reduce_sum(square_tensor)
        frobenius_norm = tf.sqrt(tensor_sum)
        return frobenius_norm


    def bid_conv(self, W, b):
        X = tf.reshape(self.X, [tf.shape(self.M)[0], tf.shape(self.M)[1]])

        feat = []
        #collect features
        for k_r in range(self.ord_row):
            for k_c in range(self.ord_col):
                row_lap = self.list_row_cheb_pol[k_r]
                col_lap = self.list_col_cheb_pol[k_c]

                #dense implementation
                c_feat = tf.matmul(row_lap, X, a_is_sparse=False)
                c_feat = tf.matmul(c_feat, col_lap, b_is_sparse=False)
                feat.append(c_feat)

        all_feat = tf.stack(feat, 2)
        all_feat = tf.reshape(all_feat, [-1, self.ord_row*self.ord_col])
        conv_feat = tf.matmul(all_feat, W) + b
        conv_feat = tf.nn.relu(conv_feat)
        conv_feat = tf.reshape(conv_feat, [tf.shape(self.M)[0], tf.shape(self.M)[1], self.n_conv_feat])
        return conv_feat

    def compute_cheb_polynomials(self, L, ord_cheb, list_cheb):
        for k in range(ord_cheb):
            if (k==0):
                list_cheb.append(tf.cast(tf.diag(tf.ones([tf.shape(L)[0],])), 'float32'))
            elif (k==1):
                list_cheb.append(tf.cast(L, 'float32'))
            else:
                list_cheb.append(2*tf.matmul(L, list_cheb[k-1])  - list_cheb[k-2])


    def __init__(self, M, Lr, Lc, Odata, Otraining, Otest, order_chebyshev_col = 5, order_chebyshev_row = 5,
                 num_iterations = 10, gamma=1.0, learning_rate=1e-4, idx_gpu = '/gpu:2'):

        #order of the spectral filters
        self.ord_col = order_chebyshev_col
        self.ord_row = order_chebyshev_row
        self.num_iterations = num_iterations
        self.n_conv_feat = 32

        with tf.Graph().as_default() as g:
                tf.logging.set_verbosity(tf.logging.ERROR)
                self.graph = g
                tf.set_random_seed(0)
                with tf.device(idx_gpu):

                        #loading of the laplacians
                        self.Lr = tf.constant(Lr.astype('float32'))
                        self.Lc = tf.constant(Lc.astype('float32'))

                        self.norm_Lr = self.Lr - tf.diag(tf.ones([Lr.shape[0], ]))
                        self.norm_Lc = self.Lc - tf.diag(tf.ones([Lc.shape[0], ]))
                        #compute all chebyshev polynomials a priori
                        self.list_row_cheb_pol = list()
                        self.compute_cheb_polynomials(self.norm_Lr, self.ord_row, self.list_row_cheb_pol)
                        self.list_col_cheb_pol = list()
                        self.compute_cheb_polynomials(self.norm_Lc, self.ord_col, self.list_col_cheb_pol)

                        #definition of constant matrices
                        self.M = tf.constant(M, dtype=tf.float32)
                        self.Odata = tf.constant(Odata, dtype=tf.float32)
                        self.Otraining = tf.constant(Otraining, dtype=tf.float32) #training mask
                        self.Otest = tf.constant(Otest, dtype=tf.float32) #test mask

                        #definition of the NN variables
                        self.W_conv = tf.get_variable("W_conv", shape=[self.ord_col*self.ord_row, self.n_conv_feat], initializer=tf.contrib.layers.xavier_initializer())
                        self.b_conv = tf.Variable(tf.zeros([self.n_conv_feat,]))

                        #recurrent N parameters
                        self.W_f = tf.get_variable("W_f", shape=[self.n_conv_feat, self.n_conv_feat], initializer=tf.contrib.layers.xavier_initializer())
                        self.W_i = tf.get_variable("W_i", shape=[self.n_conv_feat, self.n_conv_feat], initializer=tf.contrib.layers.xavier_initializer())
                        self.W_o = tf.get_variable("W_o", shape=[self.n_conv_feat, self.n_conv_feat], initializer=tf.contrib.layers.xavier_initializer())
                        self.W_c = tf.get_variable("W_c", shape=[self.n_conv_feat, self.n_conv_feat], initializer=tf.contrib.layers.xavier_initializer())
                        self.U_f = tf.get_variable("U_f", shape=[self.n_conv_feat, self.n_conv_feat], initializer=tf.contrib.layers.xavier_initializer())
                        self.U_i = tf.get_variable("U_i", shape=[self.n_conv_feat, self.n_conv_feat], initializer=tf.contrib.layers.xavier_initializer())
                        self.U_o = tf.get_variable("U_o", shape=[self.n_conv_feat, self.n_conv_feat], initializer=tf.contrib.layers.xavier_initializer())
                        self.U_c = tf.get_variable("U_c", shape=[self.n_conv_feat, self.n_conv_feat], initializer=tf.contrib.layers.xavier_initializer())
                        self.b_f = tf.Variable(tf.zeros([self.n_conv_feat,]))
                        self.b_i = tf.Variable(tf.zeros([self.n_conv_feat,]))
                        self.b_o = tf.Variable(tf.zeros([self.n_conv_feat,]))
                        self.b_c = tf.Variable(tf.zeros([self.n_conv_feat,]))

                        #output parameters
                        self.W_out = tf.get_variable("W_out", shape=[self.n_conv_feat,1], initializer=tf.contrib.layers.xavier_initializer())
                        self.b_out = tf.Variable(tf.zeros([1,1]))

                        #########definition of the NN
                        self.X = tf.multiply(self.M, self.Odata) #we may initialize it at random here
                        self.list_X = list()
                        self.list_X.append(tf.identity(self.X))
                        self.X = tf.reshape(self.X, [-1,])

                        #RNN
                        self.h = tf.zeros([M.shape[0]*M.shape[1], self.n_conv_feat])
                        self.c = tf.zeros([M.shape[0]*M.shape[1], self.n_conv_feat])

                        for k in range(self.num_iterations):
                            #bidimensional convolution
                            self.x_conv = self.bid_conv(self.W_conv, self.b_conv) #N, N, n_conv_feat
                            self.x_conv = tf.reshape(self.x_conv, [-1, self.n_conv_feat])

                            self.f = tf.sigmoid(tf.matmul(self.x_conv, self.W_f) + tf.matmul(self.h, self.U_f) + self.b_f)
                            self.i = tf.sigmoid(tf.matmul(self.x_conv, self.W_i) + tf.matmul(self.h, self.U_i) + self.b_i)
                            self.o = tf.sigmoid(tf.matmul(self.x_conv, self.W_o) + tf.matmul(self.h, self.U_o) + self.b_o)

                            self.update_c = tf.sigmoid(tf.matmul(self.x_conv, self.W_c) + tf.matmul(self.h, self.U_c) + self.b_c)
                            self.c = tf.multiply(self.f, self.c) + tf.multiply(self.i, self.update_c)
                            self.h = tf.multiply(self.o, tf.sigmoid(self.c))

                            #compute update of matrix X
                            self.delta_x = tf.tanh(tf.matmul(self.c, self.W_out) + self.b_out)
                            self.X += tf.squeeze(self.delta_x)
                            self.list_X.append(tf.identity(tf.reshape(self.X, [tf.shape(self.M)[0], tf.shape(self.M)[1]])))


                        self.X = tf.reshape(self.X, [tf.shape(self.M)[0], tf.shape(self.M)[1]])
                        #########loss definition

                        #computation of the accuracy term
                        self.norm_X = 1+4*(self.X-tf.reduce_min(self.X))/(tf.reduce_max(self.X-tf.reduce_min(self.X)))
                        frob_tensor = tf.multiply(self.Otraining + self.Odata, self.norm_X - M)
                        self.loss_frob = tf.square(self.frobenius_norm(frob_tensor))/np.sum(Otraining+Odata)

                        #computation of the regularization terms
                        trace_col_tensor = tf.matmul(tf.matmul(self.X, self.Lc), self.X, transpose_b=True)
                        self.loss_trace_col = tf.trace(trace_col_tensor)
                        trace_row_tensor = tf.matmul(tf.matmul(self.X, self.Lr, transpose_a=True), self.X)
                        self.loss_trace_row = tf.trace(trace_row_tensor)

                        #training loss definition
                        self.loss = self.loss_frob + (gamma/2)*(self.loss_trace_col + self.loss_trace_row)

                        #test loss definition
                        self.predictions = tf.multiply(self.Otest, self.norm_X - self.M)
                        self.predictions_error = self.frobenius_norm(self.predictions)

                        #definition of the solver
                        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

                        self.var_grad = tf.gradients(self.loss, tf.trainable_variables())
                        self.norm_grad = self.frobenius_norm(tf.concat([tf.reshape(g, [-1]) for g in self.var_grad], 0))

                        # Create a session for running Ops on the Graph.
                        config = tf.ConfigProto(allow_soft_placement = True)
                        config.gpu_options.allow_growth = True
                        self.session = tf.Session(config=config)

                        # Run the Op to initialize the variables.
                        init = tf.initialize_all_variables()
                        self.session.run(init)
