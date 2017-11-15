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

mgcnn.py : The interface to the MGCNN class

"""

import joblib
import numpy as np
from scipy.sparse import csgraph
import time
import pandas as pd
import tensorflow as tf

from model import Train_test_matrix_completion
from graphutils import interaction_matrix

class UserItemGraph():
    def __init__(self, users, items):
        self.sz = (users, items)
        self.M = np.zeros(self.sz, dtype=np.float)
        self.O = np.zeros(self.sz, dtype=np.int)
        self.Otraining = np.zeros(self.sz, dtype=np.int)
        self.Otest = np.zeros(self.sz, dtype=np.int)
        self.Lcol = None
        self.Lrow = None

class MCSession():
    def __init__(self, n_iter=50):
        self.graph = None
        self.ord_col = 5
        self.ord_row = 5
        self.num_iter_test = 10
        self.num_total_iter_training = n_iter
        self.best_iter = None
        self.best_pred_error = None
        self.RMSE = None
        self.list_training_loss = None
        self.list_test_pred_error = None
        self.list_X = None
        self.persistpath = 'res/'
        self.saver = None
        self.save_path = 'res/model/model.ckpt'
        self.load_existing = False

    # M = ratings
    # O = data mask
    # Otraining = training mask
    # Otest = test mask
    # Wrow = user adjacency matrix
    # Wcol = movie adjacency matrix
    def load_dataset(self, interactions, user_count=150, item_count=200, split=0.5):
        # interactions = [us,is,rs]
        user_range = len(interactions[0])

        user_idx = interactions[0]
        item_idx = interactions[1]
        ratings  = interactions[2]

        print("Initialising model variables ...")
        uig = UserItemGraph(user_count, item_count)

        print("Building dataset ...")
        for j in range(user_range):
            u = user_idx[j]-1; i=item_idx[j]-1;
            uig.M[u,i] = ratings[j]
            uig.O[u,i] = 1

        print("Computing Leave-one-out test split ...")
        for u in user_idx:
            heldout = np.random.choice(list(uig.O[u-1, :].nonzero())[0],1)
            uig.Otest[u-1, heldout] = 1

        uig.Otraining = uig.O - uig.Otest

        print("Building user interaction matrix ...")
        # User interactions
        Wrow = np.zeros((user_count, user_count), dtype=np.int)
        Wrow = interaction_matrix(Wrow, uig.O, 1)

        print("Building item interaction matrix ...")
        # Item interactions
        Wcol = np.zeros((item_count, item_count), dtype=np.int)
        Wcol = interaction_matrix(Wcol, uig.O, 0)

        print("Computing Laplacian of interactions ...")
        uig.Lrow = csgraph.laplacian(Wrow, normed=True)
        uig.Lcol = csgraph.laplacian(Wcol, normed=True)

        self.graph = uig

    def train(self):
        print("Starting training ...")
        if self.graph is None:
            raise ValueError("Must load dataset before creating model.")

        self.learning_obj = Train_test_matrix_completion(self.graph.M, self.graph.Lrow, self.graph.Lcol, self.graph.O, self.graph.Otraining, self.graph.Otest,
                                                    order_chebyshev_col = self.ord_col, order_chebyshev_row = self.ord_row,
                                                    gamma=1e-8, learning_rate=1e-3)

        list_training_loss = list(); list_training_norm_grad = list()
        list_test_pred_error = list(); list_predictions = list()
        list_X = list(); list_X_evolutions = list()

        list_training_times = list(); list_test_times = list(); list_grad_X = list()

        num_iter = 0
        for k in range(num_iter, self.num_total_iter_training):
            tic = time.time()
            _, current_training_loss, norm_grad, X_grad = self.learning_obj.session.run([self.learning_obj.optimizer, self.learning_obj.loss,
                                                                                         self.learning_obj.norm_grad, self.learning_obj.var_grad])
            training_time = time.time() - tic

            list_training_loss.append(current_training_loss)
            list_training_norm_grad.append(norm_grad)
            list_training_times.append(training_time)

            if (np.mod(num_iter, self.num_iter_test)==0):
                msg = "[TRN] iter = %03i, cost = %3.2e, |grad| = %.2e (%3.2es)" \
                                            % (num_iter, list_training_loss[-1], list_training_norm_grad[-1], training_time)
                print(msg)

                #Test Code
                tic = time.time()
                pred_error, preds, X = self.learning_obj.session.run([self.learning_obj.predictions_error,
                                                                      self.learning_obj.predictions,
                                                                      self.learning_obj.norm_X])
                c_X_evolutions = self.learning_obj.session.run(self.learning_obj.list_X)
                list_X_evolutions.append(c_X_evolutions)

                test_time = time.time() - tic

                list_test_pred_error.append(pred_error)
                list_X.append(X)
                list_test_times.append(test_time)
                msg = "[TST] iter = %03i, cost = %3.2e (%3.2es)" % (num_iter, list_test_pred_error[-1], test_time)
                print(msg)

            num_iter += 1

        self.best_iter       = (np.where(np.asarray(list_training_loss)==np.min(list_training_loss))[0][0]//self.num_iter_test)*self.num_iter_test
        self.best_pred_error = list_test_pred_error[self.best_iter//self.num_iter_test]
        self.RMSE            = np.sqrt(np.square(self.best_pred_error)/np.sum(self.graph.Otest))

        self.list_training_loss   = list_training_loss
        self.list_test_pred_error = list_test_pred_error

        self.list_X = list_X

        print("Persisting results")
        self.persist_results()

        print("Saving model in file: %s ..." % self.save_path)
        self.saver = tf.train.Saver(self.learning_obj.vars)
        self.saver.save(self.learning_obj.session, self.save_path)

    def load_saved_model(self):
        if self.graph is None:
            raise ValueError("Must load dataset before creating model.")

        self.learning_obj = Train_test_matrix_completion(self.graph.M, self.graph.Lrow, self.graph.Lcol, self.graph.O, self.graph.Otraining, self.graph.Otest,
                                                    order_chebyshev_col = self.ord_col, order_chebyshev_row = self.ord_row,
                                                    gamma=1e-8, learning_rate=1e-3)
        try:
            self.saver = tf.train.Saver(self.learning_obj.vars)
            self.saver.restore(self.learning_obj.session, self.save_path)
        except Exception as e:
            raise ValueError("Can't load existing model")

        print("Model restored.")
        _, current_training_loss, norm_grad, X_grad = self.learning_obj.session.run([self.learning_obj.optimizer, self.learning_obj.loss,
                                                                                     self.learning_obj.norm_grad, self.learning_obj.var_grad])
        msg = "[TRN] cost = %3.2e, |grad| = %.2e" % (current_training_loss, norm_grad)
        print(msg)

        pred_error, preds, X = self.learning_obj.session.run([self.learning_obj.predictions_error,
                                                              self.learning_obj.predictions,
                                                              self.learning_obj.norm_X])
        c_X = self.learning_obj.session.run(self.learning_obj.list_X)

        msg = "[TST] cost = %3.2e" % (pred_error)
        print(msg)

        self.best_iter       = 0
        self.list_X          = [X]
        self.best_pred_error = pred_error
        self.RMSE            = np.sqrt(np.square(self.best_pred_error)/np.sum(self.graph.Otest))


    def print_results(self):
        print('Best predictions at iter: %d (error: %f)' % (self.best_iter, self.best_pred_error))
        print('RMSE: %f' % self.RMSE)

    def persist_results(self):
        joblib.dump(self.list_training_loss,   open(self.persistpath+"list_training_loss.p", "wb" ) )
        joblib.dump(self.list_test_pred_error, open(self.persistpath+"test_pred_errors.p", "wb" ) )
        joblib.dump(self.list_X,               open(self.persistpath+"list_X.p", "wb" ) )

    def load_persistent(self):
        self.list_training_loss   = joblib.load(self.persistpath+"list_training_loss.p")
        self.list_test_pred_error = joblib.load(self.persistpath+"test_pred_errors.p")
        self.list_X               = joblib.load(self.persistpath+"list_X.p")
