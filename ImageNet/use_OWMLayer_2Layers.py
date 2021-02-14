# -*- coding: utf-8 -*-
import tensorflow as tf
from numpy.random import seed
from tensorflow import set_random_seed


class OWMLayer(object):

    def __init__(self, shape_list, seed_num=0):
        seed(seed_num)
        set_random_seed(seed_num)
        # Placeholders for input, output and dropout
        sequence_length = 1000
        num_classes = 1000
        self.shape_list = shape_list
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.lr_array = tf.placeholder(tf.float32, name="lr_array")
        self.alpha_array = tf.placeholder(tf.float32, name="alpha_array")
        initializer = tf.contrib.layers.xavier_initializer()

        self.input_x_validation_incorrect = tf.placeholder(tf.float32, [None, sequence_length], name="input_x_validation_incorrect")
        self.input_y_validation_incorrect = tf.placeholder(tf.float32, [None, num_classes], name="input_y_validation_incorrect")
        #end DBP

        with tf.name_scope("input"):
            self.P1 = tf.Variable(tf.eye(int(shape_list[0][0])))

            w1 = tf.get_variable("w1", shape=shape_list[0], initializer=initializer)
            y_1 = tf.concat([self.input_x, tf.tile(tf.ones([1, 1]), [tf.shape(self.input_x)[0], 1])], 1)
            r = tf.reduce_mean(y_1, 0, keep_dims=True)
            k = tf.matmul(self.P1, tf.transpose(r))
            self.dela_P1 = tf.divide(tf.matmul(k, tf.transpose(k)), self.alpha_array[0][0] + tf.matmul(r, k))
            self.P1 = tf.assign_sub(self.P1, self.dela_P1)

            y1 = tf.nn.relu(tf.matmul(y_1, w1, name="y1"))

        with tf.name_scope("output"):
            self.P2 = tf.Variable(tf.eye(int(shape_list[1][0])))
            w2 = tf.get_variable("w2", shape=shape_list[1], initializer=initializer)
            y_2 = tf.concat([y1, tf.tile(tf.ones([1, 1]), [tf.shape(y1)[0], 1])], 1)
            r = tf.reduce_mean(y_2, 0, keep_dims=True)
            k = tf.matmul(self.P2, tf.transpose(r))
            self.dela_P2 = tf.divide(tf.matmul(k, tf.transpose(k)), self.alpha_array[0][1] + tf.matmul(r, k))
            self.P2 = tf.assign_sub(self.P2, self.dela_P2)

            y2 = tf.matmul(y_2, w2, name="y2")

        scores = y2
        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)

        # Accuracy
        with tf.name_scope("accuracy"):
            predictions = tf.argmax(scores, 1, name="predictions")
            correct_predictions = tf.equal(predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))

        with tf.name_scope("predicted_validation_examples"):
            predictions = tf.argmax(scores, 1, name="predictions")
            self.predicted_validation_examples = tf.equal(predictions, tf.argmax(self.input_y, 1))

        self.optimizer = tf.train.MomentumOptimizer(self.lr_array[0][0], momentum=0.9)

        # back_forward
        grads_and_vars = self.optimizer.compute_gradients(self.loss, var_list=[w1, w2])
        for i, (g, v) in enumerate(grads_and_vars):
            if g is not None:
                grads_and_vars[i] = (tf.clip_by_norm(g, 10), v)
        grad_v_input = [self.owm(self.P1, grads_and_vars[0])]
        grad_v_out = [self.owm(self.P2, grads_and_vars[1])]
        self.back_forward = self.optimizer.apply_gradients([grad_v_input[0], grad_v_out[0]])

    def owm(self, P, g_v, lr=2.0):
        g_ = lr * tf.matmul(P, g_v[0])
        return g_, g_v[1]

    def owm_error(self, P, Q, g_v, lr=0.5):
        g_ = lr * tf.matmul(1/Q, tf.matmul(P, g_v[0]))
        return g_, g_v[1]
