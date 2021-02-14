# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import datetime
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.examples.tutorials.mnist import input_data
from OWMLayer_3Layers_ErrorCorrection import OWMLayer
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ignore warning
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use gpu0,1
import random
import sys
import math
import datetime

# Parameters
# ==================================================
tf.flags.DEFINE_integer("num_checkpoints", 3, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.app.flags.DEFINE_string("buckets", "", "")
tf.app.flags.DEFINE_string("checkpointDir", "", "oss info")
tf.flags.DEFINE_integer("num_class", 10, "")
tf.flags.DEFINE_integer("batch_size", 40, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("epoch", 20, "")
FLAGS = tf.flags.FLAGS
# ==================================================
test_type = "OWM_MNIST_RESULTS" + ".txt"
sys.stdout = open("/home/ubuntu/Desktop/mnistFixed/"+test_type, "w+")
# ==================================================
no_classes = 10
features = 784

mnist = input_data.read_data_sets("./data/MNIST_data/", one_hot=True)

#Sample size for training/validation sets
train_sample = 0.05
val_sample = 0.10


#Sample Seeding
random.seed(4)
np.random.seed(4)

#Get imagenet Dataset set up correctly
def split_mnist(mnist, cond, sample_size = 1):
    sets = ["train"]
    sets_list = []
    #random sample
    for set_name in sets:
        if set_name == "train":
            this_set = getattr(mnist, set_name)
            maxlabels = np.argmax(this_set.labels, 1)
            r = random.sample(list(np.arange(0, this_set.images[cond(maxlabels),:].shape[0], 1)), int(this_set.images[cond(maxlabels),:].shape[0] * sample_size))
            dataset = DataSet(this_set.images[cond(maxlabels),:][r], this_set.labels[cond(maxlabels)][r],
                                     dtype=dtypes.uint8, reshape=False)
            sets_list.append(dataset)
    return sets_list

#Get MNIST validation/test datasets in suitable format and sample
def split_val(mnist, cond, validation_sample_size = 1):
    sets = ["validation", "test"]
    sets_list = []
    #random 10% sample
    for set_name in sets:
        if set_name == "validation":
            this_set = getattr(mnist, set_name)
            maxlabels = np.argmax(this_set.labels, 1)
            r = random.sample(list(np.arange(0, this_set.images[cond(maxlabels),:].shape[0], 1)), int(this_set.images[cond(maxlabels),:].shape[0] * validation_sample_size))
            dataset = DataSet(this_set.images[cond(maxlabels),:][r], this_set.labels[cond(maxlabels)][r],
                                     dtype=dtypes.uint8, reshape=False)
            sets_list.append(dataset)
        else:
            this_set = getattr(mnist, set_name)
            maxlabels = np.argmax(this_set.labels, 1)
            dataset = DataSet(this_set.images[cond(maxlabels),:], this_set.labels[cond(maxlabels)],
                                     dtype=dtypes.uint8, reshape=False)
            sets_list.append(dataset)
    return sets_list


#Train Model
def train(mnist_list):
    # Training
    # ==================================================
    g1 = tf.Graph()
    middle = 800
    with g1.as_default():
        OWM = OWMLayer([[features + 1, middle], [middle + 1, middle], [middle + 1, no_classes]], seed_num=79)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    test_array = []
    with tf.Session(graph=g1, config=config) as sess1:
        # Initialize all variables
        init = [tf.global_variables_initializer(), tf.local_variables_initializer()]
        sess1.run(init)
        task_num = 10
        for j in range(0, task_num):
            print("Training Disjoint MNIST %d" % (j + 1))
            # Update the parameters
            epoch_owm = FLAGS.epoch
            batch_size_owm = FLAGS.batch_size
            all_data = len(mnist_list[j].train.labels[:])
            all_step = all_data*epoch_owm//batch_size_owm
            for current_step in range(all_step):
                lamda = current_step/all_step
                current_step = current_step+1
                batch_xs, batch_ys = mnist_list[j].train.next_batch(batch_size_owm)
                feed_dict = {
                    OWM.input_x: batch_xs,
                    OWM.input_y: batch_ys,
                    OWM.lr_array: np.array([[0.2]]),
                    OWM.alpha_array: np.array([[0.9 * 0.001 ** lamda, 1.0 * 0.1 ** lamda, 0.6]]),
                }
                acc, loss,  _, = sess1.run([OWM.accuracy, OWM.loss, OWM.back_forward], feed_dict,)

            print("Test on Previous Datasets:")
            for i_test in range(j + 1):
                feed_dict = {
                    OWM.input_x: mnist_list[i_test].test.images[:],
                    OWM.input_y: mnist_list[i_test].test.labels[:],
                }
                accu, loss = sess1.run([OWM.accuracy, OWM.loss], feed_dict)
                print("Test:->>>[{:d}/{:d}], acc: {:.2f} %".format(i_test + 1, task_num, accu * 100))

            print("Validation on Previous Datasets:")
            for i_val in range(j + 1):
                feed_dict = {
                    OWM.input_x: mnist_list[i_val].validation.images[:],
                    OWM.input_y: mnist_list[i_val].validation.labels[:],
                }

                accu, loss, predicted_validation_examples = sess1.run([OWM.accuracy, OWM.loss, OWM.predicted_validation_examples], feed_dict)
                print("Validation:->>>[{:d}/{:d}], acc: {:.2f} %".format(i_val + 1, task_num, accu * 100))

        feed_dict = {
            OWM.input_x: mnist.test.images[:],
            OWM.input_y: mnist.test.labels[:],
        }
        accu, loss = sess1.run([OWM.accuracy, OWM.loss], feed_dict)
        print("Average Final Test Accuracy over all Tasks {:g} %\n".format(accu * 100))


def main(_):
    #Training samples
    for y in range(1,31):
        print("Starting Training sample", y)
        train_list = [0] * no_classes
        #Get training data
        for i in range(no_classes):
            train_list[i] = split_mnist(mnist, lambda x: x == i, train_sample)
        #Validation samples
        for z in range(1, 31):
            print("Training sample:", y, "Validation sample:", z)
            val_list = [0] * no_classes
            dataset = [0] * no_classes
            for j in range(no_classes):
                val_list[j] = split_val(mnist, lambda x: x == j, val_sample)
                dataset[j] = base.Datasets(train=train_list[j][0], validation=val_list[j][0], test=val_list[j][1])
            mnist_list_temp = [dataset[0], dataset[1], dataset[2], dataset[3], dataset[4], dataset[5], dataset[6], dataset[7], dataset[8], dataset[9]]
            #Random task order for each new training set
            if z == 1:
                random.shuffle(mnist_list_temp)
                mnist_list = mnist_list_temp
            train(mnist_list)


if __name__ == '__main__':
    tf.app.run()
