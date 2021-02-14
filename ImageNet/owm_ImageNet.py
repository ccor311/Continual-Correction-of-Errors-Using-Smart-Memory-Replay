#Code for running OWM with ILSVRC2012 (ImageNet)
import tensorflow as tf
import numpy as np
import datetime
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.examples.tutorials.mnist import input_data
from use_OWMLayer_2Layers  import OWMLayer
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ignore warning
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use gpu
os.chdir("/home/ubuntu/Desktop/path")
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
tf.flags.DEFINE_integer("num_class", 1000, "")
tf.flags.DEFINE_integer("batch_size", 30, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("epoch", 20, "")
FLAGS = tf.flags.FLAGS
# ==================================================
test_type = "OWM_Imagenet" + ".txt"
sys.stdout = open("/home/ubuntu/Desktop/path/"+test_type, "w+")
# ==================================================

no_classes = 1000
features = 1000

#CC: Load in NP arrays
np_train_and_val_images = np.load('train_np_array.npy')
np_train_and_val_labels = np.load('train_label_array.npy')
np_test_images = np.load('val_np_array.npy')
np_test_labels = np.load('val_label_array.npy')

#Get imagenet Dataset set up correctly
def split_imagenet(cond, image_sets_list, labels_sets_list):
    sets_list = []
    for i in range(3):
        #CC: Get current class labels
        class_indices = np.where((labels_sets_list[i] == cond[0]) | (labels_sets_list[i] == cond[1]) | (labels_sets_list[i] == cond[2]) | (labels_sets_list[i] == cond[3]) | (labels_sets_list[i] == cond[4]))
        #CC: Get images in x*1000 format
        temp = image_sets_list[i][class_indices,]
        x = temp[0,:,:]
        #CC: Get images and labels in one_hot for final accuracy tests (feed dict)
        temp2 = labels_sets_list[i][class_indices,]
        ytemp = temp2[0,:]
        n_values = no_classes
        y = np.eye(n_values)[ytemp]
        #2d labels
        sets_list.append(DataSet(x, y, dtype=dtypes.float32, reshape=False))

    imagenet_list = base.Datasets(train=sets_list[0], validation=sets_list[1], test=sets_list[2])
    return imagenet_list

def train(mnist_list):

    #Get ALL imagenet test set data in one_hot format for use with feed_dict
    n_values = np.max(np_test_labels) + 1
    all_labels = np.eye(n_values)[np_test_labels]
    # Training
    # ==================================================
    g1 = tf.Graph()
    middle = 800
    with g1.as_default():
        OWM = OWMLayer([[features + 1, middle], [middle + 1, no_classes]], seed_num=79)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=g1, config=config) as sess1:
        # Initialize all variables
        init = [tf.global_variables_initializer(), tf.local_variables_initializer()]
        sess1.run(init)
        task_num = 200
        for j in range(0, task_num):
            print("Training Disjoint MNIST %d" % (j + 1))
            #CC: Print timestamp
            print(datetime.datetime.now())
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
                    OWM.lr_array: np.array([[0.1]]),
                    OWM.alpha_array: np.array([[1.0 * 0.005 ** lamda, 1.0]]),
                }
                acc, loss,  _, = sess1.run([OWM.accuracy, OWM.loss, OWM.back_forward], feed_dict,)
                if current_step % (all_step//20) == 0:
                    feed_dict = {
                        OWM.input_x: np_test_images[:],
                        OWM.input_y: all_labels[:],
                    }
                    acc, loss = sess1.run([OWM.accuracy, OWM.loss], feed_dict)

            print("Test on Previous Datasets:")
            for i_test in range(j + 1):
                feed_dict = {
                    OWM.input_x: mnist_list[i_test].test.images[:],
                    OWM.input_y: mnist_list[i_test].test.labels[:],
                }
                accu, loss = sess1.run([OWM.accuracy, OWM.loss], feed_dict)
                print("Test:->>>[{:d}/{:d}], acc: {:.2f} %".format(i_test + 1, task_num, accu * 100))
            #DBP validation accuracy
            print("Validation on Previous Datasets:")
            for i_val in range(j + 1):
                feed_dict = {
                    OWM.input_x: mnist_list[i_val].validation.images[:],
                    OWM.input_y: mnist_list[i_val].validation.labels[:],
                }
                #DBP include correct and incorrect validation examples indices
                accu, loss, predicted_validation_examples = sess1.run([OWM.accuracy, OWM.loss, OWM.predicted_validation_examples], feed_dict)
                print("Validation:->>>[{:d}/{:d}], acc: {:.2f} %".format(i_val + 1, task_num, accu * 100))
        feed_dict = {
            OWM.input_x: np_test_images[:],
            OWM.input_y: all_labels[:],
        }
        accu, loss = sess1.run([OWM.accuracy, OWM.loss], feed_dict)
        print("Average Final Test Accuracy over all Tasks {:g} %\n".format(accu * 100))
        print(datetime.datetime.now(), "END")


#Set up for 50 samples/trials
def main(_):
    p = 0
    assert len(np_train_and_val_images) == len(np_train_and_val_labels)
    #Set Seeds
    random.seed(4)
    np.random.seed(4)
    #Training/validation splits
    for n in range(5):
        p = np.random.permutation(len(np_train_and_val_images))
        np_train_and_val_images_rp = np_train_and_val_images[p]
        np_train_and_val_labels_rp = np_train_and_val_labels[p]
        #Split train and val
        np_train_images = np_train_and_val_images_rp[:881167]
        np_train_labels = np_train_and_val_labels_rp[:881167]
        np_val_images = np_train_and_val_images_rp[881167:969284]
        np_val_labels = np_train_and_val_labels_rp[881167:969284]

        image_sets_list = [np_train_images, np_val_images, np_test_images]
        labels_sets_list = [np_train_labels, np_val_labels, np_test_labels]
        task_order = [t for t in range(1000)]
        mnist_list = []
        #Randomised task orders for each training/validation split
        for s in range(0,10):
            print("Training sample:", n, "Order sample:", s)
            random.shuffle(task_order)
            mnist_list = [split_imagenet(task_order[0:5], image_sets_list, labels_sets_list)]
            for i in range(5, no_classes, 5):
                mnist_list.append(split_imagenet(task_order[i:(i+5)], image_sets_list, labels_sets_list))
            train(mnist_list)

if __name__ == '__main__':
    tf.app.run()
