#Code for running CEC with ILSVRC2012 (ImageNet)
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
test_type = "half_test_REWORKED_EC_imagenet_5_tasks_small_lr" + ".txt"
sys.stdout = open("/home/ubuntu/Desktop/path/"+test_type, "w+")
# ==================================================
no_classes = 1000
features = 1000
RVI_removal = 20

#CEC Mode
option_training = "error_correction"

#CC: Load in NP arrays
np_train_and_val_images = np.load('train_np_array.npy')
np_train_and_val_labels = np.load('train_label_array.npy')
np_test_images = np.load('val_np_array.npy')
np_test_labels = np.load('val_label_array.npy')


#Get imagenet Dataset set up correctly
def split_imagenet(cond, image_sets_list, labels_sets_list):
    sets_list = []
    for i in range(3):
        class_indices = np.where((labels_sets_list[i] == cond[0]) | (labels_sets_list[i] == cond[1]) | (labels_sets_list[i] == cond[2]) | (labels_sets_list[i] == cond[3]) | (labels_sets_list[i] == cond[4]))
        temp = image_sets_list[i][class_indices,]
        x = temp[0,:,:]
        temp2 = labels_sets_list[i][class_indices,]
        ytemp = temp2[0,:]
        n_values = no_classes
        y = np.eye(n_values)[ytemp]
        sets_list.append(DataSet(x, y, dtype=dtypes.float32, reshape=False))

    imagenet_list = base.Datasets(train=sets_list[0], validation=sets_list[1], test=sets_list[2])
    return imagenet_list

def train(mnist_list):

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

    #CEC dict for storing incorrect val examples from previous tasks
    dict_incorrect_valid = {}

    with tf.Session(graph=g1, config=config) as sess1:
        # Initialize all variables
        init = [tf.global_variables_initializer(), tf.local_variables_initializer()]
        sess1.run(init)
        task_num = 200

        #For removing volatile instances
        variation_counts = [0] * task_num
        prev_val_TFs = [0] * task_num

        for j in range(0, task_num):
            print("Training Disjoint MNIST %d" % (j + 1))
            print(datetime.datetime.now())
            # Update the parameters
            epoch_owm = FLAGS.epoch
            batch_size_owm = FLAGS.batch_size
            all_data = len(mnist_list[j].train.labels[:])
            print("No task training instances", all_data)
            #CEC: Calculate how each batch should be split between training and validation examples
            no_ec_examples = 0
            no_training_per_batch = batch_size_owm
            no_ec_per_batch = 0
            if j > 0:
                for task in dict_incorrect_valid:
                    no_ec_examples += dict_incorrect_valid[task][0].size
                no_training_per_batch = math.ceil((all_data / (all_data + no_ec_examples)) *  batch_size_owm)
                no_ec_per_batch = batch_size_owm - no_training_per_batch
                all_data += no_ec_examples
            all_step = all_data*epoch_owm//batch_size_owm
            #CEC: number of batches for current training task
            if all_data % batch_size_owm == 0:
                no_batches = all_data//batch_size_owm
            else:
                no_batches = all_data//batch_size_owm + 1

            #CEC: Checks if variation examples have all been added
            finished = False
            #CEC: Get the misclassified validation instances for all previous tasks
            if j > 0:
                all_incorrect_xs = np.empty(shape=(0,mnist_list[0].train.images.shape[1]))
                all_incorrect_ys = np.empty(shape=(0,mnist_list[0].train.labels.shape[1]))
                for k in range(j):
                    task_incorrect_xs, task_incorrect_ys = mnist_list[k].validation.images[dict_incorrect_valid[k][0]], mnist_list[k].validation.labels[dict_incorrect_valid[k][0]]
                    all_incorrect_xs = np.vstack([all_incorrect_xs, task_incorrect_xs])
                    all_incorrect_ys = np.vstack([all_incorrect_ys, task_incorrect_ys])
                #CC: Shuffle ALL misclassifed examples in unison
                assert len(all_incorrect_xs) == len(all_incorrect_ys)
                p = np.random.permutation(len(all_incorrect_xs))
                all_incorrect_xs = all_incorrect_xs[p]
                all_incorrect_ys = all_incorrect_ys[p]
            incorrect_index = 0
            batch_incorrect_xs = []
            for current_step in range(all_step):
                lamda = current_step/all_step
                current_step = current_step+1
                batch_xs, batch_ys = mnist_list[j].train.next_batch(no_training_per_batch)
                batch_incorrect_xs = np.empty(shape=(0,batch_xs.shape[1]))
                batch_incorrect_ys = np.empty(shape=(0,batch_ys.shape[1]))
                if finished == False and j > 0:
                    #CEC: Get batch of incorrect examples
                    if (incorrect_index + no_ec_per_batch) > len(all_incorrect_ys):
                        batch_incorrect_xs = all_incorrect_xs[incorrect_index:]
                        batch_incorrect_ys = all_incorrect_ys[incorrect_index:]
                        finished = True
                        incorrect_index = 0
                    else:
                        batch_incorrect_xs = all_incorrect_xs[incorrect_index:(incorrect_index + no_ec_per_batch)]
                        batch_incorrect_ys = all_incorrect_ys[incorrect_index:(incorrect_index + no_ec_per_batch)]
                        incorrect_index += no_ec_per_batch

                #CEC: Combine training batch and incorrect batch to get full batch
                batch_xs = np.vstack([batch_xs, batch_incorrect_xs])
                batch_ys = np.vstack([batch_ys, batch_incorrect_ys])

                #CEC: Reset Finished for new epoch
                if current_step % no_batches == 0:
                    finished = False

                feed_dict = {
                    OWM.input_x: batch_xs,
                    OWM.input_y: batch_ys,
                    OWM.input_x_validation_incorrect: batch_incorrect_xs,
                    OWM.input_y_validation_incorrect: batch_incorrect_ys,
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

            print("Validation on Previous Datasets:")
            for i_val in range(j + 1):
                feed_dict = {
                    OWM.input_x: mnist_list[i_val].validation.images[:],
                    OWM.input_y: mnist_list[i_val].validation.labels[:],
                }

                accu, loss, predicted_validation_examples = sess1.run([OWM.accuracy, OWM.loss, OWM.predicted_validation_examples], feed_dict)
                print("Validation:->>>[{:d}/{:d}], acc: {:.2f} %".format(i_val + 1, task_num, accu * 100))


                list_val_TFs = list(predicted_validation_examples)

                #CC: Update variation counts
                if i_val != j:
                    for i_TF in range(len(list_val_TFs)):
                        if list_val_TFs[i_TF] != prev_val_TFs[i_val][i_TF]:
                            variation_counts[i_val][i_TF] += 1
                else:
                    variation_counts[j] = [0] * len(list_val_TFs)

                prev_val_TFs[i_val] = list_val_TFs


                #Removal of Volatile Instances ON/OFF
                dict_incorrect_valid[i_val] = np.where((predicted_validation_examples == False))
                    # & ((np.asarray(variation_counts[i_val])) < RVI_removal))

        feed_dict = {
            OWM.input_x: np_test_images[:],
            OWM.input_y: all_labels[:],
        }
        accu, loss = sess1.run([OWM.accuracy, OWM.loss], feed_dict)
        print("Average Final Test Accuracy over all Tasks {:g} %\n".format(accu * 100))
        print(datetime.datetime.now(), "END")


#Set up for samples/trials
def main(_):
    p = 0
    assert len(np_train_and_val_images) == len(np_train_and_val_labels)
    #Set seeds
    random.seed(4)
    np.random.seed(4)
    #Training/validation splits
    for n in range(5):
        p = np.random.permutation(len(np_train_and_val_images))
        np_train_and_val_images_rp = np_train_and_val_images[p]
        np_train_and_val_labels_rp = np_train_and_val_labels[p]
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
