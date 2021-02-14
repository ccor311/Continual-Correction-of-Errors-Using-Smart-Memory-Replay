# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import datetime
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.examples.tutorials.mnist import input_data
from OWMLayer_3Layers_ErrorCorrection  import OWMLayer
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ignore warning
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use gpu
#os.chdir("/home/ubuntu/Desktop/imagenetMNIST")
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
test_type = "CEC_MNIST_Results" + ".txt"
sys.stdout = open("/home/ubuntu/Desktop/mnistFixed/"+test_type, "w+")
# ==================================================
no_classes = 10
features = 784
#RVI_removal = 3
mnist = input_data.read_data_sets("./data/MNIST_data/", one_hot=True)

#Sample size for training/validation sets
train_sample = 0.05
val_sample = 0.10

#Sample seeding
random.seed(4)
np.random.seed(4)



#Get MNIST training dataset in suitable format and sample
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


#Train model
def train(mnist_list):
    # Training
    # ==================================================
    g1 = tf.Graph()
    middle = 800
    with g1.as_default():
        OWM = OWMLayer([[features + 1, middle], [middle + 1, middle], [middle + 1, no_classes]], seed_num=79)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    #Dict for storing incorrect val examples from previous tasks
    dict_incorrect_valid = {}

    with tf.Session(graph=g1, config=config) as sess1:
        # Initialize all variables
        init = [tf.global_variables_initializer(), tf.local_variables_initializer()]
        sess1.run(init)
        task_num = 10

        #For Removal of Volatile Instances
        variation_counts = [0] * task_num
        prev_val_TFs = [0] * task_num

        for j in range(0, task_num):
            print("Training Disjoint MNIST %d" % (j + 1))
            # Update the parameters
            epoch_owm = FLAGS.epoch
            batch_size_owm = FLAGS.batch_size
            all_data = len(mnist_list[j].train.labels[:])
            #CC: Calculate how each batch should be split between training and validation examples
            no_ec_examples = 0
            no_training_per_batch = batch_size_owm
            no_ec_per_batch = 0
            if j > 0:
                for task in dict_incorrect_valid:
                    no_ec_examples += dict_incorrect_valid[task][0].size
                no_training_per_batch = math.ceil((all_data / (all_data + no_ec_examples)) *  batch_size_owm)
                no_ec_per_batch = batch_size_owm - no_training_per_batch
                all_data += no_ec_examples

            #For checking number of CEC and training instances
            #print("No overall task training instances:", all_data)
            #print("No EC training instances", no_ec_examples)
            #print("No EC instances per batch", no_ec_per_batch)
            #print("No training instances per batch", no_training_per_batch)

            all_step = all_data*epoch_owm//batch_size_owm
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
                    #if dict_incorrect_valid[0]:
                    task_incorrect_xs, task_incorrect_ys = mnist_list[k].validation.images[dict_incorrect_valid[k][0]], mnist_list[k].validation.labels[dict_incorrect_valid[k][0]]
                    all_incorrect_xs = np.vstack([all_incorrect_xs, task_incorrect_xs])
                    all_incorrect_ys = np.vstack([all_incorrect_ys, task_incorrect_ys])
                #CEC: Shuffle all misclassifed examples in unison
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
                    #EC: incorrect validation examples
                    OWM.input_x_validation_incorrect: batch_incorrect_xs,
                    OWM.input_y_validation_incorrect: batch_incorrect_ys,
                    #end EC
                    OWM.lr_array: np.array([[0.2]]),
                    OWM.alpha_array: np.array([[0.9 * 0.001 ** lamda, 1.0 * 0.1 ** lamda, 0.6]]),
                }
                acc, loss,  _, = sess1.run([OWM.accuracy, OWM.loss, OWM.back_forward], feed_dict,)
                if current_step % (all_step//20) == 0:
                    feed_dict = {
                        OWM.input_x: mnist.test.images[:],
                        OWM.input_y: mnist.test.labels[:],
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


                #CEC: Keep track of classification variation
                list_val_TFs = list(predicted_validation_examples)
                if i_val != j:
                    for i_TF in range(len(list_val_TFs)):
                        if list_val_TFs[i_TF] != prev_val_TFs[i_val][i_TF]:
                            variation_counts[i_val][i_TF] += 1
                else:
                    variation_counts[j] = [0] * len(list_val_TFs)

                prev_val_TFs[i_val] = list_val_TFs


                #Removal of Volatile Instances ON/OFF
                dict_incorrect_valid[i_val] = np.where((predicted_validation_examples == False))
                    #& ((np.asarray(variation_counts[i_val])) < RPI_removal))

        feed_dict = {
            OWM.input_x: mnist_list[i_test].test.images[:],
            OWM.input_y: mnist_list[i_test].test.labels[:],
        }
        accu, loss = sess1.run([OWM.accuracy, OWM.loss], feed_dict)
        print("Average Final Test Accuracy over all Tasks {:g} %\n".format(accu * 100))


def main(_):
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
