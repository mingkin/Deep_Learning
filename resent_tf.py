# -*- coding: utf-8 -*-
"""
# Author  : Ming
# File    : {NAME}.py
# Time    : 2019/4/2 0002 下午 2:25
"""

import tensorflow as tf
from resnets_utils import *
import os
import numpy as np
import datetime


class Resnet():
    def __init__(self):
        self.classes = 6
        self.initializer = tf.random_normal_initializer(stddev=0.1)
        self.input_x = tf.placeholder(tf.float32, [None, 64, 64, 3], name="input_x")  # X
        self.input_y = tf.placeholder(tf.int32, [None, self.classes], name="input_y")
        self.is_training = True
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.logits = self.ResNet50()
        self.loss = self.loss()
        self.predictions = self.pred()
        self.accuracy = self.accury()
        self.train, self.train_summary_op = self.trace_process()


    def _identity_block(self, X, f, filters, stage, block):
        """
            Implementation of the identity block as defined in Figure 3

            Arguments:
            X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
            f -- integer, specifying the shape of the middle CONV's window for the main path
            filters -- python list of integers, defining the number of filters in the CONV layers of the main path
            stage -- integer, used to name the layers, depending on their position in the network
            block -- string/character, used to name the layers, depending on their position in the network

            Returns:
            X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
            filter：相当于CNN中的卷积核，它要求是一个Tensor，具有[filter_height, filter_width, in_channels, out_channels]
            这样的shape，具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，要求类型与参数input相同，有一个地方需要注意，
            第三维in_channels，就是参数input的第四维
            strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4
            """

        # defining name basis
        with tf.variable_scope('identity_'+str(stage) + block + '_branch', reuse=tf.AUTO_REUSE):
            conv_name_base = 'res' + str(stage) + block + '_branch'
            bn_name_base = 'bn' + str(stage) + block + '_branch'

            # Retrieve Filters
            f1, f2, f3 = filters


            X_shortcut = X

            # First component of main path
            s1 = X.get_shape().as_list()
            F1 = tf.get_variable(conv_name_base + '2a_f1', [1, 1, s1[3], f1], initializer=self.initializer)
            X = tf.nn.conv2d(X, filter=F1, strides=[1, 1, 1, 1], padding="VALID", name=conv_name_base + '2a')
            # x_mean, x_var = tf.nn.moments(X, axis=3)
            # X = tf.nn.batch_normalization(X, x_mean, x_var, variance_epsilon=1e-3, name=bn_name_base + '2a')
            X = tf.contrib.layers.batch_norm(X, is_training=self.is_training, scope=bn_name_base + '2a')
            X = tf.nn.relu(X)

            # Second component of main path (≈3 lines)
            s2 = X.get_shape().as_list()
            F2 = tf.get_variable(conv_name_base + '2a_f2', [f, f, s2[3], f2], initializer=self.initializer)
            X = tf.nn.conv2d(X, filter=F2, strides=[1, 1, 1, 1], padding="SAME", name=conv_name_base + '2b')
            # x_mean, x_var = tf.nn.moments(X, axis=3)
            # X = tf.nn.batch_normalization(X, x_mean, x_var, variance_epsilon=1e-3, name=bn_name_base + '2b')
            X = tf.contrib.layers.batch_norm(X, is_training=self.is_training, scope=bn_name_base + '2b')
            X = tf.nn.relu(X)

            # Third component of main path (≈2 lines)
            s3 = X.get_shape().as_list()
            F3 = tf.get_variable(conv_name_base + '2a_f3', [1, 1, s3[3], f3], initializer=self.initializer)
            X = tf.nn.conv2d(X, filter=F3, strides=[1, 1, 1, 1], padding="VALID", name=conv_name_base + '2c')
            # x_mean, x_var = tf.nn.moments(X, axis=3)
            # X = tf.nn.batch_normalization(X, x_mean, x_var, variance_epsilon=1e-3, name=bn_name_base + '2c')
            X = tf.contrib.layers.batch_norm(X, is_training=self.is_training, scope=bn_name_base + '2c')
            X = tf.nn.relu(X)

            X = tf.add(X, X_shortcut)
            X = tf.nn.relu(X)
        return X


    'The convolutional block'

    def _convolutional_block(self, X, f, filters, stage, block, s=2):
        """
            Implementation of the convolutional block as defined in Figure 4

            Arguments:
            X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
            f -- integer, specifying the shape of the middle CONV's window for the main path
            filters -- python list of integers, defining the number of filters in the CONV layers of the main path
            stage -- integer, used to name the layers, depending on their position in the network
            block -- string/character, used to name the layers, depending on their position in the network
            s -- Integer, specifying the stride to be used

            Returns:
            X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
            """

        # defining name basis
        with tf.variable_scope('convolutional_'+str(stage) + block + '_branch', reuse=tf.AUTO_REUSE):
            conv_name_base = 'res' + str(stage) + block + '_branch'
            bn_name_base = 'bn' + str(stage) + block + '_branch'
            #print('X.SHAPE', X)

            # Retrieve Filters
            f1, f2, f3 = filters
            X_shortcut = X

            # First component of main path
            s1 = X.get_shape().as_list()
            F1 = tf.get_variable(conv_name_base + '2a_f1', [1, 1, s1[3], f1], initializer=self.initializer)
            X = tf.nn.conv2d(X, filter=F1, strides=[1, s, s, 1], padding="VALID", name=conv_name_base + '2a')
            # x_mean, x_var = tf.nn.moments(X, axis=3)
            # X = tf.nn.batch_normalization(X, x_mean, x_var, variance_epsilon=1e-3, name=bn_name_base + '2a')
            X = tf.contrib.layers.batch_norm(X, is_training=self.is_training,scope=bn_name_base + '2a')
            X = tf.nn.relu(X)

            # Second component of main path (≈3 lines)
            s2 = X.get_shape().as_list()
            F2 = tf.get_variable(conv_name_base + '2a_f2', [f, f, s2[3], f2], initializer=self.initializer)
            X = tf.nn.conv2d(X, filter=F2, strides=[1, 1, 1, 1], padding="SAME", name=conv_name_base + '2b')
            # x_mean, x_var = tf.nn.moments(X, axis=3)
            # X = tf.nn.batch_normalization(X, x_mean, x_var, variance_epsilon=1e-3, name=bn_name_base + '2b')
            X = tf.contrib.layers.batch_norm(X, is_training=self.is_training, scope=bn_name_base + '2b')
            X = tf.nn.relu(X)

            # Third component of main path (≈2 lines)
            s3 = X.get_shape().as_list()
            F3 = tf.get_variable(conv_name_base + '2a_f3', [1, 1, s3[3], f3], initializer=self.initializer)
            X = tf.nn.conv2d(X, filter=F3, strides=[1, 1, 1, 1], padding="VALID", name=conv_name_base + '2c')
            # x_mean, x_var = tf.nn.moments(X, axis=3)
            # X = tf.nn.batch_normalization(X, x_mean, x_var, variance_epsilon=1e-3, name=bn_name_base + '2c')
            X = tf.contrib.layers.batch_norm(X, is_training=self.is_training, scope=bn_name_base + '2c')
            X = tf.nn.relu(X)

            ##### SHORTCUT PATH #### (≈2 lines)
            s4 = X_shortcut.get_shape().as_list()
            F4 = tf.get_variable(conv_name_base + '2a_f4', [1, 1, s4[3], f3], initializer=self.initializer)
            X_shortcut = tf.nn.conv2d(X_shortcut, filter=F4, strides=[1, s, s, 1], padding="VALID", name=conv_name_base + '2c1')
            # x_mean, x_var = tf.nn.moments(X_shortcut, axis=3)
            # X_shortcut = tf.nn.batch_normalization(X_shortcut, x_mean, x_var, variance_epsilon=1e-3, name=bn_name_base + '2c')
            X_shortcut = tf.contrib.layers.batch_norm(X_shortcut, is_training=self.is_training, scope=bn_name_base + '2c2')
            X = tf.add(X, X_shortcut)
            X = tf.nn.relu(X)
        return X

    def ResNet50(self):
        """
        Implementation of the popular ResNet50 the following architecture:
        CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
        -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

        Arguments:
        input_shape -- shape of the images of the dataset
        classes -- integer, number of classes

        Returns:
        model -- a Model() instance in Keras
        """

        # Define the input as a tensor with shape input_shape
        X_input = self.input_x
        # Zero-Padding
        X = tf.keras.layers.ZeroPadding2D((3, 3))(X_input)
        #print(X)
        # Stage 1
        with tf.variable_scope('conv1_f1', reuse=tf.AUTO_REUSE):
            filt = tf.get_variable('conv1_f11', [7, 7, 3, 64], initializer=self.initializer)
            X = tf.nn.conv2d(X, filter=filt, strides=[1, 2, 2, 1], padding="VALID", name='conv1')
            X = tf.contrib.layers.batch_norm(X, is_training=self.is_training,  scope='bn_conv1')
            X = tf.nn.relu(X)
            X = tf.layers.MaxPooling2D((3, 3), strides=(2, 2),data_format='channels_last')(X)

        # Stage 2
        X = self._convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
        X = self._identity_block(X, 3, [64, 64, 256], stage=2, block='b')
        X = self._identity_block(X, 3, [64, 64, 256], stage=2, block='c')

        ### START CODE HERE ###

        # Stage 3 (≈4 lines)
        X = self._convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
        X = self._identity_block(X, 3, [128, 128, 512], stage=3, block='b')
        X = self._identity_block(X, 3, [128, 128, 512], stage=3, block='c')
        X = self._identity_block(X, 3, [128, 128, 512], stage=3, block='d')

        # Stage 4 (≈6 lines)
        X = self._convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
        X = self._identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
        X = self._identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
        X = self._identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
        X = self._identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
        X = self._identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

        # Stage 5 (≈3 lines)
        X = self._convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
        X = self._identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
        X = self._identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

        # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
        X = tf.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid',
                                                          data_format='channels_last', name="avg_pool")(X)
        #print('AveragePooling2D', X)
        ### END CODE HERE ###
        # output layer
        X  = tf.layers.Flatten(name='flatten')(X)
        #print('Flatten', X)
        X = tf.layers.Dense(self.classes, activation='softmax', name='fc' + str(self.classes))(X)
        return X

    def loss(self):
        with tf.name_scope("loss"):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.input_y, logits=self.logits))
        return loss

    def accury(self):

        correct_prediction = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")
        return accuracy

    def pred(self):
        predictions = tf.argmax(self.logits, axis=1, name="predictions")
        return predictions


    def trace_process(self):
        learning_rate = tf.train.exponential_decay(0.001, self.global_step, decay_steps =100,
                                                   decay_rate=0.9, staircase=True)
        # Define Training procedure
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)
        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", self.loss)
        acc_summary = tf.summary.scalar("accuracy", self.accuracy)
        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        return train_op, train_summary_op





X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T

print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))


num_epochs = 2
ckpt_dir = './'
batches = 10
session_conf = tf.ConfigProto()
sess = tf.Session(config=session_conf)
with sess.as_default():
    model = Resnet()
    saver = tf.train.Saver(max_to_keep=num_epochs)
    if not os.path.exists(os.path.join(ckpt_dir, "tf_log")):
        os.makedirs(os.path.join(ckpt_dir, "tf_log"))
    train_summary_dir = os.path.join(ckpt_dir, "tf_log")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
    checkpoint_dir = os.path.abspath(os.path.join(ckpt_dir, "checkpoints"))
    if os.path.exists(checkpoint_dir + "checkpoint"):
        print("Restoring Variables from Checkpoint for rnn model.")
        saver.restore(sess, tf.train.latest_checkpoint(os.path.join(ckpt_dir, "checkpoint")))
    # Initialize
    print('Initializing Variables')
    sess.run(tf.global_variables_initializer())
    counter = 0
    time_str = datetime.datetime.now().isoformat()
    b = X_train.shape[0]

    for j in range(num_epochs):
        for batch in range(int(X_train.shape[0]/batches)):
            s = np.random.randint(0, 1070, 1)[0]
            x_batch, y_batch = X_train[s:s+10], Y_train[s:s+10]
            curr_loss, curr_acc,  _, summaries, step = sess.run(
                [model.loss, model.accuracy,  model.train, model.train_summary_op, model.global_step],
                        feed_dict={model.input_x: x_batch, model.input_y: y_batch})
            train_summary_writer.add_summary(summaries, step)
            save_path = ckpt_dir+'checkpoint/'+ str(j) + "-model.ckpt"
            saver.save(sess, save_path, global_step=step)
            counter = counter + 1

            print(time_str, "Batch_size %d /%d \t-\tTrain Loss:%.3f \t-\tTrain Accuracy:%.3f"
                  %(batch, b, curr_loss, curr_acc))













