# -*- coding: utf-8 -*-
# """
# Created on Sat Jun  5 21:47:39 2021

# @author: VIVEK OOMMEN
# """

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Reshape, Conv2D, PReLU, Flatten, Dense, Activation
from tensorflow.keras.losses import MeanSquaredError


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

import os
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
#os.environ["CUDA_VISIBLE_DEVICES"]='0'


class PINN_Model(tf.keras.Model):

    def __init__(self, Par):
        super(PINN_Model, self).__init__()
        np.random.seed(23)
        tf.random.set_seed(23)

        #Defining some model parameters
        self.output_dim = 1

        self.Par = Par

        self.index_list = []
        self.train_loss_list = []
        self.val_loss_list = []

        self.lr=10**-4
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

        self.NN = Sequential()
        self.NN.add(Dense(50, activation='tanh'))
        self.NN.add(Dense(50, activation='tanh'))
        self.NN.add(Dense(self.output_dim))


    @tf.function
    def call(self, x):
    #x -> [BS,1]
        u = self.NN(x)
        return(u)

    @tf.function
    def PDE_loss(self, x):
        #PDE_loss
        with tf.GradientTape(persistent=True) as tape4:
            tape4.watch(x)
            with tf.GradientTape(persistent=True) as tape3:
                tape3.watch(x)
                with tf.GradientTape(persistent=True) as tape2:
                    tape2.watch(x)
                    with tf.GradientTape(persistent=True) as tape1:
                        tape1.watch(x)
                        u = self.call(x)
                    u_x = tape1.gradient(u,x)
                    del tape1
                u_xx = tape2.gradient(u_x, x)
                del tape2
            u_xxx = tape3.gradient(u_xx,x)
            del tape3
        u_xxxx = tape4.gradient(u_xxx, x)

        eq1 = u_xxxx + self.Par['Re']*(u_x*u_xx - u*u_xxx)
        pde_loss = tf.reduce_mean(tf.square(eq1))

        return [u_x, u_xx, u_xxx, u_xxxx, pde_loss]


    @tf.function
    def Data_loss(self):
        tensor = lambda x: tf.convert_to_tensor(x, dtype=tf.float32)

        #BC1
        BC1 = tf.square(self.PDE_loss( tensor( [[1]] ) )[0] )

        #BC2
        BC2 = tf.square(self.call(tensor( [[1]] )) - tensor( [[1]] ) )

        #BC3
        BC3 = tf.square(self.PDE_loss( tensor( [[-1]] ) )[0] )

        #BC4
        BC4 = tf.square(self.call(tensor( [[-1]] )) - tensor( [[-1]] ) )

        #BC5
        BC5 = tf.square(self.PDE_loss( tensor( [[0]] ) )[1] )

        #BC6
        BC6 = tf.square(self.call(tensor( [[0]] ))  )

        #-------------------------------------------------------------#
        #Total Loss
        data_loss = tf.reduce_mean( BC1 + BC2 + BC3 + BC4 + 0*BC5 + 0*BC6 )
        #-------------------------------------------------------------#

        return(data_loss)


    @tf.function
    def train_step(self, x_f):

        with tf.GradientTape() as tape:
            pde_loss  = self.PDE_loss(x_f)[-1]
            data_loss = self.Data_loss()

            total_loss = 1*pde_loss + 1*data_loss
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return [pde_loss, data_loss, total_loss]
