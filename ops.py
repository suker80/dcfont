import tensorflow.contrib as contrib
import tensorflow as tf
import numpy as np



def batch_norm(input):
    return tf.contrib.layers.batch_norm(inputs=input,epsilon=1e-5,decay=0.9,updates_collections=None,scale=True,data_format='NCHW')

def conv2d(input,num_output,kernel=[4,4],stride=[2,2],padding='SAME'):
    return tf.layers.conv2d(inputs=input,
                            filters=num_output,
                            kernel_size=kernel,
                            strides=stride,
                            padding=padding)
def leaky_relu(x,leak=0.2):
    return tf.nn.leaky_relu(x)
def deconv2d(input,num_output,kernel=[4,4],stride=[2,2],padding='SAME'):
    return tf.layers.conv2d_transpose(inputs=input,
                                      filters=num_output,
                                      kernel_size=kernel,
                                      strides=stride,
                                      padding=padding
                                      )
def relu(x):
    return tf.nn.relu(x)

def LBC(input,output_size):
    return leaky_relu(batch_norm(conv2d(input,output_size)))

def RBC(input,num_output,kernel,stride,padding='SAME'):
    return relu(batch_norm(conv2d(input,num_output=num_output,kernel=kernel,stride=stride,padding=padding)))

def RBD(input,num_output,kernel,stride,padding='SAME'):
    return relu(batch_norm(deconv2d(input,num_output=num_output,kernel=kernel,stride=stride,padding=padding)))

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
        initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias
