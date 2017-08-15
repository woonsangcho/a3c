import tensorflow as tf
import tensorflow.contrib.layers as layers

def CNN(scope, convs, hiddens, inpt, reuse=False):
    out = inpt

    with tf.variable_scope(scope, reuse=reuse):
        for num_outputs, kernel_size, stride in convs:
            out = layers.convolution2d(out,
                                       num_outputs=num_outputs,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       activation_fn=tf.nn.relu)
    out = layers.flatten(out)
    return out

def MLP(scope, hiddens, inpt, reuse=False):
    out = inpt

    with tf.variable_scope(scope,reuse=reuse):
        for hidden in hiddens:
            out = layers.fully_connected(out, num_outputs=hidden, activation_fn=tf.nn.relu)

    return out

