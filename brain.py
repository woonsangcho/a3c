
import tensorflow as tf
import models
import tensorflow.contrib.layers as layers
from helper import normalized_columns_initializer

class Brain(object):
    def __init__(self, **kwargs):
        self.env = kwargs['env']
        self.scope = kwargs['worker_name']

    def get_transition(self, session, inpt,t):
        raise NotImplementedError

    def get_value(self, session, inpt):
        raise NotImplementedError

class NeuralNetwork(Brain):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        with tf.variable_scope(self.scope):
            self.input = tf.placeholder(tf.float32,
                                        [None] + list(self.env.observation_space.shape)
                                        , name='input')

            self.cnn_output = models.CNN(scope='cnn',
                                    convs=kwargs['convs'],
                                    hiddens=kwargs['hiddens'],
                                    inpt=self.input)

            self.mlp_output = models.MLP(scope='mlp',
                                    hiddens=kwargs['hiddens'],
                                    inpt=self.cnn_output)


class A3CFeedForwardNN(NeuralNetwork):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)


        with tf.device(kwargs['device']), tf.variable_scope(self.scope):
            with tf.variable_scope('logits'):
                self.logits = layers.fully_connected(self.mlp_output, num_outputs=self.env.action_space.n,
                                                activation_fn=None,
                                                weights_initializer=normalized_columns_initializer(0.01),
                                                biases_initializer=None)

            with tf.variable_scope('value'):
                self.value = layers.fully_connected(self.mlp_output, num_outputs=1,
                                               activation_fn=None,
                                               weights_initializer=normalized_columns_initializer(1.0),
                                               biases_initializer=None
                                               )

            self.action_logits = tf.identity(self.logits)
            self.sample_action = tf.one_hot(
                                    tf.squeeze(
                                        tf.multinomial(
                                        self.action_logits - tf.reduce_max(self.action_logits, 1, keep_dims=True), 1)),
                                        self.env.action_space.n)

            self.value = tf.reshape(self.value, [-1])

            self.learning_rate = tf.placeholder(shape=[], dtype=tf.float32, name='learning_rate')
            self.target_R = tf.placeholder(shape=[None], dtype=tf.float32, name='target_R')
            self.advantages = tf.placeholder(shape=[None], dtype=tf.float32, name='advantages')
            self.actions = tf.placeholder(tf.float32, [None, self.env.action_space.n], name="actions")

            self.optimizer = tf.train.AdamOptimizer(self.learning_rate, use_locking=False)
            # self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate,
            #                                             decay=0.99,
            #                                             momentum=0,
            #                                             epsilon=0.1,
            #                                             use_locking=False)

            self.policy_loss = tf.reduce_sum(
                self.advantages * tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.action_logits,
                                                                                labels=tf.argmax(self.actions,1)))
            entropy = -tf.reduce_sum(tf.nn.softmax(self.action_logits) * tf.nn.log_softmax(self.action_logits))
            self.policy_loss -= 0.01 * entropy
            self.value_loss = tf.nn.l2_loss(self.value - self.target_R)
            self.loss = self.policy_loss + 0.5 * self.value_loss

            self.local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
            self.gradients = tf.gradients(self.loss, self.local_vars,
                                          gate_gradients=False,
                                          aggregation_method=None,
                                          colocate_gradients_with_ops=False)

            self.var_norms = tf.global_norm(self.local_vars)
            grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)
            grads_and_vars = list(zip(grads, self.local_vars))
            self.apply_grads = self.optimizer.apply_gradients(grads_and_vars)

    def get_transition(self, session, inpt, t):
        fetched = session.run([self.sample_action, self.value], {self.input: [inpt]})
        return fetched

    def get_value(self, session, inpt):
        fetched = session.run(self.value, {self.input: [inpt]})[0]
        return fetched
