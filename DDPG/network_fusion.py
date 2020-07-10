'''
## Network ##
# Defines the DDPG Value (critic) and Policy (Actor) networks - with and without batch norm
@author: Mark Sinton (msinto93@gmail.com)
'''

import tensorflow as tf
import numpy as np
from utils.ops import dense, batchnorm, relu, tanh
tf.set_random_seed(1234)
img_size = 80
class Critic:
    def __init__(self, state, action, state_dims, action_dims, args, scope='critic'):
        # state - State input to pass through the network
        # action - Action input for which the Q value should be predicted

        self.state = state
        self.action = action
        self.state_dims = np.prod(state_dims)       #Used to calculate the fan_in of the state layer (e.g. if state_dims is (3,2) fan_in should equal 6)
        self.action_dims = np.prod(action_dims)
        self.args = args
        self.scope = scope

        # Networks params
        dense1_size = self.args.dense1_size
        dense2_size = self.args.dense2_size
        final_layer_init = self.args.final_layer_init

        with tf.variable_scope(self.scope):
            self.dense1_mul = dense(self.state, dense1_size, weight_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(self.state_dims))), 1/tf.sqrt(tf.to_float(self.state_dims))),
                                bias_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(self.state_dims))), 1/tf.sqrt(tf.to_float(self.state_dims))), scope='dense1')

            self.dense1 = relu(self.dense1_mul, scope='dense1')

            #Merge first dense layer with action input to get second dense layer
            self.dense2a = dense(self.dense1, dense2_size, weight_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))), 1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))),
                                bias_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))), 1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))), scope='dense2a')

            self.dense2b = dense(self.action, dense2_size, weight_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))), 1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))),
                                bias_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))), 1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))), scope='dense2b')

            self.dense2 = relu(self.dense2a + self.dense2b, scope='dense2')

            self.output = dense(self.dense2, 1, weight_init=tf.random_uniform_initializer(-1*final_layer_init, final_layer_init),
                                bias_init=tf.random_uniform_initializer(-1*final_layer_init, final_layer_init), scope='output')


            self.network_params = tf.trainable_variables(scope=self.scope)

            self.action_grads = tf.gradients(self.output, self.action) # gradient of value output wrt action input - used to train actor network


    def train_step(self, target_Q):
        # target_Q - Target Q value (immediate reward plus expected Q from next state)

        with tf.variable_scope(self.scope):
            with tf.variable_scope('train'):
                learning_rate = self.args.critic_learning_rate
                l2_lambda = self.args.critic_l2_lambda

                self.optimizer = tf.train.AdamOptimizer(learning_rate)
                self.loss = tf.losses.mean_squared_error(target_Q, self.output)
                self.l2_reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.network_params if 'kernel' in v.name]) * l2_lambda
                self.total_loss = self.loss + self.l2_reg_loss

                train_step = self.optimizer.minimize(self.total_loss, var_list=self.network_params)

                return train_step

    def summary_writer(self,):
        tf.summary.scalar("Q value ", self.output)
        tf.summary.scalar("Critic Loss", self.loss)
        tf.summary.scalar("Actor Loss", self.l2_reg_loss)


class Actor:
    def __init__(self, state, state_dims, action_dims, action_bound_low, action_bound_high, args, scope='actor'):
        # state - State input to pass through the network
        # action_bounds - Network will output in range [-1,1]. Multiply this by action_bound to get output within desired boundaries of action space

        self.state = state
        self.state_dims = np.prod(state_dims)       #Used to calculate the fan_in of the state layer (e.g. if state_dims is (3,2) fan_in should equal 6)
        self.action_dims = np.prod(action_dims)
        self.action_bound_low = action_bound_low
        self.action_bound_high = action_bound_high
        self.args = args
        self.scope = scope

        # Networks params
        dense1_size = self.args.dense1_size
        dense2_size = self.args.dense2_size
        final_layer_init = self.args.final_layer_init

        with tf.variable_scope(self.scope):

            self.dense1_mul = dense(self.state, dense1_size, weight_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(self.state_dims))), 1/tf.sqrt(tf.to_float(self.state_dims))),
                                bias_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(self.state_dims))), 1/tf.sqrt(tf.to_float(self.state_dims))), scope='dense1')

            self.dense1 = relu(self.dense1_mul, scope='dense1')

            self.dense2_mul = dense(self.dense1, dense2_size, weight_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(dense1_size))), 1/tf.sqrt(tf.to_float(dense1_size))),
                                bias_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(dense1_size))), 1/tf.sqrt(tf.to_float(dense1_size))), scope='dense2')

            self.dense2 = relu(self.dense2_mul, scope='dense2')

            self.output_mul = dense(self.dense2, self.action_dims, weight_init=tf.random_uniform_initializer(-1*final_layer_init, final_layer_init),
                                bias_init=tf.random_uniform_initializer(-1*final_layer_init, final_layer_init), scope='output')

            self.output_tanh = tanh(self.output_mul, scope='output')

            # Scale tanh output to lower and upper action bounds
            self.output = tf.multiply(0.5, tf.multiply(self.output_tanh, (self.action_bound_high-self.action_bound_low)) + (self.action_bound_high+self.action_bound_low))


            self.network_params = tf.trainable_variables(scope=self.scope)


    def train_step(self, action_grads):
        # action_grads - gradient of value output wrt action from critic network

        with tf.variable_scope(self.scope):
            with tf.variable_scope('train'):
                learning_rate = self.args.actor_learning_rate
                batch_size = self.args.batch_size

                self.optimizer = tf.train.AdamOptimizer(learning_rate)
                self.grads = tf.gradients(self.output, self.network_params, -action_grads)
                self.grads_scaled = list(map(lambda x: tf.divide(x, batch_size), self.grads)) # tf.gradients sums over the batch dimension here, must therefore divide by batch_size to get mean gradients

                train_step = self.optimizer.apply_gradients(zip(self.grads_scaled, self.network_params))

                return train_step


class Critic_BN:
    def __init__(self, state, audio, action, state_dims, action_dims, args, is_training=False, scope='critic'):
        # state - State input to pass through the network
        # action - Action input for which the Q value should be predicted

        self.state = state
        self.audio = audio
        self.action = action
        self.state_dims = np.prod(state_dims)       #Used to calculate the fan_in of the state layer (e.g. if state_dims is (3,2) fan_in should equal 6)
        self.action_dims = np.prod(action_dims)
        self.args = args
        self.is_training = is_training
        self.scope = scope

        # Networks params
        dense1_size = self.args.dense1_size
        dense2_size = self.args.dense2_size
        final_layer_init = self.args.final_layer_init

        with tf.variable_scope(self.scope):
            self.fc1_a = tf.layers.dense(self.audio, 50, tf.nn.relu)
            self.fc2_a = tf.layers.dense(self.fc1_a, 50, tf.nn.relu)
            self.final_flat = tf.concat([createNetwork_cnn(self.state), self.fc2_a], 1)
            self.input_norm = batchnorm(self.final_flat, self.is_training, scope='input_norm')

            self.dense1_mul = dense(self.input_norm, dense1_size, weight_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(self.state_dims))), 1/tf.sqrt(tf.to_float(self.state_dims))),
                                bias_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(self.state_dims))), 1/tf.sqrt(tf.to_float(self.state_dims))), scope='dense1')

            self.dense1_bn = batchnorm(self.dense1_mul, self.is_training, scope='dense1')

            self.dense1 = relu(self.dense1_bn, scope='dense1')

            #Merge first dense layer with action input to get second dense layer
            self.dense2a = dense(self.dense1, dense2_size, weight_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))), 1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))),
                                bias_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))), 1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))), scope='dense2a')

            self.dense2b = dense(self.action, dense2_size, weight_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))), 1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))),
                                bias_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))), 1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))), scope='dense2b')

            self.dense2 = relu(self.dense2a + self.dense2b, scope='dense2')

            self.output = dense(self.dense2, 1, weight_init=tf.random_uniform_initializer(-1*final_layer_init, final_layer_init),
                                bias_init=tf.random_uniform_initializer(-1*final_layer_init, final_layer_init), scope='output')


            self.network_params = tf.trainable_variables(scope=self.scope)

            self.action_grads = tf.gradients(self.output, self.action) # Gradient of value output wrt action input - used to train actor network

    def train_step(self, target_Q):
        # target_Q - Target Q value (immediate reward plus expected Q from next state)

        with tf.variable_scope(self.scope):
            with tf.variable_scope('train'):
                learning_rate = self.args.critic_learning_rate
                l2_lambda = self.args.critic_l2_lambda

                self.optimizer = tf.train.AdamOptimizer(learning_rate)
                self.loss = tf.losses.mean_squared_error(target_Q, self.output)
                self.l2_reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.network_params if 'kernel' in v.name]) * l2_lambda
                self.total_loss = self.loss + self.l2_reg_loss

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, self.scope) # Ensure batch norm moving means and variances are updated every training step
                with tf.control_dependencies(update_ops):
                    train_step = self.optimizer.minimize(self.total_loss, var_list=self.network_params)

                return train_step


class Actor_BN:
    def __init__(self, state, audio, state_dims, action_dims, action_bound_low, action_bound_high, args, is_training=False, scope='actor'):
        # state - State input to pass through the network
        # action_bounds - Network will output in range [-1,1]. Multiply this by action_bound to get output within desired boundaries of action space

        self.state = state
        self.audio = audio
        self.state_dims = np.prod(state_dims)       #Used to calculate the fan_in of the state layer (e.g. if state_dims is (3,2) fan_in should equal 6)
        self.action_dims = np.prod(action_dims)
        self.action_bound_low = action_bound_low
        self.action_bound_high = action_bound_high
        self.args = args
        self.is_training = is_training
        self.scope = scope

        # Networks params
        dense1_size = self.args.dense1_size
        dense2_size = self.args.dense2_size
        final_layer_init = self.args.final_layer_init

        with tf.variable_scope(self.scope):
            self.fc1_a = tf.layers.dense(self.audio, 50, tf.nn.relu)
            self.fc2_a = tf.layers.dense(self.fc1_a, 50, tf.nn.relu)
            self.final_flat = tf.concat([createNetwork_cnn(self.state), self.fc2_a], 1)
            # self.state_dims = np.prod(self.final_flat)
            # print(np.shape(self.final_flat), self.state_dims)

            self.input_norm = batchnorm(self.final_flat, self.is_training, scope='input_norm')

            self.dense1_mul = dense(self.input_norm, dense1_size, weight_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(self.state_dims))), 1/tf.sqrt(tf.to_float(self.state_dims))),
                                bias_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(self.state_dims))), 1/tf.sqrt(tf.to_float(self.state_dims))), scope='dense1')

            self.dense1_bn = batchnorm(self.dense1_mul, self.is_training, scope='dense1')

            self.dense1 = relu(self.dense1_bn, scope='dense1')

            self.dense2_mul = dense(self.dense1, dense2_size, weight_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(dense1_size))), 1/tf.sqrt(tf.to_float(dense1_size))),
                                bias_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(dense1_size))), 1/tf.sqrt(tf.to_float(dense1_size))), scope='dense2')

            self.dense2_bn = batchnorm(self.dense2_mul, self.is_training, scope='dense2')

            self.dense2 = relu(self.dense2_bn, scope='dense2')

            self.output_mul = dense(self.dense2, self.action_dims, weight_init=tf.random_uniform_initializer(-1*final_layer_init, final_layer_init),
                                bias_init=tf.random_uniform_initializer(-1*final_layer_init, final_layer_init), scope='output')

            self.output_tanh = tanh(self.output_mul, scope='output')

            # Scale tanh output to lower and upper action bounds
            self.output = tf.multiply(0.5, tf.multiply(self.output_tanh, (self.action_bound_high-self.action_bound_low)) + (self.action_bound_high+self.action_bound_low))


            self.network_params = tf.trainable_variables(scope=self.scope)

    def train_step(self, action_grads):
        # action_grads - gradient of value output wrt action from critic network

        with tf.variable_scope(self.scope):
            with tf.variable_scope('train'):
                learning_rate = self.args.actor_learning_rate
                batch_size = self.args.batch_size

                self.optimizer = tf.train.AdamOptimizer(learning_rate)
                self.grads = tf.gradients(self.output, self.network_params, -action_grads)
                self.grads_scaled = list(map(lambda x: tf.divide(x, batch_size), self.grads)) # tf.gradients sums over the batch dimension here, must therefore divide by batch_size to get mean gradients

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, self.scope) # Ensure batch norm moving means and variances are updated every training step
                with tf.control_dependencies(update_ops):
                    train_step = self.optimizer.apply_gradients(zip(self.grads_scaled, self.network_params))

                return train_step

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def _activation(z, a=0.01):
    return (tf.maximum(a*z, z))

def createNetwork_cnn(image):
    # network weights
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])

    # hidden layers
    h_conv1 = tf.nn.relu(tf.layers.batch_normalization(conv2d(image, W_conv1, 4) + b_conv1))
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_pool1, W_conv2, 2) + b_conv2))

    h_conv3 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_conv2, W_conv3, 1) + b_conv3))

    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    h_fc1 = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(h_conv3_flat, W_fc1) + b_fc1))

    # readout layer
    # readout = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    return h_fc1