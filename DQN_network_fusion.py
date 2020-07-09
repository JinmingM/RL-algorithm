import tensorflow as tf
import numpy as np
import random
from collections import deque

# Hyper Parameters:
FRAME_PER_ACTION = 1
GAMMA = 0.99  # decay rate of past observations
OBSERVE = 2000.  # timesteps to observe before training
EXPLORE = 200000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.001  # 0.001 # final value of epsilon
INITIAL_EPSILON = 0.9  # 0.01 # starting value of epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH_SIZE = 32  # size of minibatch
UPDATE_TIME = 100


class DQN:
    def __init__(self, actions):
        # init replay memory
        self.replayMemory = deque()
        # init some parameters
        self.timeStep = 0
        self.epsilon = INITIAL_EPSILON
        self.actions = actions
        # init Q network
        tf.set_random_seed(1234)
        self.stateInput, self.stateInput1, self.QValue, self.W_conv1, self.b_conv1, self.W_conv2, self.b_conv2, self.W_conv3, self.b_conv3, self.W_fc1, self.b_fc1, self.W_fc3, self.b_fc3, self.W_fc4, self.b_fc4, self.W_fc_fusion, self.b_fc_fusion = self.createQNetwork()

        # init Target Q Network
        self.stateInputT, self.stateInput1T, self.QValueT, self.W_conv1T, self.b_conv1T, self.W_conv2T, self.b_conv2T, self.W_conv3T, self.b_conv3T, self.W_fc1T, self.b_fc1T, self.W_fc3T, self.b_fc3T, self.W_fc4T, self.b_fc4T, self.W_fc_fusionT, self.b_fc_fusionT = self.createQNetwork()

        self.copyTargetQNetworkOperation = [self.W_conv1T.assign(self.W_conv1), self.b_conv1T.assign(self.b_conv1),
                                            self.W_conv2T.assign(self.W_conv2), self.b_conv2T.assign(self.b_conv2),
                                            self.W_conv3T.assign(self.W_conv3), self.b_conv3T.assign(self.b_conv3),
                                            self.W_fc1T.assign(self.W_fc1), self.b_fc1T.assign(self.b_fc1),
                                            self.W_fc4T.assign(self.W_fc4), self.b_fc4T.assign(self.b_fc4),
                                            self.W_fc_fusionT.assign(self.W_fc_fusion), self.b_fc_fusionT.assign(self.b_fc_fusion),
                                            self.W_fc3T.assign(self.W_fc3), self.b_fc3T.assign(self.b_fc3)]

        self.createTrainingMethod()

        # saving and loading networks
        self.saver = tf.train.Saver()
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())
        # checkpoint = tf.train.get_checkpoint_state("saved_networks")
        # if checkpoint and checkpoint.model_checkpoint_path:
        #     self.saver.restore(self.session, checkpoint.model_checkpoint_path)
        #     print("Successfully loaded:", checkpoint.model_checkpoint_path)
        # else:
        #     print("Could not find old network weights")

    def createQNetwork(self):
        # network weights
        W_conv1 = self.weight_variable([8, 8, 4, 32])
        b_conv1 = self.bias_variable([32])

        W_conv2 = self.weight_variable([4, 4, 32, 64])
        b_conv2 = self.bias_variable([64])

        W_conv3 = self.weight_variable([3, 3, 64, 64])
        b_conv3 = self.bias_variable([64])

        W_fc1 = self.weight_variable([1600, 512])
        b_fc1 = self.bias_variable([512])

        # W_fc2 = self.weight_variable([512, self.actions])
        # b_fc2 = self.bias_variable([self.actions])

        # image input layer
        stateInput = tf.placeholder("float", [None, 80, 80, 4])

        # hidden layers
        h_conv1 = tf.nn.relu(self.conv2d(stateInput, W_conv1, 4) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2, 2) + b_conv2)

        h_conv3 = tf.nn.relu(self.conv2d(h_conv2, W_conv3, 1) + b_conv3)

        h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

        # audio input layer
        stateInput1 = tf.placeholder("float", [None, 72])

        W_fc3 = self.weight_variable([72, 256])
        b_fc3 = self.bias_variable([256])

        W_fc4 = self.weight_variable([256, 256])
        b_fc4 = self.bias_variable([256])

        audio_fc1 = tf.nn.relu(tf.matmul(stateInput1, W_fc3) + b_fc3)
        audio_fc2 = tf.nn.relu(tf.matmul(audio_fc1, W_fc4) + b_fc4)

        fusion = tf.concat([h_fc1, audio_fc2], 1)
        W_fc_fusion = self.weight_variable([768, self.actions])
        b_fc_fusion = self.bias_variable([self.actions])
        # Q Value layer
        QValue = tf.matmul(fusion, W_fc_fusion) + b_fc_fusion

        return stateInput, stateInput1, QValue, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc3, b_fc3, W_fc4, b_fc4, W_fc_fusion, b_fc_fusion

    def copyTargetQNetwork(self):
        self.session.run(self.copyTargetQNetworkOperation)

    def createTrainingMethod(self):
        self.actionInput = tf.placeholder("float", [None, self.actions])
        self.yInput = tf.placeholder("float", [None])
        Q_Action = tf.reduce_sum(tf.multiply(self.QValue, self.actionInput), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.yInput - Q_Action))
        self.trainStep = tf.train.AdamOptimizer(1e-6).minimize(self.cost)

    def trainQNetwork(self):

        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replayMemory, BATCH_SIZE)
        image_batch = [data[0] for data in minibatch]
        audio_batch = [data[1] for data in minibatch]
        action_batch = [data[2] for data in minibatch]
        reward_batch = [data[3] for data in minibatch]
        nextimage_batch = [data[4] for data in minibatch]
        nextaudio_batch = [data[5] for data in minibatch]

        # Step 2: calculate y
        y_batch = []
        QValue_batch = self.QValueT.eval(feed_dict={self.stateInputT: nextimage_batch, self.stateInput1T: nextaudio_batch})
        for i in range(0, BATCH_SIZE):
            terminal = minibatch[i][6]
            if terminal:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(QValue_batch[i]))

        self.trainStep.run(feed_dict={
            self.yInput: y_batch,
            self.actionInput: action_batch,
            self.stateInput: image_batch,
            self.stateInput1: audio_batch,
        })

        # save network every 100000 iteration
        if self.timeStep % 100000 == 0:
            self.saver.save(self.session, 'saved_networks/' + 'network' + '-dqn', global_step=self.timeStep)

        if self.timeStep % UPDATE_TIME == 0:
            self.copyTargetQNetwork()

    def setPerception(self, nextObservation, audio, action, reward, terminal):
        # newState = np.append(nextObservation,self.currentState[:,:,1:],axis = 2)
        newState = np.append(self.currentState[:, :, 1:], nextObservation, axis=2)
        newAudio = audio
        self.replayMemory.append((self.currentState, self.currentAudio, action, reward, newState, newAudio, terminal))
        if len(self.replayMemory) > REPLAY_MEMORY:
            self.replayMemory.popleft()
        if self.timeStep > OBSERVE:
            # Train the network
            self.trainQNetwork()

        # print info
        state = ""
        if self.timeStep <= OBSERVE:
            state = "observe"
        elif self.timeStep > OBSERVE and self.timeStep <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        # print("TIMESTEP", self.timeStep, "/ STATE", state, "/ EPSILON", self.epsilon)

        self.currentState = newState
        self.currentAudio = newAudio
        self.timeStep += 1

    def setPerception1(self, nextObservation, audio, action, reward, terminal):
        # newState = np.append(nextObservation,self.currentState[:,:,1:],axis = 2)
        newState = np.append(self.currentState[:, :, 1:], nextObservation, axis=2)
        newAudio = audio

        self.currentState = newState
        self.currentAudio = newAudio
        self.timeStep += 1

    def getAction(self):
        QValue = self.QValue.eval(feed_dict={self.stateInput: [self.currentState], self.stateInput1: [self.currentAudio]})[0]
        action = np.zeros(self.actions)
        action_index = 0
        if self.timeStep % FRAME_PER_ACTION == 0:
            if random.random() <= self.epsilon:
                action_index = random.randrange(self.actions)
                action[action_index] = 1
            else:
                action_index = np.argmax(QValue)
                action[action_index] = 1
        else:
            action[0] = 1  # do nothing

        # change episilon
        if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        return action

    def setInitState(self, observation, audio):
        self.currentState = np.stack((observation, observation, observation, observation), axis=2)
        self.currentAudio = audio

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")