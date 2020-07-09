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
    def __init__(self, actions, modal):
        # init replay memory
        self.replayMemory = deque()
        self.modal = modal
        # init some parameters
        self.timeStep = 0
        self.epsilon = INITIAL_EPSILON
        self.actions = actions
        # init Q network
        self.stateInput, self.QValue, self.W_fc3, self.b_fc3, self.W_fc4, self.b_fc4 ,self.W_fc, self.b_fc = self.createQNetwork()

        # init Target Q Network
        self.stateInputT, self.QValueT, self.W_fc3T, self.b_fc3T, self.W_fc4T, self.b_fc4T, self.W_fcT, self.b_fcT = self.createQNetwork()

        self.copyTargetQNetworkOperation = [self.W_fcT.assign(self.W_fc), self.b_fcT.assign(self.b_fc),
                                            self.W_fc4T.assign(self.W_fc4), self.b_fc4T.assign(self.b_fc4),
                                            self.W_fc3T.assign(self.W_fc3), self.b_fc3T.assign(self.b_fc3)]

        self.createTrainingMethod()

        # saving and loading networks
        self.saver = tf.train.Saver()
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())
        if modal == 'evaluate':
            checkpoint = tf.train.get_checkpoint_state("saved_networks_audio")
            if checkpoint and checkpoint.model_checkpoint_path:
                self.saver.restore(self.session, checkpoint.model_checkpoint_path)
                print("Successfully loaded:", checkpoint.model_checkpoint_path)
            else:
                print("Could not find old network weights")

    def createQNetwork(self):
        stateInput = tf.placeholder("float", [None, 72])

        W_fc3 = self.weight_variable([72, 512])
        b_fc3 = self.bias_variable([512])

        W_fc4 = self.weight_variable([512, 256])
        b_fc4 = self.bias_variable([256])

        audio_fc1 = tf.nn.relu(tf.matmul(stateInput, W_fc3) + b_fc3)
        audio_fc2 = tf.nn.relu(tf.matmul(audio_fc1, W_fc4) + b_fc4)

        # Q Value layer
        W_fc = self.weight_variable([256, self.actions])
        b_fc = self.bias_variable([self.actions])
        QValue = tf.matmul(audio_fc2, W_fc) + b_fc

        return stateInput, QValue, W_fc3, b_fc3, W_fc4, b_fc4, W_fc, b_fc

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
        audio_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        nextaudio_batch = [data[3] for data in minibatch]

        # Step 2: calculate y
        y_batch = []
        QValue_batch = self.QValueT.eval(feed_dict={self.stateInputT: nextaudio_batch})
        for i in range(0, BATCH_SIZE):
            terminal = minibatch[i][4]
            if terminal:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(QValue_batch[i]))

        self.trainStep.run(feed_dict={
            self.yInput: y_batch,
            self.actionInput: action_batch,
            self.stateInput: audio_batch,
        })

        # save network every 100000 iteration
        if self.timeStep % 10000 == 0:
            self.saver.save(self.session, 'saved_networks/' + 'network' + '-dqn', global_step=self.timeStep)

        if self.timeStep % UPDATE_TIME == 0:
            self.copyTargetQNetwork()

    def setPerception(self, audio, action, reward, terminal):
        # newState = np.append(nextObservation,self.currentState[:,:,1:],axis = 2)
        newAudio = audio
        self.replayMemory.append((self.currentAudio, action, reward, newAudio, terminal))
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

        self.currentAudio = newAudio
        self.timeStep += 1

    def getAction(self):
        QValue = self.QValue.eval(feed_dict={self.stateInput: [self.currentAudio]})[0]
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

    def setInitState(self, audio):
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