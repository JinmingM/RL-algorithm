'''
## Replay Memory ##
# Adapted from: https://github.com/tambetm/simple_dqn/blob/master/src/replay_memory.py
# Creates replay memory buffer to add experiences to and sample batches of experiences from
'''

import numpy as np
from collections import deque
import random

class ReplayMemory:
    def __init__(self, args, state_dims, action_dims):
        self.buffer_size = args.replay_mem_size
        self.min_buffer_size = args.initial_replay_mem_size
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.batch_size = args.batch_size
        self.count = 0
        self.current = 0
        self.buffer = deque()

    def add(self, image, audio, action, reward, image_, audio_, done):
        experience = (image, audio, action, reward, image_, audio_, done)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def getMinibatch(self):
        # memory should be initially populated with random actions up to 'min_buffer_size'
        assert self.count >= self.min_buffer_size, "Replay memory does not contain enough samples to start learning, take random actions to populate replay memory"
        minibatch = random.sample(self.buffer, self.batch_size)
        image_batch = np.asarray([data[0] for data in minibatch])
        audio_batch = np.asarray([data[1] for data in minibatch])
        action_batch = np.asarray([data[2] for data in minibatch])
        reward_batch = np.asarray([data[3] for data in minibatch])
        next_image_batch = np.asarray([data[4] for data in minibatch])
        next_audio_batch = np.asarray([data[5] for data in minibatch])
        done_batch = np.asarray([data[6] for data in minibatch])
        return (image_batch, audio_batch, action_batch, reward_batch, next_image_batch, next_audio_batch, done_batch)

class ReplayMemory1:
    def __init__(self, args, state_dims, action_dims):
        self.buffer_size = args.replay_mem_size
        self.min_buffer_size = args.initial_replay_mem_size
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.batch_size = args.batch_size
        self.count = 0
        self.current = 0
        self.buffer = deque()

    def add(self, image, action, reward, image_, done):
        experience = (image, action, reward, image_, done)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def getMinibatch(self):
        # memory should be initially populated with random actions up to 'min_buffer_size'
        assert self.count >= self.min_buffer_size, "Replay memory does not contain enough samples to start learning, take random actions to populate replay memory"
        minibatch = random.sample(self.buffer, self.batch_size)
        image_batch = np.asarray([data[0] for data in minibatch])
        action_batch = np.asarray([data[1] for data in minibatch])
        reward_batch = np.asarray([data[2] for data in minibatch])
        next_image_batch = np.asarray([data[3] for data in minibatch])
        done_batch = np.asarray([data[4] for data in minibatch])
        return (image_batch, action_batch, reward_batch, next_image_batch, done_batch)


class ReplayMemory2:
    def __init__(self, args, state_dims, action_dims):
        self.buffer_size = args.replay_mem_size
        self.min_buffer_size = args.initial_replay_mem_size
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.batch_size = args.batch_size
        self.count = 0
        self.current = 0

        # preallocate memory
        self.actions = np.empty((self.buffer_size,) + self.action_dims, dtype = np.float32)
        self.rewards = np.empty(self.buffer_size, dtype = np.float32)
        self.states = np.empty((self.buffer_size,) + self.state_dims, dtype = np.float32)
        self.terminals = np.empty(self.buffer_size, dtype = np.bool)   
        
        self.state_batch = np.empty((self.batch_size,) + self.state_dims, dtype = np.float32)
        self.next_state_batch = np.empty((self.batch_size,) + self.state_dims, dtype = np.float32)
        
        
    def add(self, action, reward, state, terminal):        
        assert state.shape == self.state_dims
        assert action.shape == self.action_dims

        self.actions[self.current, ...] = action
        self.rewards[self.current] = reward
        self.states[self.current, ...] = state
        self.terminals[self.current] = terminal
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.buffer_size
        
  
    def getState(self, index):
        # Returns the state at position 'index'.
        return self.states[index, ...]
         

    def getMinibatch(self):
        # memory should be initially populated with random actions up to 'min_buffer_size'
        assert self.count >= self.min_buffer_size, "Replay memory does not contain enough samples to start learning, take random actions to populate replay memory"
                
        # sample random indexes
        indexes = []
        # do until we have a full batch of states
        while len(indexes) < self.batch_size:
            # find random index 
            while True:
                # sample one index
                index = np.random.randint(1, self.count)
                # check index is ok
                # if state and next state wrap over current pointer, then get new one (as state from current pointer position will not be from same episode as state from previous position)
                if index == self.current:
                    continue
                # if state and next state wrap over episode end, i.e. current state is terminal, then get new one (note that next state can be terminal)
                if self.terminals[index-1]:
                    continue
                # index is ok to use
                break
            
            # Populate states and next_states with selected state and next_state
            # NB! having index first is fastest in C-order matrices
            self.state_batch[len(indexes), ...] = self.getState(index - 1)
            self.next_state_batch[len(indexes), ...] = self.getState(index)
            indexes.append(index)   
        
        actions = self.actions[indexes]
        rewards = self.rewards[indexes]
        terminals = self.terminals[indexes]
        
        return self.state_batch, actions, rewards, self.next_state_batch, terminals

