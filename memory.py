
import math
import numpy as np

_P_CHOICE_EPSILON = 0.001

class MemoryChunk(object):

    def __init__(self, weights, states, actions, rewards, next_states, is_terminal):
        self.weights = weights
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states
        self.is_terminal = is_terminal


class Memory(object):

    def __init__(self, max_capacity, state_shape, alpha, annealing_beta):
        self._alpha = alpha
        self._beta = annealing_beta
        self._cur_beta = self._beta(0)

        self._max_capacity = max_capacity
        self._insert_index = 0
        self._num_entries = 0

        state_shape = (max_capacity, ) + state_shape

        self._states = np.zeros(state_shape)
        self._actions = np.zeros(max_capacity)
        self._rewards = np.zeros(max_capacity)
        self._next_states = np.zeros(state_shape)
        self._is_terminal = np.zeros(max_capacity, dtype=np.bool)
        self._p_choice = np.full(max_capacity, 1.0)

        self._steps_since_average_updated = 0
        self._average_p = 1.0

        self._psum = 0.0

    def initialize_episode(self, episode_count):
        self._cur_beta = self._beta(episode_count)

    def num_entries(self):
        return self._num_entries

    def capacity(self):
        return self._max_capacity

    def add_memory(self, state, action, reward, next_state):
        if self._insert_index >= self._max_capacity:
            self._insert_index = 0

        # self._steps_since_average_updated += 1
        # if self._steps_since_average_updated > 64:
        #     self._average_p = self._calculate_average_p()
        #     self._steps_since_average_updated = 0

        # self._psum += self._average_p
        # if self._num_entries == self._max_capacity:
        #     self._psum -= self._p_choice[self._insert_index]

        self._p_choice[self._insert_index] = self._average_p
        self._states[self._insert_index] = state
        self._actions[self._insert_index] = action
        self._rewards[self._insert_index] = reward

        if next_state is None:
            self._is_terminal[self._insert_index] = True
        else:
            self._is_terminal[self._insert_index] = False
            self._next_states[self._insert_index] = next_state

        self._insert_index += 1

        if self._num_entries < self._max_capacity:
            self._num_entries += 1

    def sample(self, num_samples):
        indices = np.random.choice(np.arange(self._num_entries),
                                   size=num_samples)
                                #    p=(self._p_choice[:self._num_entries] / self._psum))

        self._prev_sample = indices
        return MemoryChunk(np.ones(num_samples),  # self._weights(self._p_choice[indices], self._psum),
                           self._states[indices],
                           self._actions[indices],
                           self._rewards[indices],
                           self._next_states[indices],
                           self._is_terminal[indices])

    def update_p_choice(self, td_errors):
        pass
        # new_pchoice = np.power(np.abs(td_errors) + _P_CHOICE_EPSILON, self._alpha)
        # self._p_choice[self._prev_sample] = new_pchoice
        # self._psum = np.sum(self._p_choice[:self._num_entries])

    def _calculate_average_p(self):
        return np.mean(self._p_choice[:self._num_entries])

    def _weights(self, pchoice, psum):
        return np.power(1.0 / ((pchoice / psum) * self._num_entries), self._cur_beta)
