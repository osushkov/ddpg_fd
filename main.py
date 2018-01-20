
import agent
import gym
import sys
import time

import run_loop
import decaying_value
import ddpg_agent
import memory


TRAIN_EPISODES = 50000
MAX_STEPS_PER_EPISODE = 1000


class RewardTracker(object):

    def __init__(self):
        self._episode_reward = 0.0

    def __call__(self, env, agent, episode, cur_iter, obs, action, reward):
        self._episode_reward += reward

        if cur_iter == 0:
            print("agent average reward {}: {}".format(episode, self._episode_reward))
            self._episode_reward = 0.0

def render_observer(env, agent, episode, cur_iter, obs, action, reward):
    env.render()
    time.sleep(0.025)

def _build_observers():
    observers = []
    # observers.append(lambda env, agent, episode, cur_iter, obs, action, reward: )
    observers.append(RewardTracker())
    return observers


env = gym.make('Pendulum-v0')
# env = gym.make('CartPole-v0')

exploration_rate = decaying_value.DecayingValue(0.5, 0.05, TRAIN_EPISODES)
beta = decaying_value.DecayingValue(0.4, 1.0, TRAIN_EPISODES)
memory = memory.Memory(50000, env.observation_space.high.shape, 0.6, beta)
agent = ddpg_agent.DDPGAgent(env.action_space, env.observation_space, exploration_rate, memory)
observers = _build_observers()

run_loop.run_loop(env, agent, TRAIN_EPISODES, MAX_STEPS_PER_EPISODE, observers)
wait = raw_input("Finished Training")

agent.set_learning(False)
run_loop.run_loop(env, agent, 2, None, [render_observer])
