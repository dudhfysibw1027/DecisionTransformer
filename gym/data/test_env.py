
import numpy as np

import collections
import pickle

import minari

datasets = []
import gymnasium as gym
env = gym.make("AdroitHandRelocate-v1", max_episode_steps=400)
# dataset = minari.load_dataset(test_env)
# env = dataset.recover_environment()
print(len(env.reset()))
# action = env.action_space.sample()
# output = env.step(action)
# print(np.array(output).shape)
# state, reward, done, hi, bb = output
# print("state", state)
# print("reward", reward)
# print("done", done)
# print("hi", hi)
# print('bb', bb)
import gym
env = gym.make('Hopper-v1')
print(len(env.reset()))
action = env.action_space.sample()
output = env.step(action)
# print(np.array(output).shape)
state, reward, done, hi = output
print("state", state)
print("reward", reward)
print("done", done)
print("hi", hi)

#
# for env_name in ["door-cloned-v2", "door-expert-v2", "door-human-v2"]:
#
#     dataset = minari.load_dataset(env_name)
#     env = dataset.recover_environment()
#     # env = gym.make(name)
#     data_ = collections.defaultdict(list)
#     episode_num = 0
#     paths = []
#     for episode in dataset.iterate_episodes():
#         # for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
#         #     data_[k].append(dataset[k][i])
#         data_['observations'] = episode.observations[0:len(episode.observations)-1]     # [0; 200]
#         data_['next_observations'] = episode.observations[1:len(episode.observations)]  # [1: 201]
#         data_['actions'] = episode.actions
#         data_['rewards'] = episode.rewards
#         data_['terminals'] = episode.terminations
#         data_['timeouts'] = episode.truncations
#         episode_data = {}
#         for k in data_:
#             episode_data[k] = np.array(data_[k])
#         data_ = collections.defaultdict(list)
#         paths.append(episode_data)
#         # print(episode_data.keys())
#         # print(episode_data)
#         # for i in range(0, len(episode.observations)):
#         #     pass
#         #     print(i)
#         #     # print(f"Observation {i}: {episode.observations[i]}")
#         #     print(f"Action {i}: {episode.actions[i].shape}")
#         #     print(f"Done {i}: {episode.terminations[i].shape}")
#
#     with open(f'{env_name}.pkl', 'wb') as f:
#         # pickle.dump(paths, f)
#         pass
