# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper functions for PWIL training script."""

import os
import pickle

from acme import wrappers
import dm_env
import gym

from gym.wrappers.time_limit import TimeLimit
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from core.customized_env import CustomizedEnv

from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn import svm, preprocessing


def load_demonstrations(demo_dir, env_name, state_demo=False):
    """Load expert demonstrations.

    Outputs come with the following format:
      [
        [{observation: o_1, action: a_1}, ...], # episode 1
        [{observation: o'_1, action: a'_1}, ...], # episode 2
        ...
      ]

    Args:
      demo_dir: directory path of expert demonstrations
      env_name: name of the environment

    Returns:
      demonstrations: list of expert demonstrations
    """
    if state_demo:
        demonstrations_filename = os.path.join(demo_dir, '{}_state.pkl'.format(env_name))
    else:
        demonstrations_filename = os.path.join(demo_dir, '{}.pkl'.format(env_name))
    print(demo_dir, demonstrations_filename)
    try:
        demonstrations_file = tf.io.gfile.GFile(demonstrations_filename, 'rb')
        demonstrations = pickle.load(demonstrations_file)
        return demonstrations
    except:
        f = open(demonstrations_filename, 'rb')
        demonstrations = pickle.load(f)
        f.close()
        # print(demonstrations[0])
        new_demonstrations = []
        for t in demonstrations:
            new_t = []
            for pair in t:
                s = pair['observation'][0]
                a = pair['action'][0]
                new_pair = {'observation': s, 'action': a}
                new_t.append(new_pair)
            new_demonstrations.append(new_t)
        return demonstrations[:4]


def load_environment(env_name, max_episode_steps=1000):
    """Outputs a wrapped gym environment."""
    environment = gym.make(env_name)
    environment = TimeLimit(environment, max_episode_steps=max_episode_steps)
    environment = wrappers.gym_wrapper.GymWrapper(environment)
    environment = wrappers.SinglePrecisionWrapper(environment)
    return environment


def extract_env_info_from_demo(expert_replay_buffer,
                               action_space=None,
                               observation_space=None,
                               done_criteria=False,
                               _max_episode_steps=1000,
                               use_log_transform=False,
                               use_scaler=False):
    #expert_replay_buffer = np.array(expert_replay_buffer.buffer())
    states = []
    actions = []
    init_state = []

    for traj in expert_replay_buffer:
        tmp = []
        # print(len(traj))
        for idx, trans in enumerate(traj):
            states.append(trans['observation'])
            act = trans['action']
            if use_log_transform:
                a = act
                signs = np.where((a > 0).astype(np.int) == 1, 1, a)
                signs = np.where((a < 0).astype(np.int) == 1, -1, signs)
                act = np.log2(1 + 1 * abs(act)) * signs

            actions.append(act)
            tmp.append(trans['action'])

            if idx == 0:
                # print()
                # print(act)
                # print()
                init_state.append(trans['observation'])
        #print(np.argmax(np.array(tmp), axis=0))
        # print(np.array(tmp).max(axis=0))

    #delta_states = next_states - states
    # print(states)

    if use_scaler:
        s_scaler = preprocessing.StandardScaler()
        s_scaler.fit(states)
        a_scaler = preprocessing.StandardScaler()
        a_scaler.fit(actions)

        states = s_scaler.transform(np.array(states))
        actions = a_scaler.transform(np.array(actions))
        init_state = s_scaler.transform(np.array(init_state))
    else:
        states = np.array(states)
        actions = np.array(actions)
        init_state = np.array(init_state)
        s_scaler = None
        a_scaler = None

    expert_replay_buffer = [states, actions]

    max_s = np.array(np.abs(states).max(axis=0))
    max_a = np.array(np.abs(actions).max(axis=0))
    min_s = -1 * max_s
    min_a = -1 * max_a  # np.vstack((-max_a, min_a)).max(axis=0)
    #print(max_s, min_s)
    # print(actions.min(axis=0))
    # print(min_a)
    # time.sle()

    if action_space is not None:
        action_space = action_space
    else:
        action_space = spaces.Box(low=min_a, high=max_a,
                                  shape=max_a.shape, dtype=np.float)

    if observation_space is not None:
        observation_space = observation_space
    else:
        observation_space = spaces.Box(low=min_s, high=max_s,
                                       shape=max_s.shape, dtype=np.float)

    done_criteria = done_criteria
    _max_episode_steps = _max_episode_steps

    return init_state,  expert_replay_buffer, action_space, observation_space, done_criteria, _max_episode_steps, s_scaler, a_scaler


def load_state_customized_environment(bf_pth,
                                      env_name,
                                      max_episode_steps=1000,
                                      rewarder=None,
                                      use_log_transform=False,
                                      use_scaler=False):

    # create cusomized env for state-space-only interaction
    f = open(bf_pth + env_name + '_state.pkl', 'rb')
    raw_expert_replay_buffer = pickle.load(f)
    f.close()

    init_state,  expert_replay_buffer, action_space, observation_space, done_criteria, _max_episode_steps, s_scaler, a_scaler = extract_env_info_from_demo(
        raw_expert_replay_buffer,  _max_episode_steps=max_episode_steps, use_log_transform=use_log_transform, use_scaler=use_scaler)
    env = CustomizedEnv(action_space, observation_space, init_state, expert_replay_buffer,
                        _max_episode_steps, rewarder=rewarder, done_criteria=done_criteria, use_log_transform=use_log_transform, s_scaler=s_scaler,
                        a_scaler=a_scaler, use_scaler=use_scaler)
    #env.set_params(raw_expert_replay_buffer, rewarder=rewarder)
    #environment = DummyVecEnv([lambda: env])
    # wrap it
    #environment = make_vec_env(lambda: environment, n_envs=1)
    # environment.set_params(expert_replay_buffer, rewarder=rewarder)  # observation_space=origin_env.observation_space)

    """Outputs a customized wrapped gym environment."""
    #environment = TimeLimit(environment, max_episode_steps=max_episode_steps)
    # environment = wrap#pers.gym_wrapper.GymWrapper(environment)
    #environment = wrappers.SinglePrecisionWrapper(environment)
    return env


def prefill_rb_with_demonstrations(agent, demonstrations,
                                   num_transitions_rb, reward):
    """Fill the agent's replay buffer with expert transitions."""
    num_demonstrations = len(demonstrations)
    #print('demo:', demonstrations)
    for _ in range(num_transitions_rb // num_demonstrations):
        for i in range(num_demonstrations):
            transition = demonstrations[i]
            observation = transition['observation']
            step_type = transition['step_type']
            discount = 1.0

            ts = dm_env.TimeStep(step_type, reward, discount, observation)
            ts = wrappers.single_precision._convert_value(ts)  # pylint:disable=protected-access

            if step_type == dm_env.StepType.FIRST:
                agent.observe_first(ts)
            else:
                action = demonstrations[i - 1]['action']
                #print(action, ts)
                # We take the previous action to comply with acme's api.
                agent.observe(action, ts)


class DATASET(Dataset):

    def __init__(self, X, y=None):
        self.X = X
        self.Y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index].copy()), torch.from_numpy(self.Y[index].copy())


class GT_dataset():

    def __init__(self, demonstrations, env, customize_env=False):
        self.xs = []
        self.ys = []
        self.env = env
        for traj in demonstrations:
            for pair in traj:
                if customize_env:
                    if env.use_scaler:
                        y = env.a_scaler.transform([pair['action']])[0]
                else:
                    y = pair['action'] / env.action_space.high
                x = pair['observation']
                self.xs.append(x)
                self.ys.append(y)

        self.update_dataloader()

    def update_dataloader(self, batch_size=2048, shuffle=True):
        self.train_dataset = DATASET(X=self.xs, y=self.ys)
        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=batch_size, shuffle=shuffle)
        self.enums = list(enumerate(self.train_loader))

    def append_dataset(self, x, y):
        self.xs.append(x)
        self.ys.append(y)
