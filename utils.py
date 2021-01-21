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


def load_demonstrations(demo_dir, env_name, state_demo=False, traj_number=50):
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
        return demonstrations[:traj_number]


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
    # expert_replay_buffer = np.array(expert_replay_buffer.buffer())
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
        # print(np.argmax(np.array(tmp), axis=0))
        # print(np.array(tmp).max(axis=0))

    # delta_states = next_states - states
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
    # print(max_s, min_s)
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
    # env.set_params(raw_expert_replay_buffer, rewarder=rewarder)
    # environment = DummyVecEnv([lambda: env])
    # wrap it
    # environment = make_vec_env(lambda: environment, n_envs=1)
    # environment.set_params(expert_replay_buffer, rewarder=rewarder)  # observation_space=origin_env.observation_space)

    """Outputs a customized wrapped gym environment."""
    # environment = TimeLimit(environment, max_episode_steps=max_episode_steps)
    # environment = wrap#pers.gym_wrapper.GymWrapper(environment)
    # environment = wrappers.SinglePrecisionWrapper(environment)
    return env


def prefill_rb_with_demonstrations(agent, demonstrations,
                                   num_transitions_rb, reward):
    """Fill the agent's replay buffer with expert transitions."""
    num_demonstrations = len(demonstrations)
    # print('demo:', demonstrations)
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
                # print(action, ts)
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

    def __init__(self, demonstrations, env, customize_env=False, batch_size=512):
        self.xs = []
        self.ys = []
        self.SAs = []
        self.nextSs = []
        self.enums = None
        self.env = env
        self.batch_size = batch_size
        for traj in demonstrations:
            for idx, pair in enumerate(traj):
                if customize_env:
                    if env.use_scaler:
                        y = env.a_scaler.transform([pair['action']])[0]
                else:
                    y = pair['action'] 
                x = pair['observation']
                self.xs.append(x)
                self.ys.append(y)
                try:
                    self.nextSs.append(traj[idx+1]['action'])
                    self.SAs.append(np.hstack(x, y).flatten())
                except:
                    True

        self.update_dataloader()

    def update_dataloader(self, shuffle=True):
        batch_size = self.batch_size
        self.train_dataset = DATASET(X=self.xs, y=self.ys)
        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=batch_size, shuffle=shuffle)
        self.enums = list(enumerate(self.train_loader))

    def append_dataset(self, x, y):
        self.xs.append(x)
        self.ys.append(y)


class VALUE_dataset():

    def __init__(self,
                 window_size=300,
                 max_env_steps=1000,
                 reward_constant=1.,
                 reward_gamma=0.99,
                 reward_scale=5.,
                 env_max_steps=1000,
                 batch_size=2048,
                 max_store_length=10e7,
                 imitation_rewarder=None):
        self.SAs = []
        self.Ss = []
        self.sub_ys = []
        self.opt_ys = []

        self.tuple_enums = []
        self.trajs = []

        self.window_size = window_size
        self.max_env_steps = max_env_steps
        self.reward_constant = reward_constant
        self.reward_scale = reward_scale
        self.env_max_steps = env_max_steps
        self.reward_gamma = reward_gamma
        self.batch_size = batch_size
        self.max_store_length = max_store_length

        self.optimal_Q = self.reward_scale * self.reward_constant / (1 - self.reward_gamma)  # a/(1-q)
        self.value_type = 'v'  # 'q'
        self.imitation_rewarder = imitation_rewarder

    def reset_dataset(self):
        self.SAs = []
        self.Ss = []
        self.sub_ys = []
        self.opt_ys = []

        self.tuple_enums = []

    def create_MC_subvalue_datasets_from_bc_trajectories(self, trajectories, nsteps=None):
        for traj in trajectories:
            if nsteps is None:
                traj_len = len(traj)
                window_end = traj_len
            else:
                traj_len = len(traj) - nsteps
                window_end = nsteps

            traj = np.array(traj)
            for i in range(traj_len):
                prev_obs = traj[i][0]
                act = traj[i][1]
                obs = traj[i][2]
                r = traj[i][3]
                done = traj[i][4]
                is_expert = traj[i][5]

                if is_expert:
                    self.SAs.append(np.hstack([prev_obs, act]).flatten())
                    self.Ss.append(prev_obs)

                    discounted_sub_R = 0
                    for idx, sub_trans in enumerate(traj[i:i+window_end]):
                        discounted_sub_R += self.reward_gamma**(idx) * sub_trans[3]
                    self.sub_ys.append([discounted_sub_R])
                    print(self.optimal_Q, discounted_sub_R, r, i)

        return

    def parse_demonstrations(self, demonstrations):
        new_trajs = []
        for tdx, traj in enumerate(demonstrations):
            self.imitation_rewarder.reset()
            new_traj = []
            for idx in range(len(traj) - 1):
                act = traj[idx]['action']
                obs = traj[idx]['observation']
                next_obs = traj[idx + 1]['observation']
                done = False
                obs_act = {'observation': obs, 'action': act}
                rew = self.imitation_rewarder.compute_reward(obs_act)

                new_traj.append([obs, act, next_obs, rew, done, True])

            new_trajs.append(new_traj)
        return new_trajs

    def create_MC_optvalue_datasets_from_demonstrations(self, demonstrations, nsteps=None):
        new_trajs = self.parse_demonstrations(demonstrations)
        for traj in new_trajs:
            if nsteps is None:
                traj_len = len(traj)
                window_end = traj_len
            else:
                traj_len = len(traj) - nsteps
                window_end = nsteps
            traj = np.array(traj)
            for i in range(traj_len):
                prev_obs = traj[i][0]
                act = traj[i][1]
                obs = traj[i][2]
                r = traj[i][3]

                self.SAs.append(np.hstack([prev_obs, act]).flatten())
                self.Ss.append(prev_obs)

                discounted_sub_R = 0
                for idx, sub_trans in enumerate(traj[i:i+window_end]):
                    discounted_sub_R += self.reward_gamma**(idx) * sub_trans[3]
                self.opt_ys.append([discounted_sub_R])
                
                print(discounted_sub_R, r, i)

    def create_nstep_MC_tuples_from_demonstrations(self, demonstrations, nsteps=5):
        new_trajs = self.parse_demonstrations(demonstrations)
        for traj in new_trajs:
            traj = np.array(traj)
            for i in range(len(traj) - nsteps):
                prev_obs = traj[i][0]
                act = traj[i][1]
                obs = traj[i][2]
                r = traj[i][3]

                self.SAs.append(np.hstack([prev_obs, act]).flatten())
                self.Ss.append(prev_obs)

                discounted_sub_R = 0
                for idx, sub_trans in enumerate(traj[i:i+nsteps]):
                    discounted_sub_R += self.reward_gamma**(idx) * sub_trans[3]
                self.opt_ys.append([discounted_sub_R])

                if self.value_type == 'v':
                    inputs = torch.FloatTensor(np.array([prev_obs]))
                    sub_Q = self.sub_q_model.model(inputs).detach().numpy().flatten()[0]
                elif self.value_type == 'q':
                    inputs = torch.FloatTensor(np.array([np.hstack(prev_obs, act)]))
                    sub_Q = self.sub_q_model.model(inputs).detach().numpy().flatten()[0]

                

    def create_suboptimal_value_datasets_from_bc_trajectories(self, trajectories, demonstrations=None):
        if demonstrations is not None:
            demo_trajs = self.parse_demonstrations(demonstrations)
            trajectories = demo_trajs

        for traj in trajectories:
            traj_len = len(traj)

            traj = np.array(traj)
            for i in range(traj_len - 1):
                # print(traj[i])
                prev_obs = traj[i][0]
                act = traj[i][1]
                obs = traj[i][2]
                r = traj[i][3]
                done = traj[i][4]
                is_expert = traj[i][5]

                if is_expert:
                    if self.value_type == 'v':
                        inputs = torch.FloatTensor(np.array([prev_obs]))
                        sub_Q = self.sub_q_model.model(inputs).detach().numpy().flatten()[0]
                        opt_Q = self.opt_q_model.model(inputs).detach().numpy().flatten()[0]
                    elif self.value_type == 'q':
                        inputs = torch.FloatTensor(np.array([np.hstack(prev_obs, act)]))
                        sub_Q = self.sub_q_model.model(inputs).detach().numpy().flatten()[0]
                        opt_Q = self.opt_q_model.model(inputs).detach().numpy().flatten()[0]
                    print(sub_Q, opt_Q)
                    # note here r is the suboptimal r from bc policy, 1 is the optimal ideal r
                    #self.tuple_enums.append((np.array(prev_obs), np.array(act), np.array(obs), r, done, opt_Q, sub_Q))

    def init_q_models(self, q_model, x_dim,  suboptimal_trajs, demonstrations, window_size=None, rewarder=None, epochs=500):
        y_dim = 1
        self.imitation_rewarder = rewarder
        self.create_MC_subvalue_datasets_from_bc_trajectories(suboptimal_trajs, nsteps=window_size)
        self.sub_q_model = q_model(train_loader=self.generate_dataloader(optimal=False), x_dim=x_dim,
                                   y_dim=y_dim, epochs=epochs)

        self.sub_q_model.train_BC()
        self.reset_dataset()
        self.create_MC_optvalue_datasets_from_demonstrations(demonstrations, nsteps=window_size)
        self.opt_q_model = q_model(train_loader=self.generate_dataloader(optimal=True), x_dim=x_dim,
                                   y_dim=y_dim, epochs=epochs)
        self.opt_q_model.train_BC()
        test_train_loader=self.generate_dataloader(optimal=True, batch_size=100000)
        for i, (x, labels) in enumerate(self.train_loader):
            x = x.float()
            labels = labels.float()
            outputs = self.opt_q_model.model(x)

            #print(labels - outputs)
        #r()
        self.reset_dataset()

        return

    def generate_dataloader(self, batch_size=512, shuffle=True, optimal=False):
        if optimal:
            ys = self.opt_ys
        else:
            ys = self.sub_ys
        if self.value_type == 'v':
            train_dataset = DATASET(X=np.array(self.Ss), y=np.array(ys))
        elif self.value_type == 'q':
            train_dataset = DATASET(X=np.array(self.SAs), y=np.array(ys))
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle)
        return self.train_loader


class VALUE_dataset__():

    def __init__(self,
                 window_size=300,
                 max_env_steps=1000,
                 reward_constant=1.,
                 reward_gamma=0.99,
                 reward_scale=5.,
                 env_max_steps=1000,
                 batch_size=2048,
                 max_store_length=10e7):
        self.sub_SAs = []
        self.sub_Ss = []
        self.sub_ys = []
        self.opt_SAs = []
        self.opt_Ss = []
        self.opt_ys = []

        self.opt_Q_enums = []
        self.opt_V_enums = []
        self.sub_Q_enums = []
        self.sub_V_enums = []
        self.trajs = []

        self.window_size = window_size
        self.max_env_steps = max_env_steps
        self.reward_constant = reward_constant
        self.reward_scale = reward_scale
        self.env_max_steps = env_max_steps
        self.reward_gamma = reward_gamma
        self.batch_size = batch_size
        self.max_store_length = max_store_length

        self.optimal_Q = self.reward_scale * self.reward_constant / (1 - self.reward_gamma)  # a/(1-q)

        # self.create_optimal_value_datasets_from_demos(demonstrations)

    def update_dataloader(self, batch_size=2048, shuffle=True):
        self.sub_Q_train_dataset = DATASET(X=self.sub_SAs, Y=self.sub_ys)
        self.sub_Q_train_loader = DataLoader(dataset=self.sub_Q_train_dataset, batch_size=self.batch_size, shuffle=shuffle)
        self.sub_Q_enums = list(enumerate(self.sub_Q_train_loader))

        self.sub_V_train_dataset = DATASET(X=self.sub_Ss, Y=self.sub_ys)
        self.sub_V_train_loader = DataLoader(dataset=self.sub_V_train_dataset, batch_size=self.batch_size, shuffle=shuffle)
        self.sub_V_enums = list(enumerate(self.sub_V_train_loader))

        self.opt_Q_train_dataset = DATASET(X=self.opt_SAs, Y=self.opt_ys)
        self.opt_Q_train_loader = DataLoader(dataset=self.opt_Q_train_dataset, batch_size=self.batch_size, shuffle=shuffle)
        self.opt_Q_enums = list(enumerate(self.opt_Q_train_loader))

        self.opt_V_train_dataset = DATASET(X=self.opt_Ss, Y=self.opt_ys)
        self.opt_V_train_loader = DataLoader(dataset=self.opt_V_train_dataset, batch_size=self.batch_size, shuffle=shuffle)
        self.opt_V_enums = list(enumerate(self.opt_V_train_loader))

    def create_suboptimal_value_datasets_from_trajectories(self, demonstrations):
        print("generating suboptimal values")
        for traj in demonstrations:
            traj_len = len(traj)
            for i in range(traj_len - self.window_size):
                sub_traj = traj[i:i + self.window_size]
                discounted_sub_R = 0
                for idx, trans in enumerate(sub_traj):
                    discounted_sub_R += self.reward_gamma**idx * trans[3]

                # if traj_len == max_env_steps:
                # the pairs are created only to estimate <s,a> in expert demo
                rest_optimal_Q = self.reward_gamma**self.window_size * self.optimal_Q
                suboptimal_Q = rest_optimal_Q + discounted_sub_R

                self.sub_SAs.append(np.hstack([traj[i][0], traj[i][1]]).flatten())

                self.sub_Ss.append(traj[i][0])

                self.sub_ys.append([suboptimal_Q])

                self.sub_Q_enums.append((np.hstack([traj[i][0], traj[i][1]]).flatten(), traj[i][3], suboptimal_Q))
                self.sub_V_enums.append((traj[i][0], traj[i][3], suboptimal_Q))

        # self.update_dataloader()
        return

    def create_optimal_value_datasets_from_demos(self, demonstrations):

        for traj in demonstrations:
            traj_len = len(traj)
            for i in range(traj_len - 1):
                self.opt_SAs.append(np.hstack([traj[i]['observation'], traj[i]['action']]).flatten())

                self.opt_Ss.append(traj[i]['observation'])

                self.opt_ys.append([self.optimal_Q])

                self.opt_Q_enums.append((np.hstack([traj[i]['observation'], traj[i]['action']]).flatten(), 1, self.optimal_Q))

                self.opt_V_enums.append((traj[i]['observation'], 1, self.optimal_Q))
        """
        self.opt_Q_train_dataset = DATASET(X=self.opt_SAs, Y=self.opt_ys)
        self.opt_Q_train_loader = DataLoader(dataset=self.opt_Q_train_dataset, batch_size=128, shuffle=True)

        self.opt_V_train_dataset = DATASET(X=self.opt_Ss, Y=self.opt_ys)
        self.opt_V_train_loader = DataLoader(dataset=self.opt_V_train_dataset, batch_size=128, shuffle=True)
        """

        return

    def create_value_datasets_from_online_sample(self, sub_trajs):
        print("adding estimated Q value based on known data...")
        for idx, trans in enumerate(sub_traj):  # trans = [prev_obs, action, next_obs, reward, done]
            discounted_sub_R = 0
            sub_sub_traj = sub_traj[idx:-1]
            for jdx, trans_hat in enumerate(sub_sub_traj):
                discounted_sub_R += sub_sub_traj[jdx][3]
            optimal_Q = discounted_sub_R + self.reward_gamma**(sub_sub_traj.size[0] + 1) * self.optimal_Q

            suboptimal_Q = discounted_sub_R + self.reward_gamma**(sub_sub_traj.size[0] + 1) * self.suboptimal_Q
            self.opt_SAs.append(np.stack(sub_traj[idx][0], sub_traj[idx][1]).flatten())
            self.sub_SAs.append(np.stack(sub_traj[idx][0], sub_traj[idx][1]).flatten())
            self.opt_ys.append([optimal_Q])
            self.sub_ys.append([suboptimal_Q])

            if len(self.opt_Q_enums) < self.max_store_length:
                self.opt_Q_enums.append((np.stack(sub_traj[idx][0], sub_traj[idx][1]).flatten(), reward, optimal_Q))
                self.opt_V_enums.append((sub_traj[idx][0], reward, optimal_Q))

                self.sub_Q_enums.append((np.stack(sub_traj[idx][0], sub_traj[idx][1]).flatten(), reward, suboptimal_Q))
                self.sub_V_enums.append((sub_traj[idx][0], reward, suboptimal_Q))

        return

    def create_value_datasets_from_demos(self, trajectories):

        for traj in trajectories:
            if len(traj) > 300:
                traj_len = 300
            else:
                traj_len = len(traj)

            traj = np.array(traj)
            for i in range(traj_len):
                prev_obs = traj[i][0]
                act = traj[i][1]
                obs = traj[i][2]
                r = traj[i][3]
                done = traj[i][4]
                is_expert = traj[i][5]

                if is_expert:
                    self.opt_SAs.append(np.hstack([traj[i]['observation'], traj[i]['action']]).flatten())
                    self.opt_Ss.append(traj[i]['observation'])
                    self.opt_ys.append([self.optimal_Q])

                    discounted_sub_R = r
                    for idx, sub_trans in enumerate(traj[i:]):
                        discounted_sub_R += self.reward_gamma**(idx + 1) * sub_trans[3]
                    self.sub_ys.append([self.discounted_sub_R])

                self.opt_Q_enums.append((np.hstack([traj[i]['observation'], traj[i]['action']]).flatten(), 1, self.optimal_Q))

                self.opt_V_enums.append((traj[i]['observation'], 1, self.optimal_Q))
        """
        self.opt_Q_train_dataset = DATASET(X=self.opt_SAs, Y=self.opt_ys)
        self.opt_Q_train_loader = DataLoader(dataset=self.opt_Q_train_dataset, batch_size=128, shuffle=True)

        self.opt_V_train_dataset = DATASET(X=self.opt_Ss, Y=self.opt_ys)
        self.opt_V_train_loader = DataLoader(dataset=self.opt_V_train_dataset, batch_size=128, shuffle=True)
        """

        return
