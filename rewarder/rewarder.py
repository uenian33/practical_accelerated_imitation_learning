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

"""Rewarder class implementation."""

import copy
import random

import numpy as np
import ot
from sklearn import preprocessing
import enum
# copied from ACME deep mind RL lib
import torch
from torch import nn

from torch.nn import functional as F

from tqdm import tqdm


class StepType(enum.IntEnum):
    """Defines the status of a `TimeStep` within a sequence."""
    # Denotes the first `TimeStep` in a sequence.
    FIRST = 0
    # Denotes any `TimeStep` in a sequence that is not FIRST or LAST.
    MID = 1
    # Denotes the last `TimeStep` in a sequence.
    LAST = 2

    def first(self):
        # type: () -> bool
        return self is StepType.FIRST

    def mid(self):
        # type: () -> bool
        return self is StepType.MID

    def last(self):
        # type: () -> bool
        return self is StepType.LAST


class PWILRewarder(object):
    """Rewarder class to compute PWIL rewards."""

    def __init__(self,
                 demonstrations,
                 subsampling,
                 env_specs,
                 num_demonstrations=1,
                 time_horizon=1000.,
                 alpha=5.,
                 beta=5.,
                 observation_only=False):
        """Initialize the rewarder.

        Args:
          demonstrations: list of expert episodes, comes under the following format:
            [
              [{observation: o_1, action: a_1}, ...], # episode 1
              [{observation: o'_1, action: a'_1}, ...], # episode 2
              ...
            ]
          subsampling: int describing the demonstrations subsamplig frequency.
          env_specs: description of the actions, observations, etc.
          num_demonstrations: int describing the number of demonstration episodes
                              to select at random.
          time_horizon: int time length of the task.
          alpha: float scaling the reward function.
          beta: float controling the kernel size of the reward function.
          observation_only: boolean whether or not to use action to compute reward.
        """
        self.num_demonstrations = num_demonstrations
        self.time_horizon = time_horizon
        self.subsampling = subsampling

        # Observations and actions are flat.
        dim_act = env_specs[0]
        dim_obs = env_specs[1]
        self.reward_sigma = beta * time_horizon / np.sqrt(dim_act + dim_obs)
        # print(self.reward_sigma)
        # t.s()
        self.reward_scale = alpha

        self.observation_only = observation_only
        self.all_demonstrations = demonstrations
        self.demonstrations = self.filter_demonstrations(demonstrations)
        self.vectorized_demonstrations = self.vectorize(self.demonstrations, self.observation_only)
        self.vectorized_observations = self.vectorize(self.demonstrations, True)
        self.scaler = self.get_scaler(self.vectorized_demonstrations)
        self.state_scaler = self.get_scaler(self.vectorized_observations)

    def filter_demonstrations(self, demonstrations):
        """Select a subset of expert demonstrations.

        Select n episodes at random.
        Subsample transitions in these episodes.
        Offset the start transition before subsampling at random.

        Args:
          demonstrations: list of expert demonstrations

        Returns:
          filtered_demonstrations: list of filtered expert demonstrations
        """
        filtered_demonstrations = []
        random.shuffle(demonstrations)
        for episode in demonstrations[:self.num_demonstrations]:
            # Random episode start.
            random_offset = random.randint(0, self.subsampling - 1)
            print("random ep start:", random_offset)
            # Subsampling.
            subsampled_episode = episode[random_offset::self.subsampling]
            # Specify step types of demonstrations.
            for transition in subsampled_episode:
                transition['step_type'] = StepType.MID  # dm_env.StepType.MID
            subsampled_episode[0]['step_type'] = StepType.FIRST  # dm_env.StepType.FIRST
            subsampled_episode[-1]['step_type'] = StepType.LAST  # dm_env.StepType.LAST
            filtered_demonstrations += subsampled_episode
        return filtered_demonstrations

    def vectorize(self, demonstrations, observation_only):
        """Convert filtered expert demonstrations to numpy array.

        Args:
          demonstrations: list of expert demonstrations

        Returns:
          numpy array with dimension:
          [num_expert_transitions, dim_observation] if observation_only
          [num_expert_transitions, (dim_observation + dim_action)] otherwise
        """
        if observation_only:
            demonstrations = [t['observation'] for t in demonstrations]
        else:
            demonstrations = [np.concatenate([t['observation'], t['action']])
                              for t in demonstrations]
        return np.array(demonstrations)

    def get_scaler(self, demos):
        """Defines a scaler to derive the standardized Euclidean distance."""
        scaler = preprocessing.StandardScaler()
        scaler.fit(demos)
        return scaler

    def reset(self):
        """Makes all expert transitions available and initialize weights."""
        self.expert_atoms = copy.deepcopy(
            self.scaler.transform(self.vectorized_demonstrations)
        )
        self.expert_state_atoms = copy.deepcopy(
            self.state_scaler.transform(self.vectorized_observations)
        )
        num_expert_atoms = len(self.expert_atoms)
        self.expert_weights = np.ones(num_expert_atoms) / (num_expert_atoms)

    def get_ideal_target_pair(self, obs_act, a_scaler):
        agent_atom = obs_act['observation']
        agent_atom = np.expand_dims(agent_atom, axis=0)  # add dim for scaler
        agent_atom = self.scaler.transform(agent_atom)[0]

        norms = np.linalg.norm(self.expert_atoms - agent_atom, axis=1)
        argmin = norms.argmin()
        agent_atom = self.scaler.inverse_transform([agent_atom])[0]
        try:
            target_observation = self.scaler.inverse_transform([self.expert_atoms[argmin + 1]])[0]
        except:
            target_observation = self.scaler.inverse_transform([self.expert_atoms[argmin]])[0]
        #print('target_obs', target_observation)
        action = target_observation - agent_atom
        ideal_reward = 1  # self.reward_scale * np.exp(-self.reward_sigma * cost), cost=0

        return [agent_atom, action, target_observation, ideal_reward]

    def get_closest_target_action(self, obs_act):
        # if self.observation_only:
        agent_atom = obs_act['observation']
        # else:
        #    agent_atom = np.concatenate([obs_act['observation'], obs_act['action']])
        agent_atom = np.expand_dims(agent_atom, axis=0)  # add dim for scaler

        agent_atom = self.state_scaler.transform(agent_atom)[0]
        norms = np.linalg.norm(self.expert_state_atoms - agent_atom, axis=1)
        argmin = norms.argmin()
        agent_atom = self.state_scaler.inverse_transform([agent_atom])[0]
        if argmin + 1 < self.expert_state_atoms.shape[0]:
            target_observation = self.state_scaler.inverse_transform([self.expert_state_atoms[argmin + 1]])[0]
        else:
            target_observation = self.state_scaler.inverse_transform([self.expert_state_atoms[argmin]])[0]
        #print('target_obs', target_observation)
        action = target_observation - agent_atom

        return action

    def compute_reward(self, obs_act):
        """Computes reward as presented in Algorithm 1."""
        # Scale observation and action.
        if self.observation_only:
            agent_atom = obs_act['observation']
        else:
            agent_atom = np.concatenate([obs_act['observation'], obs_act['action']])
        agent_atom = np.expand_dims(agent_atom, axis=0)  # add dim for scaler

        agent_atom = self.scaler.transform(agent_atom)[0]

        cost = 0.
        # As we match the expert's weights with the agent's weights, we might
        # raise an error due to float precision, we substract a small epsilon from
        # the agent's weights to prevent that.
        weight = 1. / self.time_horizon - 1e-6
        norms = np.linalg.norm(self.expert_atoms - agent_atom, axis=1)
        # print(norms)
        while weight > 0:
            # Get closest expert state action to agent's state action.
            argmin = norms.argmin()
            expert_weight = self.expert_weights[argmin]
            # print(self.expert_atoms[argmin])
            # Update cost and weights.
            if weight >= expert_weight:
                weight -= expert_weight
                cost += expert_weight * norms[argmin]
                self.expert_weights = np.delete(self.expert_weights, argmin, 0)
                self.expert_atoms = np.delete(self.expert_atoms, argmin, 0)
                norms = np.delete(norms, argmin, 0)
            else:
                cost += weight * norms[argmin]
                self.expert_weights[argmin] -= weight
                weight = 0

        reward = self.reward_scale * np.exp(-self.reward_sigma * cost)
        return reward.astype('float32')

    def compute_w2_dist_to_expert(self, trajectory):
        """Computes Wasserstein 2 distance to expert demonstrations."""
        self.reset()
        # print(trajectory)
        if self.observation_only:
            trajectory = [t['observation'] for t in trajectory]
        else:
            trajectory = [np.concatenate([t['observation'], t['action']])
                          for t in trajectory]
        trajectory = self.scaler.transform(trajectory)
        trajectory_weights = 1. / len(trajectory) * np.ones(len(trajectory))
        cost_matrix = ot.dist(trajectory, self.expert_atoms, metric='euclidean')
        w2_dist = ot.emd2(trajectory_weights, self.expert_weights, cost_matrix, numItermax=200000)
        print(w2_dist)
        return w2_dist


from torch import optim
from torch.utils.data import DataLoader, Dataset

# Gaussian/radial basis function/exponentiated quadratic kernel


def _gaussian_kernel(distances, gamma=100):
    return torch.exp(-gamma * distances)


class EmbeddingNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, target_net=True):
        super().__init__()
        if target_net:
            self.embedding = nn.Sequential(
                nn.Linear(input_size, hidden_size), nn.Tanh(),
                nn.Linear(hidden_size, hidden_size), nn.Tanh(),
                nn.Linear(hidden_size, input_size))
        else:

            self.embedding = nn.Sequential(
                nn.Linear(input_size, hidden_size), nn.Tanh(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, hidden_size), nn.Tanh(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, hidden_size), nn.Tanh(),
                nn.Linear(hidden_size, input_size))

    def forward(self, input):
        return self.embedding(input.float())


class REDRewarder(object):

    def __init__(self,
                 demonstrations,
                 state_size,
                 action_size,
                 hidden_size=128,
                 state_only=True,
                 learning_rate=0.001):
        self.rewarder = REDDiscriminator(state_size, action_size, hidden_size, state_only=state_only)
        self.dataset = REDDataset(demonstrations)
        self.discriminator_optimiser = optim.RMSprop(self.rewarder.parameters(), lr=learning_rate)
        self.max_dis = 1

    def train_rewarder(self, iter_epochs=100, imitation_batch_size=128):
        for _ in tqdm(range(iter_epochs), leave=False):
            # Train predictor network to match random target network
            self.target_estimation_update(imitation_batch_size)

    def eval_rewarder(self):
        eval_dataloader = DataLoader(self.dataset, batch_size=self.dataset.__len__() - 1, shuffle=True, drop_last=True)
        loss = nn.MSELoss(reduction='none')
        #print(self.dataset.pair_number, eval_dataloader)
        # print(list(enumerate(eval_dataloader)))
        for idx, expert_transition in enumerate(eval_dataloader):
            # print(idx)
            # t.s()
            expert_state, expert_action, expert_nect_state = expert_transition[
                'states'], expert_transition['actions'], expert_transition['next_states']

            self.discriminator_optimiser.zero_grad()
            prediction, target = self.rewarder(expert_state, expert_action, expert_nect_state)
            regression_loss = loss(prediction, target)

            self.max_dis = torch.max(torch.sum(loss(prediction, target), dim=1)).detach().numpy()
            print(self.max_dis)
            #a = torch.randn(target.shape[0], target.shape[1])
            #print(torch.sum(loss(a, target), dim=1))

    def get_reward(self, state, action, next_state):
        # try:
        state = torch.from_numpy(state)
        action = torch.from_numpy(action)
        next_state = torch.from_numpy(next_state)
        return self.rewarder.predict_reward(state, action, next_state)
        # except:
        #    return self.rewarder.predict_reward(state, action, next_state)

    def predict_class(self, state, action, next_state,):
        return self.rewarder.predict_class_distance < self.max_dis

    def target_estimation_update(self, batch_size=128):
        expert_dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        for expert_transition in expert_dataloader:
            expert_state, expert_action, expert_nect_state = expert_transition[
                'states'], expert_transition['actions'], expert_transition['next_states']

            self.discriminator_optimiser.zero_grad()
            prediction, target = self.rewarder(expert_state, expert_action, expert_nect_state)
            regression_loss = F.mse_loss(prediction, target)
            regression_loss.backward()
            self.discriminator_optimiser.step()

    def add_new_data(self, state, action, next_state):
        self.dataset.__add_new_data__(state, action, next_state)


class REDDiscriminator(nn.Module):

    def __init__(self, state_size, action_size, hidden_size, state_only=False):
        super().__init__()
        self.action_size, self.state_size, self.state_only = action_size, state_size, state_only
        self.gamma = None
        self.predictor = EmbeddingNetwork(state_size * 2 if state_only else state_size + action_size,
                                          hidden_size,
                                          target_net=True)
        self.target = EmbeddingNetwork(state_size * 2 if state_only else state_size + action_size,
                                       hidden_size * 8,
                                       target_net=False)
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, state, action, next_state):
        state_action = self._join_state_action(
            state, next_state, self.state_size) if self.state_only else self._join_state_action(state, action, self.action_size)
        prediction, target = self.predictor(state_action), self.target(state_action)
        return prediction, target

    def predict_reward(self, state, action, next_state):  # TODO: Set sigma based such that r(s, a) from expert demonstrations ≈ 1
        if self.state_only:
            prediction, target = self.forward(state, action, next_state)
        else:
            prediction, target = self.forward(state, action, next_state)
        # return _gaussian_kernel(F.pairwise_distance(prediction, target, p=2).pow(2), gamma=1e5)
        # return -1 + torch.exp(-1e5 * F.pairwise_distance(prediction, target, p=2).pow(2))
        mse = nn.MSELoss(reduction='none')
        return torch.sum(mse(prediction, target), dim=0)
        # return -1 + torch.exp(-1e4 * torch.square(torch.sum(mse(prediction, target), dim=0)))
    # Concatenates the state and one-hot version of an action

    # TODO: Set sigma based such that r(s, a) from expert demonstrations ≈ 1
    def predict_class_distance(self, state, action, next_state, sigma=1):
        if self.state_only:
            prediction, target = self.forward(state, action, next_state)
        else:
            prediction, target = self.forward(state, action, next_state)
        # return _gaussian_kernel(F.pairwise_distance(prediction, target, p=2).pow(2), gamma=1e5)
        # return -1 + torch.exp(-1e5 * F.pairwise_distance(prediction, target, p=2).pow(2))
        mse = nn.MSELoss(reduction='none')
        return torch.sum(mse(prediction, target), dim=0)

    def _join_state_action(self, state, action, action_size):
        # return torch.cat([state, F.one_hot(action, action_size).to(dtype=torch.float32)], dim=1)
        #print(state, action)
        return torch.cat([state, action], dim=1)

# Dataset that returns transition tuples of the form (s, a, r, s', terminal)


class REDDataset(Dataset):

    def __init__(self, transitions):
        super().__init__()
        self.states = []
        self.actions = []
        self.terminals = []
        self.next_states = []
        self.pair_number = 0
        for traj in transitions[:]:
            # print(traj)
            for idx, t in enumerate(traj):
                if idx != (len(traj) - 1):
                    tmp_states = t['observation']
                    tmp_actions = t['action']
                    #step_type = t['step_type']
                    self.states.append(tmp_states)
                    self.actions.append(tmp_actions)
                    # if StepType.last(step_type):
                    #    self.terminals.append(1)
                    # else:
                    self.terminals.append(0)
                    self.next_states.append(traj[idx + 1]['observation'])
                else:
                    tmp_states = t['observation']
                    tmp_actions = t['action']
                    self.states.append(tmp_states)
                    self.actions.append(tmp_actions)
                    # if StepType.last(step_type):
                    #    self.terminals.append(1)
                    # else:
                    self.terminals.append(1)
                    self.next_states.append(tmp_states)

        self.states = self.states
        self.actions = self.actions
        self.terminals = np.array(self.terminals)
        self.next_states = self.next_states
        self.pair_number = self.terminals.size

        # self.states, self.actions= transitions['states'],
        # transitions['actions'].detach(), transitions['rewards'],
        # transitions['terminals']  # Detach actions

    # Allows string-based access for entire data of one type, or int-based access for single transition
    def __getitem__(self, idx):
        if isinstance(idx, str):
            if idx == 'states':
                return self.states
            elif idx == 'actions':
                return self.actions
        else:
            return dict(states=self.states[idx], actions=self.actions[idx], next_states=self.next_states[idx])
            # return dict(states=self.states[idx], actions=self.actions[idx],
            # rewards=self.rewards[idx], next_states=self.states[idx + 1],
            # terminals=self.terminals[idx])

    def __len__(self):
        return self.pair_number - 1  # Need to return state and next state

    def __add_new_data__(self, s, a, ns):
        self.states.append(s)
        self.actions.append(a)
        self.next_states.append(ns)
        self.pair_number = self.pair_number + 1
        return
