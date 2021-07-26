from gym import error, spaces, utils
from gym.utils import seeding
# from sklearn import svm

import numpy as np
from typing import Optional

from gym import Env, Space
from gym.spaces import Discrete, MultiDiscrete, MultiBinary, Box
import matplotlib.pyplot as plt
import os
import copy

plt.figure()


class Customize_scaler():

    def __init__(self, start=-1, end=1):
        #self.data = np.array(data)
        self.start = start
        self.end = end
        self.width = end - start
        self.origin_data_min = None
        self.origin_data_max = None

    def fit(self, data):
        self.origin_data_min = np.array(data).min(axis=0)
        self.origin_data_max = np.array(data).max(axis=0)

    def transform(self, data):
        data = np.array(data)
        res = (data - self.origin_data_min) / (self.origin_data_max - self.origin_data_min) * self.width + self.start

        return res

    def inverse_transform(self, data):
        data = np.array(data)
        res = ((data - self.start) / self.width) * (self.origin_data_max -
                                                    self.origin_data_min) + self.origin_data_min
        return res


def plot_traj(policy_traj_all, rewarder, ep_num):
    exp_traj_all = rewarder.all_demonstrations[0]
    fig_path = 'figs/'
    exp_traj = [t['observation'] for t in exp_traj_all]
    exp_traj = np.array(exp_traj)
    #exp_traj = rewarder.scaler.transform(exp_traj)

    policy_traj = [t['observation'] for t in policy_traj_all]
    policy_traj = np.array(policy_traj)
    #policy_traj = rewarder.scaler.transform(policy_traj)

    for s_id in range(exp_traj.shape[1]):
        save_path = fig_path + str(s_id) + '/'
        exp_s = exp_traj[:, s_id].flatten()
        policy_s = policy_traj[:, s_id].flatten()
        plt.plot(exp_s, color='red', linewidth=.04)
        plt.ylabel('expert')
        plt.plot(policy_s, linewidth=.04)
        plt.ylabel('policy')
        # plt.show()
        fig_final_path = save_path + 'compare_traj_' + str(ep_num) + '.pdf'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(fig_final_path)
        plt.clf()

    exp_traj = [t['action'] for t in exp_traj_all]
    exp_traj = np.array(exp_traj)
    #exp_traj = rewarder.scaler.transform(exp_traj)

    policy_traj = [t['action'] for t in policy_traj_all]
    policy_traj = np.array(policy_traj)
    #policy_traj = rewarder.scaler.transform(policy_traj)

    for s_id in range(exp_traj.shape[1]):

        save_path = fig_path + str(s_id) + '/'
        exp_s = exp_traj[:, s_id].flatten()
        policy_s = policy_traj[:, s_id].flatten()
        plt.plot(exp_s, color='red', linewidth=.04)
        plt.ylabel('expert')
        plt.plot(policy_s, linewidth=.04)
        plt.ylabel('policy')
        # plt.show()
        fig_final_path = save_path + 'compare_delta_traj_' + str(ep_num) + '.pdf'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(fig_final_path)
        plt.clf()
    # print(policy_traj)
    #print(policy_traj.shape[0], policy_traj.shape[1])
    plt.clf()

    return


class CustomizedEnv(Env):
    metadata = {'render.modes': ['human']}
    reward_range = (-float('inf'), float('inf'))

    def __init__(self, action_space, observation_space, init_state, expert_replay_buffer,
                 _max_episode_steps,
                 rewarder=None,
                 done_criteria=False,
                 use_log_transform=False,
                 a_scaler=None,
                 s_scaler=None,
                 use_scaler=False):
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        # spaces.Box(low=np.array([-1]), high=np.array([1]), shape=np.array([1, 1]), dtype=np.float)
        self.action_space = action_space
        # Example for using image as input:
        # spaces.Box(low=np.array([-np.inf]), high=np.array([np.inf]), shape=np.array([1, 1]), dtype=np.float)
        self.observation_space = observation_space

        self.done_criteria = done_criteria
        self.state = None
        self.done = False
        self._max_episode_steps = _max_episode_steps
        self.rewarder = rewarder
        self.ideal_rewarder = copy.deepcopy(self.rewarder)
        self.expert_replay_buffer = expert_replay_buffer
        self.use_log_transform = use_log_transform
        self.init_state = init_state
        self.step_cnt = 0
        self.ep_num = 0
        self.a_scaler = a_scaler
        self.s_scaler = s_scaler
        self.use_scaler = use_scaler

    def set_params(self,
                   expert_replay_buffer,
                   action_space=None,
                   observation_space=None,
                   done_criteria=False,
                   _max_episode_steps=1000,
                   rewarder=None,
                   use_scaler=False):
        #self.expert_replay_buffer = np.array(expert_replay_buffer.buffer())
        states = []
        actions = []
        self.init_state = []

        self.use_scaler = use_scaler

        self.rewarder = rewarder
        self.ideal_rewarder = copy.deepcopy(self.rewarder)
        print(len(expert_replay_buffer))
        for traj in expert_replay_buffer:
            tmp = []
            # print(len(traj))
            for idx, trans in enumerate(traj):
                states.append(trans['observation'])
                act = trans['action']
                if self.use_log_transform:
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
                    self.init_state.append(trans['observation'])
            #print(np.argmax(np.array(tmp), axis=0))
            # print(np.array(tmp).max(axis=0))

        if self.use_scaler:
            self.s_scaler = Customize_scaler()
            self.s_scaler.fit(states)
            self.a_scaler = Customize_scaler()
            self.a_scaler.fit(actions)
            # print(np.array(actions).max(axis=0))
            # print(np.array(actions).min(axis=0))
            states = self.s_scaler.transform(np.array(states))
            actions = self.a_scaler.transform(np.array(actions))
            self.init_state = self.s_scaler.transform(np.array(self.init_state))
            # print(states.shape)
            max_s = np.ones(states.shape[1])
            max_a = np.ones(states.shape[1])
            min_s = np.ones(states.shape[1]) * -1
            min_a = np.ones(states.shape[1]) * -1

            # print(self.a_scaler.inverse_transform(max_a))
            # print(self.a_scaler.inverse_transform(min_a))

        else:
            states = np.array(states)
            actions = np.array(actions)
            self.init_state = np.array(self.init_state)
            #max_s = states.max(axis=0)
            #max_a = actions.max(axis=0)
            #min_s = states.min(axis=0)
            #min_a = actions.min(axis=0)
            #print(max_s, '\n', min_s, '\n', max_a, '\n', min_a,)
            # print()
            #print('\n', max_a - np.abs(min_a,))
            # print()

            max_s = np.abs(states).max(axis=0)
            max_a = np.abs(actions).max(axis=0)
            min_s = -1 * max_s
            min_a = -1 * max_a
            #print('\n', max_a, max_s)
            # time.sl()

        self.expert_replay_buffer = [states, actions]

        if action_space is not None:
            self.action_space = action_space
        else:
            self.action_space = spaces.Box(low=min_a, high=max_a,
                                           shape=max_a.shape, dtype=np.float)

        if observation_space is not None:
            self.observation_space = observation_space
        else:
            self.observation_space = spaces.Box(low=min_s, high=max_s,
                                                shape=max_s.shape, dtype=np.float)

        print(self.action_space, self.observation_space)

        self.done_criteria = done_criteria
        self._max_episode_steps = _max_episode_steps

        return

    def step(self, action):
        reward = None
        a = action  # * self.action_space.high

        # print(a)

        a = np.clip(a, self.action_space.low, self.action_space.high)

        if self.use_scaler:
            a = self.a_scaler.inverse_transform(a)

        # print(self.action_space.high)
        # print(a)

        # print()

        # if use log transform, then revert the action value
        if self.use_log_transform:
            tmp_a = a
            signs = np.where((tmp_a > 0).astype(np.int) == 1, 1, a)
            signs = np.where((tmp_a < 0).astype(np.int) == 1, -1, signs)
            # print(signs)

            a = (2**tmp_a - 1.) / 1 * signs  # act = np.log10(1 + 1e7 * abs(act)) * signs 1+ 10e7*a = 10**v - 1 /
            # print(a)

        prev_obs = self.state
        next_obs = prev_obs + a
        next_obs = np.clip(next_obs, self.observation_space.low, self.observation_space.high)
        self.state = next_obs

        if self.use_scaler:
            return_state = self.s_scaler.inverse_transform([self.state])[0]
        else:
            return_state = self.state

        if self.done_criteria:
            self.done = self.check_done()
        else:
            bigger = next_obs > self.observation_space.high
            smaller = next_obs < self.observation_space.low

            if self.step_cnt > self._max_episode_steps - 1:  # or bigger.any() or smaller.any():
                self.done = True
                if self.ep_num % 20 == 0 or self.ep_num == 1:
                    plot_traj(self.tmp_traj, self.rewarder, ep_num=self.ep_num)
            else:
                self.done = False

        reward = self._get_reward(prev_obs, return_state, a)
       # print(reward)
        self.step_cnt += 1
        if self.use_scaler:
            self.tmp_traj.append({"observation": self.s_scaler.inverse_transform([obs])[0], "action": a})
        else:
            self.tmp_traj.append({"observation": prev_obs, "action": a})

        return return_state, reward, self.done, {'ideal_tuple': self.ideal_tuple}

    def _get_reward(self, prev_obs, next_obs, a):
        if self.use_scaler == True:
            prev_obs = self.s_scaler.transform(prev_obs)

        if self.rewarder is not None:
            try:
                reward = self.svm_potential_reward_func(next_obs)
            except:
                obs_act = {'observation': prev_obs, 'action': a}
                not_right_obs_act = {'observation': next_obs, 'action': a}
                ideal_action = self.ideal_rewarder.get_closest_target_action(obs_act)
                if self.use_scaler:
                    clip_ideal_action = np.clip(self.a_scaler.transform(ideal_action),
                                                self.action_space.low, self.action_space.high)
                else:
                    clip_ideal_action = np.clip(ideal_action, self.action_space.low, self.action_space.high)
                ideal_state = np.clip(prev_obs + clip_ideal_action, self.observation_space.low, self.observation_space.high)
                ideal_obs_act = {'observation': ideal_state, 'action': clip_ideal_action}

                reward = self.rewarder.compute_reward(not_right_obs_act)
                ideal_reward = self.ideal_rewarder.compute_reward(ideal_obs_act)
                #print((reward - ideal_reward) > 0)
                tmp_tuple = [prev_obs, clip_ideal_action, ideal_state, ideal_reward]
                self.ideal_tuple = self.get_ideal_tuple(tmp_tuple)

        else:
            reward = 0

        return reward

    def get_ideal_tuple(self, ideal_tuple):
        obs = ideal_tuple[0]
        if self.use_scaler:
            action = self.a_scaler.transform([ideal_tuple[1]])[0]
        else:
            action = ideal_tuple[1]
        action = np.clip(action, self.action_space.low, self.action_space.high)
        next_obs = ideal_tuple[2]
        reward = ideal_tuple[3]
        done = self.done
        return obs, action, next_obs, reward, done

    def reset(self, init_obs=None, out_of_set=False):
        self.ep_num += 1
        self.tmp_traj = []
        self.rewarder.reset()
        self.ideal_rewarder.reset()

        if init_obs != None:
            self.state = init_obs
        else:
            s = np.random.choice(np.linspace(0, self.init_state.shape[0] - 1, self.init_state.shape[0], dtype=np.int16), 1)[0]
            # print(s, np.linspace(0, states.shape[0] - 1, states.shape[0], dtype=np.int16))
            s = self.init_state[s]
            # print('random choose initial state from demo', s)
            if not out_of_set:
                # print(s)
                # print(self.observation_space.low, self.observation_space.high)
                self.state = s + np.random.uniform(self.observation_space.low, self.observation_space.high) * 1e-5
            else:
                # print(self.observation_space.low)
                # print(self.observation_space.high)
                self.state = s + np.random.uniform(self.observation_space.low, self.observation_space.high) * 1e-5

        if self.use_scaler:
            return_state = self.s_scaler.inverse_transform(self.state)
        else:
            return_state = self.state

        self.done = False
        self.step_cnt = 0
        # print('reset', self.state)
        return return_state

    def check_done(self):
        return False

    def render(self, mode='human'):
        return

    def close(self):
        return

    def set_OCC_rewarder(self):
        states = np.vstack(self.expert_replay_buffer[:, 0])

        demo = states
        print('training one class svm...')
        self.rewarder = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
        self.rewarder.fit(demo)
        """
                                                                obs = np.expand_dims(np.array(states[0]), axis=0)
                                                                r = self.rewarder.decision_function(obs)
                                                                print(obs, r)
                                                                """

        return

    def svm_potential_reward_func(self, obs):
        obs = np.expand_dims(np.array(obs), axis=0)
        r = self.rewarder.decision_function(obs)

        return r
