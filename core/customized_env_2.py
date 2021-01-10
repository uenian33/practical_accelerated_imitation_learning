import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from sklearn import svm, preprocessing


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


class CustomizedEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    reward_range = (-float('inf'), float('inf'))
    spec = None

    def __init__(self, done_criteria=False):
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = None  # spaces.Box(low=np.array([-1]), high=np.array([1]), shape=np.array([1, 1]), dtype=np.float)
        # Example for using image as input:
        # spaces.Box(low=np.array([-np.inf]), high=np.array([np.inf]), shape=np.array([1, 1]), dtype=np.float)
        self.observation_space = None

        self.done_criteria = done_criteria
        self.state = None
        self.done = False
        self._max_episode_steps = 1000
        self.rewarder = None
        self.expert_replay_buffer = None
        self.use_log_transform = False
        self.a_scaler = None
        self.s_scaler = None
        self.use_scaler = False

    def step(self, action):
        reward = None
        a = action * self.action_space.high

        # print(a)

        a = np.clip(a, self.action_space.low, self.action_space.high)

        if self.use_scaler:
            a = self.a_scaler.inverse_transform([a])[0]
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

        obs = self.state
        next_obs = obs + a
        next_obs = np.clip(next_obs, self.observation_space.low, self.observation_space.high)

        if self.use_scaler:
            return_state = self.s_scaler.inverse_transform([next_obs])[0]
        else:
            return_state = next_obs

        self.state = next_obs

        if self.done_criteria:
            self.done = self.check_done()
        else:
            self.done = False

        if self.rewarder is not None:
            try:
                reward = self.svm_potential_reward_func(obs)
            except:
                obs_act = {'observation': obs, 'action': a}
                reward = self.rewarder.compute_reward(obs_act)
        else:
            reward = 0
            """
        if self.state[0] > 0.45:
            self.done = True
            reward = 100
        else:
            reward = 0
            """
        # print(next_obs)
        return return_state, reward, self.done, {}

    def reset(self, init_obs=None, out_of_set=False):
        if self.rewarder is not None:
            try:
                self.rewarder.reset()
            except:
                raise Exception('cannot reset rewarder')
        if init_obs != None:
            self.state = init_obs
        else:
            s = np.random.choice(np.linspace(0, self.init_state.shape[0] - 1, self.init_state.shape[0], dtype=np.int16), 1)[0]
            #print(s, np.linspace(0, states.shape[0] - 1, states.shape[0], dtype=np.int16))
            s = self.init_state[s]
            #print('random choose initial state from demo', s)
            if not out_of_set:
                # print(s)
                #print(self.observation_space.low, self.observation_space.high)
                # elf.observation_space.low, self.observation_space.high) * 0.0001
                self.state = s + np.random.uniform(-4e-3, 4e-3)
            else:
                # print(self.observation_space.low)
                # print(self.observation_space.high)
                # self.observation_space.low, self.observation_space.high) * 0.0001
                self.state = s + np.random.uniform(-4e-3, 4e-3)

        if self.use_scaler:
            self.state = self.s_scaler.inverse_transform([self.state])[0]
            # print(self.state)
            # print()
            # time.sl()
        self.done = False

        return self.state

    def check_done(self):
        return False

    def render(self, mode='human'):
        return

    def close(self):
        return

    def set_params(self, expert_replay_buffer, action_space=None, observation_space=None, done_criteria=False, _max_episode_steps=1000, rewarder=None, use_scaler=True):
        #self.expert_replay_buffer = np.array(expert_replay_buffer.buffer())
        states = []
        actions = []
        self.init_state = []

        self.use_scaler = use_scaler

        self.rewarder = rewarder
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
            #print('\n', max_a - np.abs(min_a,))
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

        self.done_criteria = done_criteria
        self._max_episode_steps = _max_episode_steps

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
