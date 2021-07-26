import numpy as np
from typing import Optional

from gym import Env, Space
from gym.spaces import Discrete, MultiDiscrete, MultiBinary, Box


class IdentityEnv(Env):

    def __init__(self,
                 dim: Optional[int] = None,
                 space: Optional[Space] = None,
                 ep_length: int = 100):
        """
        Identity environment for testing purposes

        :param dim: the size of the action and observation dimension you want
            to learn. Provide at most one of `dim` and `space`. If both are
            None, then initialization proceeds with `dim=1` and `space=None`.
        :param space: the action and observation space. Provide at most one of
            `dim` and `space`.
        :param ep_length: the length of each episode in timesteps
        """
        if space is None:
            if dim is None:
                dim = 1
            space = Discrete(dim)
        else:
            assert dim is None, "arguments for both 'dim' and 'space' provided: at most one allowed"

        self.action_space = self.observation_space = space
        self.ep_length = ep_length
        self.current_step = 0
        self.num_resets = -1  # Becomes 0 after __init__ exits.
        self.reset()

    def reset(self):
        self.current_step = 0
        self.num_resets += 1
        self._choose_next_state()
        return self.state

    def step(self, action):
        reward = self._get_reward(action)
        self._choose_next_state()
        self.current_step += 1
        done = self.current_step >= self.ep_length
        return self.state, reward, done, {}

    def _choose_next_state(self):
        self.state = self.action_space.sample()

    def _get_reward(self, action):
        return 1 if np.all(self.state == action) else 0

    def render(self, mode='human'):
        pass
