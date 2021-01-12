import pytest
import numpy as np

from stable_baselines3 import A2C
from stable_baselines3.common.identity_env import IdentityEnv, IdentityEnvBox
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


# Hyperparameters for learning identity for each RL model
LEARN_FUNC_DICT = {
    'a2c': lambda e: A2C(policy="MlpPolicy", learning_rate=1e-3, n_steps=1,
                         gamma=0.7, env=e, seed=0).learn(total_timesteps=10000)
}


@pytest.mark.slow
@pytest.mark.parametrize("model_name", ['a2c', 'acer', 'acktr', 'dqn', 'ppo1', 'ppo2', 'trpo'])
def test_identity(model_name):
    """
    Test if the algorithm (with a given policy)
    can learn an identity transformation (i.e. return observation as an action)

    :param model_name: (str) Name of the RL model
    """
    env = DummyVecEnv([lambda: IdentityEnv(10)])

    model = LEARN_FUNC_DICT[model_name](env)
    evaluate_policy(model, env, n_eval_episodes=20, reward_threshold=90)

    obs = env.reset()
    print('test')
    action = env.action_space.sample()
    del model, env


test_identity('a2c')
