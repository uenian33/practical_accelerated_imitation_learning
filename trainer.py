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

"""PWIL training script."""

from absl import app
from absl import flags
from acme import specs

import imitation_loop
import rewarder.rewarder as rewarder
import utils

from core.TD3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

from stable_baselines3.common import logger

import gym
from gym.wrappers.time_limit import TimeLimit

flags.DEFINE_string('workdir', None, 'Logging directory')
flags.DEFINE_string('env_name', None, 'Environment name.')
flags.DEFINE_string('demo_dir', 'demo/', 'Directory of expert demonstrations.')
flags.DEFINE_boolean('state_only', False,
                     'Use only state for reward computation')
flags.DEFINE_float('sigma', 0.2, 'Exploration noise.')
flags.DEFINE_integer('num_transitions_rb', 80,
                     'Number of transitions to fill the rb with.')
flags.DEFINE_integer('num_demonstrations', 50, 'Number of expert episodes.')
flags.DEFINE_integer('subsampling', 1, 'Subsampling factor of demonstrations.')
flags.DEFINE_integer('random_seed', 1, 'Experiment random seed.')
flags.DEFINE_integer('num_steps_per_iteration', 10000000000,
                     'Number of training steps per iteration.')
flags.DEFINE_integer('num_iterations', 100, 'Number of iterations.')
flags.DEFINE_integer('num_eval_episodes', 10, 'Number of evaluation episodes.')
flags.DEFINE_integer('samples_per_insert', 256, 'Controls update frequency.')
flags.DEFINE_float('policy_learning_rate', 1e-4,
                   'Larning rate for policy updates')
flags.DEFINE_float('critic_learning_rate', 1e-4,
                   'Larning rate for critic updates')

flags.DEFINE_string('original_trainer_type', 'False', 'Directory of expert demonstrations.')
flags.DEFINE_integer('ep_steps', 1000, 'envionrment ep running steps')

FLAGS = flags.FLAGS


def main(_):

    from stable_baselines3.common.env_checker import check_env
    # It will check your custom environment and output additional warnings if needed
    from core.customized_env import CustomizedEnv
    # If the environment don't follow the interface, an error will be thrown

    # real_environment = utils.load_environment(FLAGS.env_name, max_episode_steps=FLAGS.ep_steps)
    # environment = DummyVecEnv([environment])
    # Create Rewarder.

    # show initial config
    print("Logger outputs at startup:", logger.Logger.CURRENT.output_formats)

    # set up logger
    logger.configure("logs/" + FLAGS.env_name, ["stdout", "tensorboard"])
    print("Logger outputs before training:", logger.Logger.CURRENT.output_formats)

    demonstrations = utils.load_demonstrations(
        demo_dir=FLAGS.demo_dir, env_name=FLAGS.env_name, state_demo=False)

    environment_spec = [demonstrations[0][0]['action'].shape[0], demonstrations[0][0]['observation'].shape[0]]

    pwil_rewarder = rewarder.PWILRewarder(
        demonstrations,
        subsampling=FLAGS.subsampling,
        env_specs=environment_spec,
        num_demonstrations=FLAGS.num_demonstrations,
        observation_only=FLAGS.state_only)
    spec_rewarder = rewarder.REDRewarder(demonstrations, environment_spec[1],
                                         environment_spec[0], 128, state_only=FLAGS.state_only)
    spec_rewarder.train_rewarder(iter_epochs=0)

    # Load environment.
    # environment = utils.load_state_customized_environment(
    #    FLAGS.demo_dir, FLAGS.env_name, rewarder=pwil_rewarder, max_episode_steps=FLAGS.ep_steps)
   # """
    environment = gym.make(FLAGS.env_name)
    environment = TimeLimit(environment, max_episode_steps=FLAGS.ep_steps)

    n_actions = environment.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.4 * np.ones(n_actions))
    # define cutomized td3
    model = TD3('MlpPolicy', environment, action_noise=action_noise, verbose=1,
                rewarder=pwil_rewarder,  # spec_rewarder,
                reward_type='pwil',)

    # load expert supervised dataset
    sl_dataset = utils.GT_dataset(demonstrations, environment)
    model.sl_dataset = sl_dataset
    model.value_dataset = utils.VALUE_dataset(demonstrations)
    model.use_acceleration = True

    # pretrain the actor using behaviour cloning
    model.pretrain_actor_using_demo(sl_dataset, epochs=300)

    model.sample_trajs(n_episodes=2)
    # model.pretrain_critic_using_demo()
    model.learn(total_timesteps=1e6)

    print("Logger outputs after training:", logger.Logger.CURRENT.output_formats)

    from datetime import date

    today = date.today()
    print('weights/' + FLAGS.env_name + today.strftime("%b-%d-%Y"))
    model.save('weights/' + FLAGS.env_name + '_state_' + str(FLAGS.state_only) + '_' + today.strftime("%b-%d-%Y"))

   # """
    # train model
    # from stable_baselines3 import TD3
    # MountainCarContinuous-v0 LunarLanderContinuous-v2
    """
    env = gym.make('Hopper-v2')
    env._max_episode_steps = 1000

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    model = TD3('MlpPolicy', env, action_noise=action_noise, verbose=1)
    model.learn(total_timesteps=1000000)

    obs = env.reset()
    for i in range(1000000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()

    env = model.get_env()
    for i in range(0):
        obs = env.reset()
        rs = 0
        done = False
        while not done:
            action, _states = model.predict(obs)
            # print(action)
            obs, rewards, done, info = env.step(action)

            rs += rewards
        print(rs)
    env.close()
    """


if __name__ == '__main__':
    app.run(main)  # origin_trainer
    #
