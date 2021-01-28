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
from rewarder import rewarder, ensemble_models, behavior_cloning
import utils

from core.TD3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

from stable_baselines3.common import logger

import gym
from gym.wrappers.time_limit import TimeLimit
import pybullet_envs
import pickle

flags.DEFINE_string('workdir', None, 'Logging directory')
flags.DEFINE_string('env_name', None, 'Environment name.')
flags.DEFINE_string('q_bound_type', None, 'Define how to add constrain to q learning') #[None, ,'DDPGfD','standard_lower_bound','expert_lower_bound','expert_upper_bound','target_bound','hybrid']
flags.DEFINE_string('demo_dir', 'demo/', 'Directory of expert demonstrations.')
flags.DEFINE_boolean('state_only', False,
                     'Use only state for reward computation')
flags.DEFINE_float('sigma', 0.2, 'Exploration noise.')
flags.DEFINE_integer('num_transitions_rb', 80,
                     'Number of transitions to fill the rb with.')
flags.DEFINE_integer('num_demonstrations', 4, 'Number of expert episodes.')
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
flags.DEFINE_integer('ep_steps', 300, 'envionrment ep running steps')

FLAGS = flags.FLAGS

from torchensemble import BaggingRegressor           # import ensemble method (e.g., VotingClassifier)
import torch
import time

RANDOM_SEED = 0
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def generate_suboptimal_trajectories(environment, bc_model, rewarder, sa_classifier, n_trajs=15):
    obs = environment.reset()
    trajectories = []
    for ep in range(n_trajs):
        print(ep)
        traj = []
        finished = False
        rewarder.reset()
        steps = 0
        sum_r = 0
        discounted_sum_r = 0
        step_id = 0
        while not finished:
            prev_obs = obs
            inputs = torch.FloatTensor(np.array([obs]))
            # action =
            action = bc_model.model(inputs).detach().numpy()[0] + (np.random.random_sample((environment.action_space.shape[-1],)) - 0.5) * 1.3
            # print(action)
            obs, reward, done, info = environment.step(action)
            obs_act = {'observation': prev_obs, 'action': action}
            imitation_reward = rewarder.compute_reward(obs_act)
            sum_r += imitation_reward
            discounted_sum_r += 0.99**step_id * imitation_reward
            step_id += 1
            in_expert = sa_classifier.predict_class(prev_obs, action, obs)
            traj.append([prev_obs, action, obs, imitation_reward, done, in_expert])
            environment.render()
            finished = done
            steps += 1
            if (steps > 999 and not finished) or finished:
                obs = environment.reset()
                trajectories.append(traj)

        if not done:
            trajectories.append(traj)

        print(sum_r, discounted_sum_r)

    return trajectories


def init_datasets_and_models(demonstrations, environment, imitation_rewarder,
                             bc_dataset=None,
                             sa_classifier=None,
                             bc_model=None,
                             suboptimal_trajs=None,
                             ensemble_models_save_pth=None):
    if bc_dataset is None:
        # load expert supervised dataset
        bc_dataset = utils.GT_dataset(demonstrations, environment, 
                                    imitation_rewarder=imitation_rewarder,
                                    bc=True, 
                                    nsteps=10, 
                                    reward_gamma=0.99)

    if bc_model is None:
        bc_model = behavior_cloning.BehaviorCloning(train_loader=bc_dataset.train_loader, x_dim=bc_dataset.xs[0].shape[0],
                                                    y_dim=bc_dataset.ys[0].shape[0], epochs=200)
        bc_model.train_BC()


    if sa_classifier is None:
        """ use RED as classifier
        sa_classifier = rewarder.REDRewarder(demonstrations, environment_spec[1],
                                             environment_spec[0], 128, state_only=FLAGS.state_only)
        sa_classifier.train_rewarder(iter_epochs=100)
        sa_classifier.eval_rewarder()
        """

        sa_classifier = ensemble_models.EnsembleModels(bc_dataset,
                                                       n_estimators=5,
                                                       lr=1e-3,
                                                       weight_decay=5e-4,
                                                       epochs=220,
                                                       batch_size=512,
                                                       cuda=False,
                                                       n_jobs=1,
                                                       save_dir=ensemble_models_save_pth,
                                                       value_type='q',
                                                       dynamic_pair=False)


    if suboptimal_trajs is None:
        # use noise injected BC to collect trajectories
        suboptimal_trajs = generate_suboptimal_trajectories(environment, bc_model, imitation_rewarder, sa_classifier,
                                                            n_trajs=5)

    # init value dataset
    value_dataset = utils.VALUE_dataset(value_type='q')

    value_dataset.init_q_models(behavior_cloning.BehaviorCloning,
                                suboptimal_trajs=suboptimal_trajs,
                                demonstrations=demonstrations,
                                window_size=10,
                                rewarder=imitation_rewarder, epochs=2000)

    value_dataset.create_suboptimal_value_datasets_from_bc_trajectories(suboptimal_trajs, 
                                                                        demonstrations=demonstrations,
                                                                       )
    return bc_dataset, sa_classifier, bc_model, suboptimal_trajs, value_dataset


def main(_):

    # the path of saved objects
    pth_name = 'pkl/' + FLAGS.env_name + '_' + 'subsampling' + str(FLAGS.subsampling) + '_' + 'trajs' + str(FLAGS.num_demonstrations)
    bc_dataset_pkl_pth = pth_name + '_bc_dataset.pkl'
    sa_classifier_pkl_pth = pth_name + '_sa_classifier.pkl'
    bc_model_pkl_pth = pth_name + '_bc_model.pkl'
    suboptimal_trajs_pkl_pth = pth_name + '_suboptimal_trajs.pkl'
    value_dataset_pkl_pth = pth_name + '_value_dataset.pkl'
    ensemble_models_save_pth = 'pkl/ensemble_' + FLAGS.env_name + '_' + 'subsampling' + \
        str(FLAGS.subsampling) + '_' + 'trajs' + str(FLAGS.num_demonstrations) + '/'

    from stable_baselines3.common.env_checker import check_env
    # It will check your custom environment and output additional warnings if needed
    from core.customized_env import CustomizedEnv
    # If the environment don't follow the interface, an error will be thrown

    # show initial config
    print("Logger outputs at startup:", logger.Logger.CURRENT.output_formats)

    # set up logger
    logger.configure("logs/" + FLAGS.env_name, ["stdout", "tensorboard"])
    print("Logger outputs before training:", logger.Logger.CURRENT.output_formats)

    demonstrations = utils.load_demonstrations(
        demo_dir=FLAGS.demo_dir, env_name=FLAGS.env_name, state_demo=False, traj_number=FLAGS.num_demonstrations)

    imitation_rewarder = rewarder.PWILRewarder(
        demonstrations,
        subsampling=FLAGS.subsampling,
        env_specs=[demonstrations[0][0]['action'].shape[0], demonstrations[0][0]['observation'].shape[0]],
        num_demonstrations=FLAGS.num_demonstrations,
        observation_only=FLAGS.state_only)
    # Load environment.
    # environment = utils.load_state_customized_environment(
    #    FLAGS.demo_dir, FLAGS.env_name, rewarder=imitation_rewarder, max_episode_steps=FLAGS.ep_steps)
   # """
    environment_spec = [demonstrations[0][0]['action'].shape[0], demonstrations[0][0]['observation'].shape[0]]
    environment = gym.make(FLAGS.env_name)
    environment = TimeLimit(environment, max_episode_steps=1000)


    environment.seed(RANDOM_SEED)
    environment.action_space.seed(RANDOM_SEED)

    n_actions = environment.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0. * np.ones(n_actions))

   

    try:
        with open(bc_dataset_pkl_pth, 'rb') as inputs:
            bc_dataset = pickle.load(inputs)

        with open(sa_classifier_pkl_pth, 'rb') as inputs:
            sa_classifier = pickle.load(inputs)

        with open(bc_model_pkl_pth, 'rb') as inputs:
            bc_model = pickle.load(inputs)

        with open(suboptimal_trajs_pkl_pth, 'rb') as inputs:
            suboptimal_trajs = pickle.load(inputs)

        with open(value_dataset_pkl_pth, 'rb') as inputs:
            value_dataset = pickle.load(inputs)

    except:
        bc_dataset, sa_classifier, bc_model, suboptimal_trajs, value_dataset = init_datasets_and_models(demonstrations,
                                                                                                        environment,
                                                                                                        imitation_rewarder,
                                                                                                        ensemble_models_save_pth=ensemble_models_save_pth)

        with open(bc_dataset_pkl_pth, 'wb') as output:
            pickle.dump(bc_dataset, output, pickle.HIGHEST_PROTOCOL)

        with open(sa_classifier_pkl_pth, 'wb') as output:
            pickle.dump(sa_classifier, output, pickle.HIGHEST_PROTOCOL)

        with open(bc_model_pkl_pth, 'wb') as output:
            pickle.dump(bc_model, output, pickle.HIGHEST_PROTOCOL)

        with open(suboptimal_trajs_pkl_pth, 'wb') as output:
            pickle.dump(suboptimal_trajs, output, pickle.HIGHEST_PROTOCOL)

        with open(value_dataset_pkl_pth, 'wb') as output:
            pickle.dump(value_dataset, output, pickle.HIGHEST_PROTOCOL)

    # define cutomized td3
    use_acceleration = True

    model = TD3('MlpPolicy', environment, action_noise=action_noise, verbose=1,
                rewarder=imitation_rewarder,
                reward_type='pwil',
                sl_dataset=bc_dataset,
                value_dataset=value_dataset,
                use_acceleration=use_acceleration,
                expert_classifier=sa_classifier,
                sub_Q_estimator=sa_classifier, # value_dataset.sub_q_model,
                opt_Q_estimator=sa_classifier, # value_dataset.opt_q_model)
                bound_type=FLAGS.q_bound_type)

    parsed_trajs = value_dataset.parse_demonstrations(demonstrations)
   
    #model.pretrain_actor_using_demo()
    model.add_expert_trajs_to_buffer(parsed_trajs, value_dataset)
    model.pretrain_critic_using_demo()
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
