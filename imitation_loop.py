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

"""Imitation loop for PWIL."""

import time

import acme
from acme.utils import counting
from acme.utils import loggers
import dm_env
import matplotlib.pyplot as plt


def plot_traj(exp_traj_all, policy_traj_all, rewarder, ep_num):
    plt.figure()
    exp_traj = [t['observation'] for t in exp_traj_all]
    exp_traj = rewarder.scaler.transform(exp_traj)

    policy_traj = [t['observation'] for t in policy_traj_all]
    policy_traj = rewarder.scaler.transform(policy_traj)

    exp_s = exp_traj[:, -1].flatten()
    policy_s = policy_traj[:, -1].flatten()
    plt.plot(exp_s, color='red')
    plt.ylabel('expert')
    plt.plot(policy_s)
    plt.ylabel('policy')
    # plt.show()
    print('save fig')
    plt.savefig('compare_traj_' + str(ep_num) + '.png')

    plt.figure()
    exp_traj = [t['action'] for t in exp_traj_all]
    exp_traj = rewarder.scaler.transform(exp_traj)

    policy_traj = [t['action'] for t in policy_traj_all]
    policy_traj = rewarder.scaler.transform(policy_traj)

    exp_s = exp_traj[:, -1].flatten()
    policy_s = policy_traj[:, -1].flatten()
    plt.plot(exp_s, color='red')
    plt.ylabel('expert')
    plt.plot(policy_s)
    plt.ylabel('policy')
    # plt.show()
    print('save fig')
    plt.savefig('compare_traj_delta_' + str(ep_num) + '.png')
    plt.cla()
    # print(policy_traj)
    #print(policy_traj.shape[0], policy_traj.shape[1])

    return


class TrainEnvironmentLoop(acme.core.Worker):
    """PWIL environment loop.

    This takes `Environment` and `Actor` instances and coordinates their
    interaction. This can be used as:

      loop = TrainEnvironmentLoop(environment, actor, rewarder)
      loop.run(num_steps)

    The `Rewarder` overwrites the timestep from the environment to define
    a custom reward.

    The runner stores episode rewards and a series of statistics in the provided
    `Logger`.
    """

    def __init__(
        self,
        environment,
        actor,
        rewarder,
        high_actor=None,
        counter=None,
        logger=None
    ):
        self._environment = environment
        self._actor = actor
        self._high_actor = high_actor
        self._rewarder = rewarder
        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.make_default_logger()

    def run(self, num_steps):
        """Perform the run loop.

        Args:
          num_steps: number of steps to run the loop for.
        """
        current_steps = 0
        current_ep = 0
        while current_steps < num_steps:

            # Reset any counts and start the environment.
            start_time = time.time()
            self._rewarder.reset()

            episode_steps = 0
            episode_return = 0
            episode_imitation_return = 0
            timestep = self._environment.reset()
            current_ep += 1

            self._actor.observe_first(timestep)

            # Run an episode.
            tmp_traj = []
            while not timestep.last():
                action = self._actor.select_action(timestep.observation)
                # print(action)
                obs_act = {'observation': timestep.observation, 'action': action}
                imitation_reward, __ = self._rewarder.compute_reward(obs_act)
                timestep = self._environment.step(action)
                # print(timestep)
                imitation_timestep = dm_env.TimeStep(step_type=timestep.step_type,
                                                     reward=imitation_reward,
                                                     discount=timestep.discount,
                                                     observation=timestep.observation)
                #print("\n\n\n", imitation_timestep)

                self._actor.observe(action, next_timestep=imitation_timestep)
                self._actor.update()
                print('reward:', imitation_reward)
                try:
                    self._environment.render(12)
                except:
                    continue

                # Book-keeping.
                episode_steps += 1
                episode_return += timestep.reward
                episode_imitation_return += imitation_reward

                obs_act['action'] = obs_act['action'] * self._environment.action_space.high
                tmp_traj.append(obs_act)

            if current_ep % 2 == 0 or current_ep == 1:
                exp_traj = self._rewarder.all_demonstrations[0]
                plot_traj(exp_traj, tmp_traj, self._rewarder, current_ep)

            # Collect the results and combine with counts.
            counts = self._counter.increment(episodes=1, steps=episode_steps)
            steps_per_second = episode_steps / (time.time() - start_time)
            w2_dist = self._rewarder.compute_w2_dist_to_expert(tmp_traj)
            result = {
                'episode_length': episode_steps,
                'episode_return': episode_return,
                'episode_return_imitation': episode_imitation_return,
                'steps_per_second': steps_per_second,
                'w2_dist': w2_dist
            }
            print(current_ep, result, w2_dist)
            result.update(counts)

            self._logger.write(result)
            current_steps += episode_steps

    def hierarchical_run(self, num_steps):
        """Perform the run loop.

        Args:
          num_steps: number of steps to run the loop for.
        """
        current_steps = 0
        current_ep = 0
        while current_steps < num_steps:

            # Reset any counts and start the environment.
            start_time = time.time()
            self._rewarder.reset()

            episode_steps = 0
            episode_return = 0
            episode_imitation_return = 0
            timestep = self._environment.reset()
            current_ep += 1

            self._high_actor.observe_first(timestep)

            # Run an episode.
            while not timestep.last():
                delta_obs = self._high_actor.select_action(timestep.observation)
                action = self._actor.predict(timestep.observation, delta_obs)
                obs_act = {'observation': timestep.observation, 'action': action}
                imitation_reward, __ = self._rewarder.compute_reward(obs_act)
                timestep = self._environment.step(action)
                imitation_timestep = dm_env.TimeStep(step_type=timestep.step_type,
                                                     reward=imitation_reward,
                                                     discount=timestep.discount,
                                                     observation=timestep.observation)
               #print("\n\n\n", imitation_timestep)

                self._actor.observe(action, next_timestep=imitation_timestep)
                self._actor.update()
                # self._environment.render()

                # Book-keeping.
                episode_steps += 1
                episode_return += timestep.reward
                episode_imitation_return += imitation_reward

            # Collect the results and combine with counts.
            counts = self._counter.increment(episodes=1, steps=episode_steps)
            steps_per_second = episode_steps / (time.time() - start_time)
            result = {
                'episode_length': episode_steps,
                'episode_return': episode_return,
                'episode_return_imitation': episode_imitation_return,
                'steps_per_second': steps_per_second,
            }
            print(current_ep, result)
            result.update(counts)

            self._logger.write(result)
            current_steps += episode_steps


class EvalEnvironmentLoop(acme.core.Worker):
    """PWIL evaluation environment loop.

    This takes `Environment` and `Actor` instances and coordinates their
    interaction. This can be used as:

      loop = EvalEnvironmentLoop(environment, actor, rewarder)
      loop.run(num_episodes)

    The `Rewarder` overwrites the timestep from the environment to define
    a custom reward. The evaluation environment loop does not update the agent,
    and computes the wasserstein distance with expert demonstrations.

    The runner stores episode rewards and a series of statistics in the provided
    `Logger`.
    """

    def __init__(
        self,
        environment,
        actor,
        rewarder,
        counter=None,
        logger=None
    ):
        self._environment = environment
        self._actor = actor
        self._rewarder = rewarder
        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.make_default_logger()

    def run(self, num_episodes):
        """Perform the run loop.

        Args:
          num_episodes: number of episodes to run the loop for.
        """
        for _ in range(num_episodes):
            # Reset any counts and start the environment.
            start_time = time.time()
            self._rewarder.reset()

            episode_steps = 0
            episode_return = 0
            episode_imitation_return = 0
            timestep = self._environment.reset()

            # Run an episode.
            trajectory = []
            while not timestep.last():
                action = self._actor.select_action(timestep.observation)
                obs_act = {'observation': timestep.observation, 'action': action}
                trajectory.append(obs_act)
                imitation_reward, __ = self._rewarder.compute_reward(obs_act)

                timestep = self._environment.step(action)

                # Book-keeping.
                episode_steps += 1
                episode_return += timestep.reward
                episode_imitation_return += imitation_reward

            counts = self._counter.increment(episodes=1, steps=episode_steps)
            w2_dist = self._rewarder.compute_w2_dist_to_expert(trajectory)

            # Collect the results and combine with counts.
            steps_per_second = episode_steps / (time.time() - start_time)
            result = {
                'episode_length': episode_steps,
                'episode_return': episode_return,
                'episode_wasserstein_distance': w2_dist,
                'episode_return_imitation': episode_imitation_return,
                'steps_per_second': steps_per_second,
            }
            result.update(counts)

            self._logger.write(result)
