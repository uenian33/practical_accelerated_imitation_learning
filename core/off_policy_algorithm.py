import io
import pathlib
import time
import warnings
from typing import Any, Dict, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th

from stable_baselines3.common import logger
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl
from stable_baselines3.common.type_aliases import GymEnv, Schedule, MaybeCallback, RolloutReturn
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.vec_env import VecEnv

from core.buffers import IdealReplayBuffer, ValueReplayBuffer, StateActionReplayBuffer

th.manual_seed(0)

class OffPolicyAlgorithm(BaseAlgorithm):
    """
    The base for Off-Policy algorithms (ex: SAC/TD3)

    :param policy: Policy object
    :param env: The environment to learn from
                (if registered in Gym, can be str. Can be None for loading trained models)
    :param policy_base: The base policy used by this method
    :param learning_rate: learning rate for the optimizer,
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Set to `-1` to disable.
    :param gradient_steps: How many gradient steps to do after each rollout
        (see ``train_freq`` and ``n_episodes_rollout``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param n_episodes_rollout: Update the model every ``n_episodes_rollout`` episodes.
        Note that this cannot be used at the same time as ``train_freq``. Set to `-1` to disable.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param policy_kwargs: Additional arguments to be passed to the policy on creation
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param verbose: The verbosity level: 0 none, 1 training information, 2 debug
    :param device: Device on which the code should run.
        By default, it will try to use a Cuda compatible device and fallback to cpu
        if it is not possible.
    :param support_multi_env: Whether the algorithm supports training
        with multiple environments (as in A2C)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param seed: Seed for the pseudo random generators
    :param use_sde: Whether to use State Dependent Exploration (SDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
    :param sde_support: Whether the model support gSDE or not
    :param remove_time_limit_termination: Remove terminations (dones) that are due to time limit.
        See https://github.com/hill-a/stable-baselines/issues/863
    """

    def __init__(
        self,
        policy: Type[BasePolicy],
        env: Union[GymEnv, str],
        policy_base: Type[BasePolicy],
        learning_rate: Union[float, Schedule],
        buffer_size: int = int(1e6),
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int = 1,
        gradient_steps: int = 1,
        n_episodes_rollout: int = -1,
        action_noise: Optional[ActionNoise] = None,
        optimize_memory_usage: bool = False,
        policy_kwargs: Dict[str, Any]=None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[th.device, str]="auto",
        support_multi_env: bool = False,
        create_eval_env: bool = False,
        monitor_wrapper: bool = True,
        seed: Optional[int] = None,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        sde_support: bool = True,
        remove_time_limit_termination: bool = False,
        rewarder=None,
        reward_type=None,
        sl_dataset=None,
        value_dataset=None,
        use_acceleration=False,
        expert_classifier=None,
        sub_Q_estimator=None,
        opt_Q_estimator=None,
        bound_tyoe=None
    ):

        super(OffPolicyAlgorithm, self).__init__(
            policy=policy,
            env=env,
            policy_base=policy_base,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            support_multi_env=support_multi_env,
            create_eval_env=create_eval_env,
            monitor_wrapper=monitor_wrapper,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
        )
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.tau = tau
        self.gamma = gamma
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.n_episodes_rollout = n_episodes_rollout
        self.action_noise = action_noise
        self.optimize_memory_usage = optimize_memory_usage

        # Remove terminations (dones) that are due to time limit
        # see https://github.com/hill-a/stable-baselines/issues/863
        self.remove_time_limit_termination = remove_time_limit_termination

        self.rewarder = rewarder
        self.reward_type = reward_type
        # print(rewarder, reward_type)
        # time.sl()

        if train_freq > 0 and n_episodes_rollout > 0:
            warnings.warn(
                "You passed a positive value for `train_freq` and `n_episodes_rollout`."
                "Please make sure this is intended. "
                "The agent will collect data by stepping in the environment "
                "until both conditions are true: "
                "`number of steps in the env` >= `train_freq` and "
                "`number of episodes` > `n_episodes_rollout`"
            )

        self.actor = None  # type: Optional[th.nn.Module]
        self.replay_buffer = None  # type: Optional[ReplayBuffer]
        self.constrained_replay_buffer = None  # type: Optional[ReplayBuffer]
        self.expert_replay_buffer = None  # type: Optional[ReplayBuffer]
        # Update policy keyword arguments
        if sde_support:
            self.policy_kwargs["use_sde"] = self.use_sde
        # For gSDE only
        self.use_sde_at_warmup = use_sde_at_warmup

        # set boostrapping window size
        self.forward_window = 10
        self.backward_window = 2
        self.normal_window = 3

        self.expert_mean_reward = 300
        self.intermediate_max_r = 3

    def _set_expert_mean_reward(self, expert_mean_reward) -> None:
        self.expert_mean_reward = expert_mean_reward * 0.6

    def _set_nstep_size(self,
                        forward_window=10,
                        backward_window=2,
                        normal_window=3) -> None:

        self.forward_window = forward_window#10
        self.backward_window = backward_window #2
        self.normal_window = normal_window #3

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)
        self.replay_buffer = IdealReplayBuffer(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            self.device,
            optimize_memory_usage=self.optimize_memory_usage,
        )
       
        self.constrained_replay_buffer = ValueReplayBuffer(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            self.device,
            optimize_memory_usage=self.optimize_memory_usage,
        )
        self.expert_replay_buffer = ValueReplayBuffer(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            self.device,
            optimize_memory_usage=self.optimize_memory_usage,
        )

        self.state_action_buffer = StateActionReplayBuffer(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            self.device,
            optimize_memory_usage=self.optimize_memory_usage,
        )
        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

    def save_replay_buffer(self, path: Union[str, pathlib.Path, io.BufferedIOBase]) -> None:
        """
        Save the replay buffer as a pickle file.

        :param path: Path to the file where the replay buffer should be saved.
            if path is a str or pathlib.Path, the path is automatically created if necessary.
        """
        assert self.replay_buffer is not None, "The replay buffer is not defined"
        save_to_pkl(path, self.replay_buffer, self.verbose)

    def load_replay_buffer(self, path: Union[str, pathlib.Path, io.BufferedIOBase]) -> None:
        """
        Load a replay buffer from a pickle file.

        :param path: Path to the pickled replay buffer.
        """
        self.replay_buffer = load_from_pkl(path, self.verbose)
        assert isinstance(self.replay_buffer, IdealReplayBuffer), "The replay buffer must inherit from IdealReplayBuffer class"

    def load_value_replay_buffer(self, buffer):
        assert self.replay_buffer is not None, "The replay buffer is not defined"
        for item in buffer:
            obs, act, next_obs, r, done, opt_Q, sub_Q = item
            self.constrained_replay_buffer.add(obs, act, next_obs, r, done, opt_Q, sub_Q)

    def _setup_learn(
        self,
        total_timesteps: int,
        eval_env: Optional[GymEnv],
        callback: MaybeCallback = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
    ) -> Tuple[int, BaseCallback]:
        """
        cf `BaseAlgorithm`.
        """
        # Prevent continuity issue by truncating trajectory
        # when using memory efficient replay buffer
        # see https://github.com/DLR-RM/stable-baselines3/issues/46
        truncate_last_traj = (
            self.optimize_memory_usage
            and reset_num_timesteps
            and self.replay_buffer is not None
            and (self.replay_buffer.full or self.replay_buffer.pos > 0)
        )

        if truncate_last_traj:
            warnings.warn(
                "The last trajectory in the replay buffer will be truncated, "
                "see https://github.com/DLR-RM/stable-baselines3/issues/46."
                "You should use `reset_num_timesteps=False` or `optimize_memory_usage=False`"
                "to avoid that issue."
            )
            # Go to the previous index
            pos = (self.replay_buffer.pos - 1) % self.replay_buffer.buffer_size
            self.replay_buffer.dones[pos] = True

        return super()._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, log_path, reset_num_timesteps, tb_log_name
        )

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "run",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        update_model=True
    ) -> "OffPolicyAlgorithm":

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )

        callback.on_training_start(locals(), globals())
        #self.env.render()
        while self.num_timesteps < total_timesteps:
            print('collecting rollout for learning')
            if not self.use_acceleration:
                rollout = self.collect_rollouts(
                    self.env,
                    n_episodes=self.n_episodes_rollout,
                    n_steps=self.train_freq,
                    action_noise=self.action_noise,
                    callback=callback,
                    learning_starts=self.learning_starts,
                    replay_buffer=self.replay_buffer,
                    log_interval=log_interval,
                )
            else:
                rollout, trajectories = self.collect_acceleration_rollouts(
                    self.env,
                    n_episodes=self.n_episodes_rollout,
                    n_steps=self.train_freq,
                    action_noise=self.action_noise,
                    callback=callback,
                    learning_starts=self.learning_starts,
                    replay_buffer=self.replay_buffer,
                    log_interval=log_interval,
                )
                # self.value_dataset.create_suboptimal_value_datasets_from_trajectories(trajectories)

            print(rollout)

            if rollout.continue_training is False:
                break

            if self.num_timesteps > 1000 and self.num_timesteps > self.learning_starts and update_model:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = 30  # self.gradient_steps if self.gradient_steps > 0 else rollout.episode_timesteps
                self.train(batch_size=self.batch_size, gradient_steps=gradient_steps,
                           update_actor=True, weight_factor=self.num_timesteps % 1000)

        callback.on_training_end()

        return self

    def sample_trajs(self, n_episodes=10,
                     log_interval: int = 4,
                     callback: MaybeCallback = None,

                     eval_env: Optional[GymEnv] = None,
                     eval_freq: int = -1,
                     n_eval_episodes: int = 5,
                     tb_log_name: str = "run",
                     eval_log_path: Optional[str] = None,
                     reset_num_timesteps: bool = True,):
        print("sample interactions...")

        n_timesteps, callback = self._setup_learn(
            n_episodes * 1000, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )

        callback.on_training_start(locals(), globals())
        num_ep = 0
        if not self.use_acceleration:
            while num_ep < n_episodes:
                rollout = self.collect_rollouts(
                    self.env,
                    n_episodes=n_episodes,
                    n_steps=self.train_freq,
                    action_noise=self.action_noise,
                    callback=callback,
                    learning_starts=self.learning_starts,
                    replay_buffer=self.replay_buffer,
                    log_interval=log_interval,
                )
                num_ep += 1
                print(num_ep)
        else:
            while num_ep < n_episodes:
                rollout, trajectories = self.collect_acceleration_rollouts(
                    self.env,
                    n_episodes=n_episodes,
                    n_steps=self.train_freq,
                    action_noise=self.action_noise,
                    callback=callback,
                    learning_starts=self.learning_starts,
                    replay_buffer=self.replay_buffer,
                    log_interval=log_interval,
                )
                self.value_dataset.create_suboptimal_value_datasets_from_trajectories(trajectories)
                num_ep += 1
                print(num_ep)
        print(rollout)
        callback.on_training_end()

        return

    def train(self, gradient_steps: int, batch_size: int) -> None:
        """
        Sample the replay buffer and do the updates
        (gradient descent and update target networks)
        """
        raise NotImplementedError()

    def _sample_action(
        self, learning_starts: int, action_noise: Optional[ActionNoise] = None, predefined_action=None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            unscaled_action = np.array([self.action_space.sample()])
        else:
            if predefined_action is not None:
                unscaled_action = predefined_action
            else:
                # Note: when using continuous actions,
                # we assume that the policy uses tanh to scale the action
                # We use non-deterministic action in the case of SAC, for TD3, it does not matter
                unscaled_action, _ = self.predict(self._last_obs, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, gym.spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action

    def _dump_logs(self) -> None:
        """
        Write log.
        """
        fps = int(self.num_timesteps / (time.time() - self.start_time))
        logger.record("time/episodes", self._episode_num, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            #print([ep_info["r"] for ep_info in self.ep_info_buffer])
            logger.record("rollout/ep_rew_mean", [ep_info["r"] for ep_info in self.ep_info_buffer][-1])#safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            logger.record("rollout/ep_len_mean", [ep_info["l"] for ep_info in self.ep_info_buffer][-1])#safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        logger.record("time/fps", fps)
        logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
        logger.record("time/total timesteps", self.num_timesteps, exclude="tensorboard")
        if self.use_sde:
            logger.record("train/std", (self.actor.get_std()).mean().item())

        if len(self.ep_success_buffer) > 0:
            logger.record("rollout/success rate", safe_mean(self.ep_success_buffer))
        # Pass the number of timesteps for tensorboard
        logger.dump(step=self.num_timesteps)

    def _on_step(self) -> None:
        """
        Method called after each step in the environment.
        It is meant to trigger DQN target network update
        but can be used for other purposes
        """
        pass

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        n_episodes: int = 1,
        n_steps: int = -1,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        replay_buffer: Optional[ValueReplayBuffer] = None,
        log_interval: Optional[int] = None,
        #
    ) -> RolloutReturn:
        rollout_type = self.reward_type  # self.reward_type  # ['pwil', 'w2_dist', 'RED']
        print(rollout_type)
        env.reset()
        if rollout_type == 'pwil':
            return self.original_collect_rollouts(env, callback, n_episodes, n_steps, action_noise, learning_starts,
                                                  replay_buffer, log_interval)
        elif rollout_type == 'w2_dist':
            return self.collect_raw_wasserstein_rollouts(env, callback, n_episodes, n_steps, action_noise, learning_starts,
                                                         replay_buffer, log_interval)

        elif rollout_type == 'RED':
            return self.collect_RED_rollouts(env, callback, n_episodes, n_steps, action_noise, learning_starts,
                                             replay_buffer, log_interval)

    def original_collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        n_episodes: int = 1,
        n_steps: int = -1,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        replay_buffer: Optional[IdealReplayBuffer] = None,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param n_episodes: Number of episodes to use to collect rollout data
            You can also specify a ``n_steps`` instead
        :param n_steps: Number of steps to use to collect rollout data
            You can also specify a ``n_episodes`` instead.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        episode_rewards, total_timesteps = [], []
        total_steps, total_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert env.num_envs == 1, "OffPolicyAlgorithm only support single environment"

        if self.use_sde:
            self.actor.reset_noise()

        callback.on_rollout_start()
        continue_training = True

        while total_steps < n_steps or total_episodes < n_episodes:
            done = False
            episode_reward, episode_timesteps, original_episode_reward = 0.0, 0, 0.0

            self.rewarder.reset()

            while not done:
                if self.use_sde and self.sde_sample_freq > 0 and total_steps % self.sde_sample_freq == 0:
                    # Sample a new noise matrix
                    self.actor.reset_noise()

                # Select action randomly or according to policy
                action, buffer_action = self._sample_action(learning_starts, action_noise)

                # Rescale and perform action
                new_obs, origin_reward, done, infos = env.step(action)
                if self._vec_normalize_env is not None:
                    origin_new_obs = self._vec_normalize_env.get_original_obs()
                    obs_act = {'observation': origin_new_obs[0], 'action': buffer_action[0]}
                    print(origin_new_obs, new_obs)
                else:
                    obs_act = {'observation': self._last_obs[0], 'action': buffer_action[0]}
                imitation_reward = self.rewarder.compute_reward(obs_act)
                reward = imitation_reward
                #env.render()

                self.num_timesteps += 1
                episode_timesteps += 1
                total_steps += 1

                # Give access to local variables
                callback.update_locals(locals())
                # Only stop training if return value is False, not when it is None.
                if callback.on_step() is False:
                    return RolloutReturn(0.0, total_steps, total_episodes, continue_training=False)

                episode_reward += reward
                original_episode_reward += origin_reward

                # Retrieve reward and episode length if using Monitor wrapper
                self._update_info_buffer(infos, done)

                # Store data in replay buffer
                if replay_buffer is not None:
                    # Store only the unnormalized version
                    if self._vec_normalize_env is not None:
                        new_obs_ = self._vec_normalize_env.get_original_obs()
                        reward_ = self._vec_normalize_env.get_original_reward()
                    else:
                        # Avoid changing the original ones
                        self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

                    replay_buffer.add(self._last_original_obs, new_obs_, buffer_action, reward_,
                                      done, None, None, None, None, None, use_ideal=False)

                self._last_obs = new_obs
                # Save the unnormalized observation
                if self._vec_normalize_env is not None:
                    self._last_original_obs = new_obs_

                self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

                # For DQN, check if the target network should be updated
                # and update the exploration schedule
                # For SAC/TD3, the update is done as the same time as the gradient update
                # see https://github.com/hill-a/stable-baselines/issues/900
                self._on_step()

                if 0 < n_steps <= total_steps:
                    break

                if self.num_timesteps > 1000 and self.num_timesteps > self.learning_starts:
                    # If no `gradient_steps` is specified,
                    # do as many gradients steps as steps performed during the rollout
                    self.train(batch_size=self.batch_size, gradient_steps=1,
                               update_actor=True, weight_factor=self.num_timesteps % 1000)

            if done:
                total_episodes += 1
                self._episode_num += 1
                episode_rewards.append(episode_reward)
                total_timesteps.append(episode_timesteps)

                if action_noise is not None:
                    action_noise.reset()

                # Log training infos
                if log_interval is not None and self._episode_num % log_interval == 0:
                    self._dump_logs()
                print(episode_rewards, original_episode_reward, total_steps, total_episodes, continue_training)

        mean_reward = np.mean(episode_rewards) if total_episodes > 0 else 0.0

        callback.on_rollout_end()

        return RolloutReturn(mean_reward, total_steps, total_episodes, continue_training)

    def original_collect_rollouts_(
        self,
        env: VecEnv,
        callback: BaseCallback,
        n_episodes: int = 1,
        n_steps: int = -1,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        replay_buffer: Optional[IdealReplayBuffer] = None,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param n_episodes: Number of episodes to use to collect rollout data
            You can also specify a ``n_steps`` instead
        :param n_steps: Number of steps to use to collect rollout data
            You can also specify a ``n_episodes`` instead.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        episode_rewards, total_timesteps = [], []
        total_steps, total_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert env.num_envs == 1, "OffPolicyAlgorithm only support single environment"

        if self.use_sde:
            self.actor.reset_noise()

        callback.on_rollout_start()
        continue_training = True

        while total_steps < n_steps or total_episodes < n_episodes:
            done = False
            episode_reward, episode_timesteps = 0.0, 0
            while not done:

                if self.use_sde and self.sde_sample_freq > 0 and total_steps % self.sde_sample_freq == 0:
                    # Sample a new noise matrix
                    self.actor.reset_noise()

                # Select action randomly or according to policy
                action, buffer_action = self._sample_action(learning_starts, action_noise)
                # print(action)

                # Rescale and perform action
                try:
                    use_ideal = True
                    new_obs, reward, done_, infos = env.step(action)

                    if total_episodes % 10 == 0:
                        env.render()
                    # print(infos[0]['ideal_tuple'][0])
                    ideal_pre_obs = infos[0]['ideal_tuple'][0]
                    ideal_action = infos[0]['ideal_tuple'][1]
                    ideal_next_obs = infos[0]['ideal_tuple'][2]
                    ideal_reward = infos[0]['ideal_tuple'][3]
                    ideal_done = infos[0]['ideal_tuple'][4]
                    # print(new_obs - pre_obs)
                    ideal_action, ideal_buffer_action = self._sample_action(0, predefined_action=ideal_action, action_noise=None)

                except:
                    use_ideal = False
                    new_obs, reward, done, infos = env.step(action)

                try:
                    done = done_[0]
                except:
                    done = done_
                self.num_timesteps += 1
                episode_timesteps += 1
                total_steps += 1

                # Give access to local variables
                callback.update_locals(locals())
                # Only stop training if return value is False, not when it is None.
                if callback.on_step() is False:
                    return RolloutReturn(0.0, total_steps, total_episodes, continue_training=False)

                episode_reward += reward

                # Retrieve reward and episode length if using Monitor wrapper
                self._update_info_buffer(infos, done)

                # Store data in replay buffer
                if replay_buffer is not None:
                    # Store only the unnormalized version
                    if self._vec_normalize_env is not None:
                        print('not none')
                        new_obs_ = self._vec_normalize_env.get_original_obs()
                        reward_ = self._vec_normalize_env.get_original_reward()
                    else:
                        # print('no vec env')
                        # Avoid changing the original ones
                        self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

                    if use_ideal:
                        replay_buffer.add(self._last_original_obs, new_obs_, buffer_action, reward_, done,
                                          ideal_obs=ideal_pre_obs, ideal_next_obs=ideal_next_obs, ideal_action=ideal_buffer_action,
                                          ideal_reward=ideal_reward, ideal_done=done)
                    else:
                        replay_buffer.add(self._last_original_obs, new_obs_, buffer_action, reward_,
                                          done, None, None, None, None, None, use_ideal=False)

                self._last_obs = new_obs
                # Save the unnormalized observation
                if self._vec_normalize_env is not None:
                    self._last_original_obs = new_obs_

                self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

                # For DQN, check if the target network should be updated
                # and update the exploration schedule
                # For SAC/TD3, the update is done as the same time as the gradient update
                # see https://github.com/hill-a/stable-baselines/issues/900
                self._on_step()
                if 0 < n_steps <= total_steps:
                    break

                if self.num_timesteps > 1000 and self.num_timesteps > self.learning_starts:
                    # If no `gradient_steps` is specified,
                    # do as many gradients steps as steps performed during the rollout
                    self.train(batch_size=self.batch_size, gradient_steps=1,
                               update_actor=True, weight_factor=self.num_timesteps % 1000)

                # if 0 < n_steps <= total_steps:
                #    break

            if done:

                total_episodes += 1
                self._episode_num += 1
                episode_rewards.append(episode_reward)
                total_timesteps.append(episode_timesteps)

                print(total_episodes)
                if action_noise is not None:
                    action_noise.reset()

                # Log training infos
                if log_interval is not None and self._episode_num % log_interval == 0:
                    self._dump_logs()

        mean_reward = np.mean(episode_rewards) if total_episodes > 0 else 0.0

        callback.on_rollout_end()

        return RolloutReturn(mean_reward, total_steps, total_episodes, continue_training)

    def collect_raw_wasserstein_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        n_episodes: int = 1,
        n_steps: int = -1,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        replay_buffer: Optional[IdealReplayBuffer] = None,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param n_episodes: Number of episodes to use to collect rollout data
            You can also specify a ``n_steps`` instead
        :param n_steps: Number of steps to use to collect rollout data
            You can also specify a ``n_episodes`` instead.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        episode_rewards, total_timesteps = [], []
        total_steps, total_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert env.num_envs == 1, "OffPolicyAlgorithm only support single environment"

        if self.use_sde:
            self.actor.reset_noise()

        callback.on_rollout_start()
        continue_training = True

        while total_steps < n_steps and total_episodes < n_episodes:
            done = False
            episode_reward, episode_timesteps = 0.0, 0
            wdistance_trajectory = []
            trajectory = []

            while not done:

                if self.use_sde and self.sde_sample_freq > 0 and total_steps % self.sde_sample_freq == 0:
                    # Sample a new noise matrix
                    self.actor.reset_noise()

                # Select action randomly or according to policy
                action, buffer_action = self._sample_action(learning_starts, action_noise)
                # print(action)

                # Rescale and perform action
                try:
                    use_ideal = True
                    new_obs, reward, done, infos = env.step(action)
                    if total_episodes % 10 == 0:
                        env.render()
                    # print(infos[0]['ideal_tuple'][0])
                    ideal_pre_obs = infos[0]['ideal_tuple'][0]
                    ideal_action = infos[0]['ideal_tuple'][1]
                    ideal_next_obs = infos[0]['ideal_tuple'][2]
                    ideal_reward = infos[0]['ideal_tuple'][3]
                    ideal_done = infos[0]['ideal_tuple'][4]
                    # print(new_obs - pre_obs)
                    ideal_action, ideal_buffer_action = self._sample_action(0, predefined_action=ideal_action, action_noise=None)

                    # print(self._last_original_obs[0])
                except:
                    use_ideal = False
                    new_obs, reward, done, infos = env.step(action)

                # print(wdistance_trajectory)
                # Store only the unnormalized version
                if self._vec_normalize_env is not None:
                    print('not none')
                    new_obs_ = self._vec_normalize_env.get_original_obs()
                    reward_ = self._vec_normalize_env.get_original_reward()
                else:
                    # print('no vec env')
                    # Avoid changing the original ones
                    self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

                wdistance_trajectory.append({'observation': self._last_original_obs[
                                            0], 'action':  (new_obs_ - self._last_original_obs[0])[0]})

                trajectory.append([self._last_original_obs, new_obs_, buffer_action, done,
                                   ideal_pre_obs, ideal_next_obs, ideal_buffer_action, done, ])

                self.num_timesteps += 1
                episode_timesteps += 1
                total_steps += 1

                # Give access to local variables
                callback.update_locals(locals())
                # Only stop training if return value is False, not when it is None.
                if callback.on_step() is False:
                    return RolloutReturn(0.0, total_steps, total_episodes, continue_training=False)

                # Retrieve reward and episode length if using Monitor wrapper
                self._update_info_buffer(infos, done)

                self._last_obs = new_obs
                # Save the unnormalized observation
                if self._vec_normalize_env is not None:
                    self._last_original_obs = new_obs_

                self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

                # For DQN, check if the target network should be updated
                # and update the exploration schedule
                # For SAC/TD3, the update is done as the same time as the gradient update
                # see https://github.com/hill-a/stable-baselines/issues/900
                self._on_step()

                if 0 < n_steps <= total_steps:
                    break

            if done:
                # Store data in replay buffer
                if replay_buffer is not None:
                    traj_len = len(wdistance_trajectory)
                    w2_dist = self.rewarder.compute_w2_dist_to_expert(wdistance_trajectory)
                    episode_reward = np.exp(-1.4 * w2_dist + 7)  # / len(wdistance_trajectory)
                    step_reward = w2_dist / traj_len

                    for i in range(traj_len):
                        if use_ideal:
                            replay_buffer.add(trajectory[i][0], trajectory[i][1], trajectory[i][2], step_reward, trajectory[i][3],
                                              trajectory[i][4], trajectory[i][5], trajectory[i][6], 0, trajectory[i][7],)
                        else:
                            replay_buffer.add(trajectory[i][0], trajectory[i][1], trajectory[i][2], step_reward, trajectory[i][3],
                                              None, None, None, None, None, use_ideal=False)

                total_episodes += 1
                self._episode_num += 1
                episode_rewards.append(episode_reward)
                total_timesteps.append(episode_timesteps)

                if action_noise is not None:
                    action_noise.reset()

                # Log training infos
                if log_interval is not None and self._episode_num % log_interval == 0:
                    self._dump_logs()

        mean_reward = np.mean(episode_rewards) if total_episodes > 0 else 0.0

        callback.on_rollout_end()

        return RolloutReturn(mean_reward, total_steps, total_episodes, continue_training)

    def collect_RED_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        n_episodes: int = 1,
        n_steps: int = -1,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        replay_buffer: Optional[IdealReplayBuffer] = None,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param n_episodes: Number of episodes to use to collect rollout data
            You can also specify a ``n_steps`` instead
        :param n_steps: Number of steps to use to collect rollout data
            You can also specify a ``n_episodes`` instead.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        episode_rewards, total_timesteps = [], []
        total_steps, total_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert env.num_envs == 1, "OffPolicyAlgorithm only support single environment"

        if self.use_sde:
            self.actor.reset_noise()

        callback.on_rollout_start()
        continue_training = True

        while total_steps < env.max_episode_steps:
            print(total_steps, env.max_episode_steps)
            done = False
            episode_reward, episode_timesteps = 0.0, 0

            while not done:

                if self.use_sde and self.sde_sample_freq > 0 and total_steps % self.sde_sample_freq == 0:
                    # Sample a new noise matrix
                    self.actor.reset_noise()

                # Select action randomly or according to policy
                action, buffer_action = self._sample_action(learning_starts, action_noise)
                # print(action)

                # Rescale and perform action
                try:
                    use_ideal = True
                    new_obs, __, done, infos = env.step(action)
                    if total_episodes % 10 == 1:
                        env.render()
                    # print(infos[0]['ideal_tuple'][0])
                    ideal_pre_obs = infos[0]['ideal_tuple'][0]
                    ideal_action = infos[0]['ideal_tuple'][1]
                    ideal_next_obs = infos[0]['ideal_tuple'][2]
                    ideal_reward = infos[0]['ideal_tuple'][3]
                    ideal_done = infos[0]['ideal_tuple'][4]
                    # print(new_obs - pre_obs)
                    ideal_action, ideal_buffer_action = self._sample_action(0, predefined_action=ideal_action, action_noise=None)
                    reward = self.rewarder.get_reward(self._last_obs,
                                                      buffer_action,
                                                      new_obs).detach().numpy()[0]
                except:
                    use_ideal = False
                    new_obs, reward, done, infos = env.step(action)
                    reward = self.rewarder.get_reward(self._last_obs,
                                                      buffer_action,
                                                      new_obs).detach().numpy()[0]

                self.num_timesteps += 1
                episode_timesteps += 1
                total_steps += 1

                # Give access to local variables
                callback.update_locals(locals())
                # Only stop training if return value is False, not when it is None.
                if callback.on_step() is False:
                    return RolloutReturn(0.0, total_steps, total_episodes, continue_training=False)

                episode_reward += reward

                # Retrieve reward and episode length if using Monitor wrapper
                self._update_info_buffer(infos, done)

                # Store data in replay buffer
                if replay_buffer is not None:
                    # Store only the unnormalized version
                    if self._vec_normalize_env is not None:
                        print('not none')
                        new_obs_ = self._vec_normalize_env.get_original_obs()
                        reward_ = self._vec_normalize_env.get_original_reward()
                    else:
                        # print('no vec env')
                        # Avoid changing the original ones
                        self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

                    if use_ideal:
                        replay_buffer.add(self._last_original_obs, new_obs_, buffer_action, reward_, done,
                                          ideal_obs=ideal_pre_obs, ideal_next_obs=ideal_next_obs, ideal_action=ideal_buffer_action,
                                          ideal_reward=ideal_reward, ideal_done=done)
                    else:
                        replay_buffer.add(self._last_original_obs, new_obs_, buffer_action, reward_,
                                          done, None, None, None, None, None, use_ideal=False)

                self._last_obs = new_obs
                # Save the unnormalized observation
                if self._vec_normalize_env is not None:
                    self._last_original_obs = new_obs_

                self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

                # For DQN, check if the target network should be updated
                # and update the exploration schedule
                # For SAC/TD3, the update is done as the same time as the gradient update
                # see https://github.com/hill-a/stable-baselines/issues/900
                self._on_step()

                if 0 < n_steps <= total_steps:
                    break

            print('done', total_steps, n_steps, total_episodes, n_episodes)
            total_episodes += 1
            self._episode_num += 1
            episode_rewards.append(episode_reward)
            total_timesteps.append(episode_timesteps)

            if action_noise is not None:
                action_noise.reset()

            # Log training infos
            if log_interval is not None and self._episode_num % log_interval == 0:
                self._dump_logs()

        mean_reward = np.mean(episode_rewards) if total_episodes > 0 else 0.0

        callback.on_rollout_end()

        return RolloutReturn(mean_reward, total_steps, total_episodes, continue_training)

    def collect_acceleration_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        n_episodes: int = 1,
        n_steps: int = -1,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        replay_buffer: Optional[IdealReplayBuffer] = None,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param n_episodes: Number of episodes to use to collect rollout data
            You can also specify a ``n_steps`` instead
        :param n_steps: Number of steps to use to collect rollout data
            You can also specify a ``n_episodes`` instead.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        episode_rewards, total_timesteps = [], []
        total_steps, total_episodes = 0, 0
        trajectories = []

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert env.num_envs == 1, "OffPolicyAlgorithm only support single environment"

        if self.use_sde:
            self.actor.reset_noise()

        callback.on_rollout_start()
        continue_training = True

        while total_steps < n_steps or total_episodes < n_episodes:
            done = False
            traj = []
            episode_reward, episode_timesteps, original_episode_reward = 0.0, 0, 0.0
            self.rewarder.reset()

            while not done:
                if self.use_sde and self.sde_sample_freq > 0 and total_steps % self.sde_sample_freq == 0:
                    # Sample a new noise matrix
                    self.actor.reset_noise()

                # Select action randomly or according to policy
                action, buffer_action = self._sample_action(learning_starts, action_noise)

                # Rescale and perform action
                new_obs, origin_reward, done, infos = env.step(action)
                obs_act = {'observation': self._last_obs[0], 'action': buffer_action[0]}
                imitation_reward = self.rewarder.compute_reward(obs_act)
                reward = imitation_reward
                #env.render()
                if self.intermediate_max_r < imitation_reward:
                    self.intermediate_max_r = imitation_reward

                self.num_timesteps += 1
                episode_timesteps += 1
                total_steps += 1

                # Give access to local variables
                callback.update_locals(locals())
                # Only stop training if return value is False, not when it is None.
                if callback.on_step() is False:
                    return RolloutReturn(0.0, total_steps, total_episodes, continue_training=False)

                episode_reward += reward
                original_episode_reward += origin_reward

                # Retrieve reward and episode length if using Monitor wrapper
                #print(infos)
                #t()
                self._update_info_buffer(infos, done)

                # Store data in replay buffer
                if replay_buffer is not None:
                    # Store only the unnormalized version
                    if self._vec_normalize_env is not None:
                        new_obs_ = self._vec_normalize_env.get_original_obs()
                    else:
                        # Avoid changing the original ones
                        self._last_original_obs, new_obs_ = self._last_obs, new_obs

                    #print(self.critic_target(th.tensor(self._last_original_obs).float(), th.tensor(buffer_action).float()))
                    replay_buffer.add(self._last_original_obs, new_obs_, buffer_action, imitation_reward,
                                      done, None, None, None, None, None, use_ideal=False)

                # checking if current <s,a> or <s> belongs to expert trajs
                in_expert = self.expert_classifier.predict_class(self._last_original_obs[0], buffer_action[0], new_obs_[0])
                try:
                     obs_input = th.FloatTensor(self._last_original_obs[0])
                     tmp_opt_value = self.sub_Q_estimator.model(obs_input).detach().numpy().flatten()[0]
                     tmp_sub_value = self.opt_Q_estimator.model(obs_input).detach().numpy().flatten()[0]
                except:
                     obs_input = th.FloatTensor(np.hstack([self._last_original_obs[0], buffer_action[0]]))
                     tmp_opt_value = self.sub_Q_estimator.model(obs_input).detach().numpy().flatten()[0]
                     tmp_sub_value = self.opt_Q_estimator.model(obs_input).detach().numpy().flatten()[0]

                """
                self.constrained_replay_buffer.add(self._last_original_obs,
                                                   new_obs_,
                                                   buffer_action,
                                                   reward_,
                                                   done,
                                                   tmp_sub_value,
                                                   tmp_opt_value,
                                                   tmp_sub_value,
                                                   tmp_opt_value,
                                                   1e3)
                """
                if in_expert:
                    #print("in")
                    traj.append([self._last_original_obs,  new_obs_, buffer_action,
                                 imitation_reward, done, in_expert, tmp_sub_value, tmp_opt_value])
                else:
                    traj.append([self._last_original_obs,  new_obs_, buffer_action,
                                 imitation_reward, done, in_expert, None, None])

                self._last_obs = new_obs
                # Save the unnormalized observation
                if self._vec_normalize_env is not None:
                    self._last_original_obs = new_obs_

                self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

                # For DQN, check if the target network should be updated
                # and update the exploration schedule
                # For SAC/TD3, the update is done as the same time as the gradient update
                # see https://github.com/hill-a/stable-baselines/issues/900
                self._on_step()

                if 0 < n_steps <= total_steps:
                    break
                if episode_reward > 200:
                    use_expert_Q = False
                else:
                    use_expert_Q = True

                if self.num_timesteps > 1000 and self.num_timesteps > self.learning_starts:
                    # If no `gradient_steps` is specified,
                    # do as many gradients steps as steps performed during the rollout
                    self.train(batch_size=self.batch_size, 
                               gradient_steps=1,
                               update_actor=True, 
                               weight_factor=self.num_timesteps % 1000,
                               use_expert_Q=use_expert_Q)

            self.add_tuple_with_nsteps_to_buffer(traj, episode_reward=episode_reward)
            # trajectories.append(traj)
            if done:
                total_episodes += 1
                self._episode_num += 1
                episode_rewards.append(episode_reward)
                total_timesteps.append(episode_timesteps)

                if action_noise is not None:
                    action_noise.reset()

                # Log training infos
                if log_interval is not None and self._episode_num % log_interval == 0:
                    self._dump_logs()
                    if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                        logger.record("rollout/ep_imitation_mean", episode_reward)#safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            
                print(episode_rewards, original_episode_reward, total_steps, total_episodes, continue_training)

        mean_reward = np.mean(episode_rewards) if total_episodes > 0 else 0.0

        callback.on_rollout_end()

        return RolloutReturn(mean_reward, total_steps, total_episodes, continue_training), trajectories

  

    def add_tuple_with_nsteps_to_buffer(
        self,
        traj,
        episode_reward,
        optimal_r: int=5
    ) -> None:

        window = self.forward_window#: int=10
        optimal_window = self.backward_window#: int=2
        normal_window = self.normal_window#: int=3

        skip_ids = []
        traj_len = len(traj)
        for idx in range(traj_len):
            trans = traj[idx]
            in_expert = trans[5]

            prev_nth_discounted_R = 0
            prev_id = np.max([idx - window, 0])
            for istep, prev_sub_trans in enumerate(traj[prev_id:idx]):
                prev_nth_discounted_R += self.gamma**(istep) * prev_sub_trans[3]

            if False:#in_expert and (idx > (window-1)) and (idx < len(traj)-window):
                # first add this expert transition
                sub_Q = trans[-2]
                opt_Q = trans[-1]
                sub_traj = traj[idx : window+idx]
                discount_sum_r = 0
                greedy_sum_r = 0

                for ndx, sub_tranj_trans in enumerate(sub_traj):
                    discount_sum_r += self.gamma**ndx*sub_tranj_trans[3]

                
                # note here the estimated MC-Q is only useful for accelerating the training in the begining
                # continuing using sub_Q, opt_Q will introduce bias in the later stage of training
                # when policy is able to generate good traj, replace upper bound with greedy R + nstep R
                if episode_reward > self.expert_mean_reward:
                    # reset upper bound instaed using pretrained MC-Q value
                    for ndx, sub_tranj_trans in enumerate(sub_traj):
                        if ndx < normal_window:
                            r = optimal_r
                        else:
                            r = sub_tranj_trans[3]
                        greedy_sum_r += self.gamma**ndx*r
                    expert_sub_Q = discount_sum_r
                    expert_opt_Q = greedy_sum_r
                else:
                    expert_sub_Q = opt_Q
                    expert_opt_Q = opt_Q

                #print(discount_sum_r, opt_Q, sub_Q, discount_sum_r)
                self.constrained_replay_buffer.add(trans[0],
                                           trans[1],
                                           trans[2],
                                           trans[3],
                                           trans[4],
                                           expert_sub_Q, # nstep suboptimal Q is the accumulative R
                                           discount_sum_r, # nstep accumulative R
                                           expert_opt_Q, # nstep MC-optimal Q
                                           traj[idx+window][0], # state-action after window_step_th (lower-bound) 
                                           traj[idx+window][2], # state-action after window_step_th (lower-bound) 
                                           traj[idx+window][4], # check if after window_step_th is terminate state
                                           traj[prev_id][0],
                                           traj[prev_id][2],
                                           prev_nth_discounted_R,
                                           self.gamma**window, # discounted gamma after window_step_th
                                           )
                        

                # for the state-action before expert-similar state-action, do backward boostrapping
                prev_id = idx - optimal_window
                prev_sub_traj = traj[ prev_id: idx]
                prev_trans = traj[prev_id]
                prev_discount_sum_r = 0
                prev_optimal_sum_r = 0

                for ndx, prev_sub_tranj_trans in enumerate(prev_sub_traj):
                    prev_discount_sum_r += self.gamma**ndx*prev_sub_tranj_trans[3]

                for ndx, prev_sub_tranj_trans in enumerate(prev_sub_traj):
                    prev_optimal_sum_r += self.gamma**ndx*prev_optimal_sum_r

                self.constrained_replay_buffer.add(prev_trans[0],
                                                   prev_trans[1],
                                                   prev_trans[2],
                                                   prev_trans[3],
                                                   prev_trans[4],
                                                   prev_discount_sum_r + self.gamma**optimal_window * expert_sub_Q, #  nstep sum R + discounted nstep MC-optimal Q
                                                   prev_discount_sum_r + self.gamma**optimal_window * discount_sum_r, # optimal_window+n step discounted accumulative R
                                                   prev_optimal_sum_r + self.gamma**optimal_window * expert_opt_Q, # greedy optimal nstep sum R + discounted nstep MC-optimal Q
                                                   traj[idx+window][0], # state-action after pred_id -> window_step_th (lower-bound) 
                                                   traj[idx+window][2], # state-action after pred_id -> window_step_th (lower-bound) 
                                                   traj[idx+window][4], # check if after window_step_th is terminate state
                                                   traj[prev_id][0],
                                                   traj[prev_id][2],
                                                   prev_nth_discounted_R,
                                                   self.gamma**(window+optimal_window), # discounted gamma after window_step_th
                                                   )

                #self.state_action_buffer.add(trans[0],trans[2])
            else:
                # first add this expert transition
                sub_Q = trans[-2]
                opt_Q = trans[-1]
                tmp_end_idx = np.min((window+idx, traj_len-1)) # in case tmp_window+idx > len(traj)

                horizon_n = tmp_end_idx - idx
                sub_traj = traj[idx : ]
                discount_sum_r = 0
                greedy_sum_r = 0
                for ndx, sub_tranj_trans in enumerate(sub_traj):
                    discount_sum_r += self.gamma**ndx*sub_tranj_trans[3]

                for ndx, sub_tranj_trans in enumerate(sub_traj):
                    greedy_sum_r += self.gamma**ndx*optimal_r

                #print(discount_sum_r, opt_Q, sub_Q, discount_sum_r)
                self.constrained_replay_buffer.add(trans[0],
                                                   trans[1],
                                                   trans[2],
                                                   trans[3],
                                                   trans[4],
                                                   discount_sum_r, # nstep suboptimal Q is the accumulative R
                                                   discount_sum_r, # nstep accumulative R
                                                   greedy_sum_r, # # nstep greedy accumulative R_optimal set as the upper bound
                                                   traj[idx+horizon_n][0],
                                                   traj[idx+horizon_n][2],
                                                   traj[idx+horizon_n][4],
                                                   traj[prev_id][0],
                                                   traj[prev_id][2],
                                                   prev_nth_discounted_R,
                                                   self.gamma**horizon_n
                                                   )

    def add_expert_trajs_to_buffer(self, parsed_trajs, value_dataset,  max_window=10,):  
        window = max_window#: int=10  
        for traj in parsed_trajs:    
            traj = np.array(traj)
            for i in range(len(traj) - max_window):
                prev_obs = traj[i][0]
                act = traj[i][1]
                obs = traj[i][2]
                r = traj[i][3]
                discounted_R = 0
                #for window in range(max_window):
                for idx, sub_trans in enumerate(traj[i:i + window]):
                    discounted_R += self.gamma**(idx) * sub_trans[3]

                prev_nth_discounted_R = 0
                prev_id = np.max([i - window, 0])
                for idx, prev_sub_trans in enumerate(traj[prev_id:i]):
                    prev_nth_discounted_R += self.gamma**(idx) * prev_sub_trans[3]
                
                self.expert_replay_buffer.add(prev_obs,
                                               obs,
                                               act,
                                               r,
                                               False,
                                               discounted_R,
                                               discounted_R,
                                               discounted_R,
                                               traj[i + window][0],
                                               traj[i + window][1],
                                               False,
                                               traj[prev_id][0],
                                               traj[prev_id][1],
                                               prev_nth_discounted_R,
                                               self.gamma**window)

                self.replay_buffer.add(prev_obs,
                                       obs,
                                       act,
                                       r,
                                       False, None, None, None, None, None, use_ideal=False)

                self.constrained_replay_buffer.add(prev_obs,
                                               obs,
                                               act,
                                               r,
                                               False,
                                               discounted_R,
                                               discounted_R,
                                               discounted_R,
                                               traj[i + window][0],
                                               traj[i + window][1],
                                               False,
                                               traj[prev_id][0],
                                               traj[prev_id][1],
                                               prev_nth_discounted_R,
                                               self.gamma**window)

                self.state_action_buffer.add(prev_obs,act)
                #print(discounted_R, r, i)


