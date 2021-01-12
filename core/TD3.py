from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch as th
import torch.nn as nn
from torch.nn import functional as F

from stable_baselines3.common import logger
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, Schedule, MaybeCallback
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.td3.policies import TD3Policy
from core.off_policy_algorithm import OffPolicyAlgorithm

import random


class TD3(OffPolicyAlgorithm):
    """
    Twin Delayed DDPG (TD3)
    Addressing Function Approximation Error in Actor-Critic Methods.

    Original implementation: https://github.com/sfujim/TD3
    Paper: https://arxiv.org/abs/1802.09477
    Introduction to TD3: https://spinningup.openai.com/en/latest/algorithms/td3.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
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
    :param policy_delay: Policy and target networks will only be updated once every policy_delay steps
        per training steps. The Q values will be updated policy_delay more often (update every training step).
    :param target_policy_noise: Standard deviation of Gaussian noise added to target policy
        (smoothing noise)
    :param target_noise_clip: Limit for absolute value of target policy smoothing noise.
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[TD3Policy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule]=1e-3,
        buffer_size: int = int(1e6),
        learning_starts: int = 100,
        batch_size: int = 100,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int = -1,
        gradient_steps: int = -1,
        n_episodes_rollout: int = 1,
        action_noise: Optional[ActionNoise] = None,
        optimize_memory_usage: bool = False,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Dict[str, Any]=None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str]="cpu",
        _init_setup_model: bool = True,
        rewarder=None,
        reward_type='pwil',  # 'pwil', 'w2_dist',
        sl_dataset=None,
        value_dataset=None,
        use_acceleration=False,
        expert_classifier=None,

    ):

        super(TD3, self).__init__(
            policy,
            env,
            TD3Policy,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            n_episodes_rollout,
            action_noise=action_noise,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
        )

        self.policy_delay = policy_delay
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise
        self.rewarder = rewarder
        self.reward_type = reward_type

        self.sl_dataset = sl_dataset
        self.value_dataset = value_dataset
        self.use_acceleration = use_acceleration
        self.expert_classifier = expert_classifier

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(TD3, self)._setup_model()
        self._create_aliases()

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.actor_target = self.policy.actor_target
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def train(self, gradient_steps: int, batch_size: int = 100, update_actor=False, weight_factor=0) -> None:
        # print('update')
        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses, sl_losses, bc_losses = [], [], [], []
        #"""
        for gradient_step in range(gradient_steps):
            # print('update')

            # Sample replay buffer
            ideal_data = False
            buffers = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            if len(buffers) > 1:
                replay_data, ideal_replay_data, combined_replay_data = buffers[0], buffers[1], buffers[2]
                ideal_data = True
                sl_id, (bc_obs, bc_acts) = random.choice(self.sl_dataset.enums)

            else:
                replay_data = buffers[0]
                sl_id, (bc_obs, bc_acts) = random.choice(self.sl_dataset.enums)

            with th.no_grad():
                # Select action according to policy and add clipped noise
                # print(replay_data.actions)
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                tmp = self.critic_target(replay_data.next_observations.float(), next_actions.float())

                # Compute the target Q value: min over all critics targets
                targets = th.cat(self.critic_target(replay_data.next_observations.float(), next_actions.float()), dim=1)
                target_q, _ = th.min(targets, dim=1, keepdim=True)
                target_q = replay_data.rewards + (1 - replay_data.dones) * self.gamma * target_q

            # Get current Q estimates for each critic network
            current_q_estimates = self.critic(replay_data.observations.float(), replay_data.actions.float())
           # print(replay_data.observations.float().shape)

            # Compute critic loss
            critic_loss = sum([F.mse_loss(current_q, target_q) for current_q in current_q_estimates])
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Delayed policy updates
            if gradient_step % self.policy_delay == 0:
                # Compute actor loss
                actor_loss = -self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations)).mean()

                actor_losses.append(actor_loss.item())
                # compute sl loss
                if ideal_data:
                    sl_loss = sum([F.mse_loss(ideal_replay_data.actions.float(),
                                              self.actor(ideal_replay_data.observations.float()))])
                    sl_losses.append(sl_loss.item())
                    # print(sl_loss)
                    rl_weight = np.clip([np.log10(weight_factor * 5e-1 + 1)], 0, 1)[0]
                    #print(rl_weight, weight_factor, np.log(weight_factor * 2e-3 + 1))
                    # nn.MSELoss(self.actor(bc_obs), bc_acts)
                    bc_loss = sum([F.mse_loss(bc_acts.float(), self.actor(bc_obs.float()))])
                    bc_losses.append(bc_loss.item())
                    hybrid_loss = 0.5 * (rl_weight * actor_loss + (1 - rl_weight) * sl_loss) + 0.5 * bc_loss

                    logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
                    logger.record("train/acto_critic_loss", np.mean(actor_losses))
                    logger.record("train/bc_loss", np.mean(bc_losses))
                    logger.record("train/sl_loss", np.mean(sl_losses))
                    logger.record("train/sl_weight", 1 - rl_weight)
                    logger.record("train/critic_loss", np.mean(critic_losses))
                else:
                    bc_loss = sum([F.mse_loss(bc_acts.float(), self.actor(bc_obs.float()))])
                    bc_losses.append(bc_loss.item())
                    hybrid_loss = 1 * actor_loss + 0. * bc_loss

                    logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
                    logger.record("train/acto_critic_loss", np.mean(actor_losses))
                    logger.record("train/bc_loss", np.mean(bc_losses))
                    logger.record("train/critic_loss", np.mean(critic_losses))

                # Optimize the actor
                if update_actor:
                    self.actor.optimizer.zero_grad()
                    hybrid_loss.backward()
                    self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

        self._n_updates += gradient_steps
        #"""

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback=None,
        log_interval: int=4,
        eval_env: Optional[GymEnv]=None,
        eval_freq: int=-1,
        n_eval_episodes: int=5,
        tb_log_name: str="TD3",
        eval_log_path: Optional[str]=None,
        reset_num_timesteps: bool=True,
        update_model=True,
    ) -> OffPolicyAlgorithm:

        return super(TD3, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
            update_model=True
        )

    def pretrain_critic_using_demo(self, gradient_steps=1000, batch_size: int=100):
        print('pretrain Q func')
        actor_losses, critic_losses = [], []
        for gradient_step in range(gradient_steps):

            # Sample replay buffer
            buffers = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            if len(buffers) > 1:
                original_replay_data, ideal_replay_data, replay_data = buffers[0], buffers[1], buffers[2]
                ideal_data = True
                sl_id, (bc_obs, bc_acts) = random.choice(self.sl_dataset.enums)

            else:
                replay_data = buffers[0]

            with th.no_grad():
                # Select action according to policy and add clipped noise
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                tmp = self.critic_target(replay_data.next_observations.float(), next_actions.float())

                # Compute the target Q value: min over all critics targets
                targets = th.cat(self.critic_target(replay_data.next_observations.float(), next_actions.float()), dim=1)
                target_q, _ = th.min(targets, dim=1, keepdim=True)
                target_q = replay_data.rewards + (1 - replay_data.dones) * self.gamma * target_q

            # Get current Q estimates for each critic network
            current_q_estimates = self.critic(replay_data.observations.float(), replay_data.actions.float())

            # Compute critic loss
            critic_loss = sum([F.mse_loss(current_q, target_q) for current_q in current_q_estimates])
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
        polyak_update(self.critic.parameters(), self.critic_target.parameters(), 1)

    def pretrain_actor_using_demo(self, sl_dataset, epochs=110):
        self.sl_dataset = sl_dataset
        loss_fn = nn.MSELoss()
        epoch = 0

        while epoch < epochs:  # for epoch in range(epochs):
            self.actor.train()

            train_losses = []
            valid_losses = []
            for i, (x, labels) in enumerate(self.sl_dataset.train_loader):
                x = x.float()
                labels = labels.float()

                self.actor.optimizer.zero_grad()
                outputs = self.actor(x)
                loss = loss_fn(outputs, labels)
                loss.backward()
                self.actor.optimizer.step()

                train_losses.append(loss.item())

            print('BC epoch : {}, train loss : {:.4f},'.format(epoch + 1, np.mean(train_losses)))
            epoch += 1

        polyak_update(self.actor.parameters(), self.actor_target.parameters(), 1)

    def _excluded_save_params(self) -> List[str]:
        return super(TD3, self)._excluded_save_params() + ["actor", "critic", "actor_target", "critic_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        return state_dicts, []
