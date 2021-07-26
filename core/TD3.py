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

th.manual_seed(0)
random.seed(0)


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
        sub_Q_estimator=None,
        opt_Q_estimator=None,
        bound_type=None

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
        self.sub_Q_estimator = sub_Q_estimator
        self.opt_Q_estimator = opt_Q_estimator
        self.bound_type = bound_type

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

    def train(self, gradient_steps: int, batch_size: int = 100, update_actor=False, weight_factor=0, use_expert_Q=True) -> None:
        # print('update')
        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses, sl_losses, bc_losses = [], [], [], []
        #"""
        for gradient_step in range(gradient_steps):
            # print('update')

            # Sample replay buffer
            #buffers = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            expert_replay_data = self.expert_replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            constrained_replay_data = self.constrained_replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            state_action_data = self.state_action_buffer.sample(batch_size, env=self._vec_normalize_env)
            #replay_data = buffers[0]
            sl_id, (bc_obs, bc_acts) = random.choice(self.sl_dataset.enums)

            with th.no_grad():
                # calculate the BC loss
                pi_value = self.critic.q1_forward(state_action_data.observations, self.actor(state_action_data.observations))
                exp_value = self.critic.q1_forward(state_action_data.observations, state_action_data.actions)
                bc_filter = (exp_value > pi_value).type(th.float)

                # Select action according to policy and add clipped noise
                # print(replay_data.actions)
                noise = constrained_replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(constrained_replay_data.next_observations) + noise).clamp(-1, 1)

                # Compute the target Q value: min over all critics targets
                targets = th.cat(self.critic_target(constrained_replay_data.next_observations.float(), next_actions.float()), dim=1)
                target, _ = th.min(targets, dim=1, keepdim=True)
                targets_q = constrained_replay_data.rewards + (1 - constrained_replay_data.dones) * self.gamma * target

                # Compute the LfD n-step bootstrap Q value and the action after n-steps
                noise = expert_replay_data.nth_actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                nth_actions = (self.actor_target(expert_replay_data.nth_observations) + noise).clamp(-1, 1)

                targets_expert = th.cat(self.critic_target(expert_replay_data.nth_observations.float(),
                                                           nth_actions.float()), dim=1)
                target_q_expert, _ = th.min(targets_expert, dim=1, keepdim=True)
                target_q_expert = expert_replay_data.nstep_reward + \
                    (1 - expert_replay_data.nth_dones) * expert_replay_data.nstep_gamma * target_q_expert

                # Compute the Lower bound and upper bound of Q when <s,a> is similar to expert
                nth_noise = constrained_replay_data.nth_actions.clone().data.normal_(0, self.target_policy_noise)
                nth_noise = nth_noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                nth_actions = (self.actor_target(constrained_replay_data.nth_observations) + noise).clamp(-1, 1)
                nth_targets_q = th.cat(self.critic_target(constrained_replay_data.nth_observations.float(),
                                                          nth_actions.float()), dim=1)
                nth_targets_q_constrained_, _ = th.min(nth_targets_q, dim=1, keepdim=True)
                
                target_q_real_nstep = constrained_replay_data.nstep_reward + \
                    (1 - constrained_replay_data.nth_dones) * constrained_replay_data.nstep_gamma * nth_targets_q_constrained_
                
                target_q_lower_bound = constrained_replay_data.subopt_values + \
                    (1 - constrained_replay_data.nth_dones) * constrained_replay_data.nstep_gamma * nth_targets_q_constrained_
                
                #target_q_upper_bound = constrained_replay_data.optimal_values + \
                #    (1 - constrained_replay_data.nth_dones) * constrained_replay_data.nstep_gamma * nth_targets_q_constrained_
                
                target_q_upper_bound = constrained_replay_data.optimal_values + \
                    (1 - constrained_replay_data.dones) * self.gamma * target   


                if self.bound_type == 'constrained_lower_upper':
                    constrained_mask_lower = th.logical_not(targets_q < target_q_lower_bound).type(th.float)
                    constrained_mask_upper = th.logical_not(targets_q > target_q_upper_bound).type(th.float)
                    targets_q_constrained_masked = constrained_mask_lower * \
                        targets_q + (1 - constrained_mask_lower) * target_q_real_nstep
                    targets_q_constrained_masked = constrained_mask_upper * \
                        targets_q_constrained_masked + (1 - constrained_mask_upper) * target_q_upper_bound

                elif self.bound_type == 'constrained_lower':
                    constrained_mask_lower = th.logical_not(targets_q < target_q_lower_bound).type(th.float)
                    targets_q_constrained_masked = constrained_mask_lower * \
                        targets_q + (1 - constrained_mask_lower) * target_q_real_nstep

                elif self.bound_type == 'nstep_lower':
                    constrained_mask_lower = th.logical_not(targets_q < target_q_real_nstep).type(th.float)
                    targets_q_constrained_masked = constrained_mask_lower * \
                        targets_q + (1 - constrained_mask_lower) * target_q_real_nstep

                elif self.bound_type == 'nstep_lower_upper':
                    constrained_mask_lower = th.logical_not(targets_q < target_q_real_nstep).type(th.float)
                    constrained_mask_upper = th.logical_not(targets_q > target_q_upper_bound).type(th.float)
                    targets_q_constrained_masked = constrained_mask_lower * \
                        targets_q + (1 - constrained_mask_lower) * target_q_real_nstep
                    targets_q_constrained_masked = constrained_mask_upper * \
                        targets_q_constrained_masked + (1 - constrained_mask_upper) * target_q_upper_bound

                elif self.bound_type == 'upper':
                    constrained_mask_upper = th.logical_not(targets_q > target_q_upper_bound).type(th.float)
                    targets_q_constrained_masked = constrained_mask_upper * \
                        targets_q + (1 - constrained_mask_upper) * target_q_upper_bound

                elif self.bound_type == 'optimal_tight':
                    optimal_upper_q = (targets_q - constrained_replay_data.nstep_reward) / constrained_replay_data.nstep_gamma
                    optimal_lower_q = target_q_real_nstep

                elif self.bound_type == 'constrained_optimal_tight':
                    """ bad version
                    optimal_upper_q = (targets_q - constrained_replay_data.subopt_values) / constrained_replay_data.nstep_gamma
                    optimal_lower_q = target_q_real_nstep
                    """
                    noise = constrained_replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                    noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                    current_actions = (self.actor_target(constrained_replay_data.observations) + noise).clamp(-1, 1)

                    # Compute the target Q value: min over all critics targets
                    upper_targets = th.cat(self.critic_target(constrained_replay_data.observations.float(), current_actions.float()), dim=1)
                    upper_targets_q, _ = th.min(upper_targets, dim=1, keepdim=True)

                    # bound for next obs and next act
                    optimal_upper_q = (upper_targets_q - constrained_replay_data.optimal_values) / self.gamma
                    optimal_next_q = (targets_q - constrained_replay_data.rewards) / self.gamma      
                    optimal_lower_q = target_q_real_nstep

                else:
                    targets_q_constrained_masked = targets_q

            # Get current Q estimates for each critic network
            current_q_estimates = self.critic(constrained_replay_data.observations.float(), constrained_replay_data.actions.float())
            # Compute critic loss
            critic_loss_origin = sum([F.mse_loss(current_q, targets_q) for current_q in current_q_estimates])

            expert_current_q_estimates = self.critic(expert_replay_data.observations.float(), expert_replay_data.actions.float())
            critic_loss_expert = sum([F.mse_loss(current_q, target_q_expert) for current_q in expert_current_q_estimates])


            """ first version of lower-bound loss
            zero_tensors = th.zeros(constrained_current_q_estimates1.shape)
            lower_bound_filtered_q_dis1, _ = th.max(th.cat(((target_q_real_nstep - constrained_current_q_estimates1), zero_tensors), dim=1),dim=1, keepdim=True)
            lower_bound_filtered_q_dis2, _ = th.max(th.cat(((target_q_real_nstep - constrained_current_q_estimates2), zero_tensors), dim=1),dim=1, keepdim=True)
            critic_loss_low_constrained1 = th.mean(lower_bound_filtered_q_dis1**2)
            critic_loss_low_constrained2 = th.mean(lower_bound_filtered_q_dis2**2)
            critic_loss_low_constrained = sum([critic_loss_low_constrained1, critic_loss_low_constrained2])
            
            upper_bound_filtered_q_dis1, _ = th.max(th.cat(((constrained_current_q_estimates1 - target_q_upper_bound), zero_tensors), dim=1),dim=1, keepdim=True)
            upper_bound_filtered_q_dis2, _ = th.max(th.cat(((constrained_current_q_estimates2 - target_q_upper_bound), zero_tensors), dim=1),dim=1, keepdim=True)
            critic_loss_upper_constrained1 = th.mean(upper_bound_filtered_q_dis1**2)
            critic_loss_upper_constrained2 = th.mean(upper_bound_filtered_q_dis2**2)
            critic_loss_upper_constrained = sum([critic_loss_upper_constrained1, critic_loss_upper_constrained2])
            
            """
            critic_loss_nstep = sum([F.mse_loss(current_q, target_q_real_nstep) for current_q in current_q_estimates])
            # second version of lower-bound loss
            """
            critic_loss_constrained1 = F.mse_loss(constrained_current_q_estimates1*constrained_mask, targets_q_constrained*constrained_mask)
            critic_loss_constrained2 = F.mse_loss(constrained_current_q_estimates2*constrained_mask, targets_q_constrained*constrained_mask)
            critic_loss_constrained_target = sum([critic_loss_constrained1, critic_loss_constrained2])
            """
           
            critic_loss_lower = critic_loss_nstep
            critic_loss_upper = sum([F.mse_loss(current_q, target_q_upper_bound) for current_q in current_q_estimates])
            #critic_loss_lower_upper = critic_loss_lower
            critic_loss_lower_upper = critic_loss_lower + critic_loss_upper 

            #"""
            #print(target_q_real_nstep < target_q_upper_bound)

            # critic_loss =  1*critic_loss_origin + 0*expert_weight*critic_loss_expert
            # #0.2*(critic_loss_low_constrained + critic_loss_upper_constrained)
            if self.bound_type == 'none' or self.bound_type is None or self.bound_type == 'None':
                critic_loss = critic_loss_origin  # 0.2*(critic_loss_low_constrained + critic_loss_upper_constrained)
            elif self.bound_type == 'DDPGfD' or self.bound_type == 'nstep_only':
                # print(self.bound_type)
                critic_loss = 1 * critic_loss_origin + 0.3 * critic_loss_nstep
            
            elif self.bound_type == 'optimal_tight' :
                nth_q_estimates = self.critic(constrained_replay_data.nth_observations.float(), constrained_replay_data.nth_actions.float())
                optimal_up_loss = sum([F.mse_loss(nth_q, optimal_upper_q) for nth_q in nth_q_estimates])
                optimal_low_loss = sum([F.mse_loss(current_q, optimal_lower_q) for current_q in current_q_estimates])
                optimal_tight_loss = optimal_up_loss + optimal_low_loss
                critic_loss = 1 * critic_loss_origin + 0.3 * optimal_tight_loss

            elif self.bound_type == 'constrained_optimal_tight':
                next_q_estimates = self.critic(constrained_replay_data.next_observations.float(), constrained_replay_data.next_actions.float())
                optimal_up_loss = sum([F.mse_loss(next_q, optimal_upper_q) for next_q in next_q_estimates])
                optimal_next_loss = sum([F.mse_loss(next_q, optimal_next_q) for next_q in next_q_estimates])
                #optimal_tight_loss = optimal_up_loss + optimal_low_loss
                #print('tst')
                critic_loss = 1 * critic_loss_origin + 0.3*critic_loss_nstep + 0.3*(optimal_up_loss + critic_loss_upper)



            else:  # nstep_lower, nstep_lower_upper, constrained_lower, constrained_lower_upper, upper
                # print(self.bound_type,'else')
                # critic_loss_constrained_target = sum([F.mse_loss(current_q, targets_q_constrained_masked) for current_q in current_q_estimates])
                #critic_loss = 1 * critic_loss_origin + expert_weight * critic_loss_expert + 0.3 * critic_loss_constrained_target
                """
                current_q_estimates1, current_q_estimates2 = current_q_estimates[0], current_q_estimates[1]
                
                up_gains_1 = F.relu(current_q_estimates1 - target_q_upper_bound )
                up_gains_2 = F.relu(current_q_estimates2 - target_q_upper_bound )

                q1_loss_up = th.mean(up_gains_1**2)
                q2_loss_up = th.mean(up_gains_2**2)

                q_loss_up = q1_loss_up + q2_loss_up

                low_gains_1 = F.relu(target_q_lower_bound - current_q_estimates1)
                low_gains_2 = F.relu(target_q_lower_bound - current_q_estimates2)

                q1_loss_low = th.mean(low_gains_1**2)
                q2_loss_low = th.mean(low_gains_2**2)

                q_loss_low = q1_loss_low + q2_loss_low

                critic_loss_lower_upper = q_loss_up + q_loss_low
                """
                next_q_estimates = self.critic(constrained_replay_data.next_observations.float(), constrained_replay_data.next_actions.float())
                optimal_up_loss = sum([F.mse_loss(nth_q, optimal_upper_q) for nth_q in nth_q_estimates])
                critic_loss = 1 * critic_loss_origin +  0.3 * critic_loss_lower_upper


            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Delayed policy updates
            if gradient_step % self.policy_delay == 0:
                # Compute actor loss
                actor_loss = -self.critic.q1_forward(constrained_replay_data.observations, self.actor(constrained_replay_data.observations)).mean()

                actor_losses.append(actor_loss.item())
                # compute sl loss

                bc_eval_loss = sum([F.mse_loss(bc_acts.float(), self.actor(bc_obs.float()))])
                bc_losses.append(bc_eval_loss.item())

                self.use_bc = False
                if self.use_bc:
                    filtered_actions = (state_action_data.actions - self.actor(state_action_data.observations)) * bc_filter
                    bc_loss = th.mean(filtered_actions**2)
                    # print(bc_loss)
                    hybrid_loss = 1 * actor_loss + 0.5 * bc_loss
                else:
                    hybrid_loss = actor_loss

                logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
                logger.record("train/actor_critic_loss", np.mean(actor_losses))
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

    def pretrain_actor_using_demo(self, epochs=30):
        sl_dataset = self.sl_dataset
        loss_fn = nn.MSELoss()
        epoch = 0

        train_losses = []
        valid_losses = []

        while epoch < epochs:  # for epoch in range(epochs):
            self.actor.train()

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
