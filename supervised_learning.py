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

"""An implementation of GAIL with WGAN discriminator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf
#from tensorflow.contrib.eager.python import tfe as contrib_eager_python_tfe


class SLPolicy(tf.keras.Model):
    """Implementation of a discriminator network."""

    def __init__(self, input_dim, action_dim):
        """Initializes a policy network.

        Args:
          input_dim: size of the input space
          action_dim: size of the action space
        """
        super(SLPolicy, self).__init__()

        self.main = tf.keras.Sequential([
            tf.keras.layers.Dense(
                units=400,
                activation='relu',
                kernel_initializer=tf.keras.initializers.orthogonal(),
                input_shape=(input_dim,)),
            tf.keras.layers.Dense(
                units=300,
                activation='relu',
                kernel_initializer=tf.keras.initializers.orthogonal()),
            tf.keras.layers.Dense(
                units=action_dim,
                activation=None,  # 'tanh',
                kernel_initializer=tf.keras.initializers.orthogonal(0.01))
        ])

    def call(self, inputs):
        """Performs a forward pass given the inputs.

        Args:
          inputs: a batch of observations (tfe.Variable).

        Returns:
          Actions produced by a policy.
        """
        # print(inputs.shape)
        return self.main(inputs)


class SL(object):
    """Implementation of GAIL (https://arxiv.org/abs/1606.03476).

    Instead of the original GAN, it uses WGAN (https://arxiv.org/pdf/1704.00028).
    """

    def __init__(self, input_dim, output_dim, lambd=10.0,):
        """Initializes actor, critic, target networks and optimizers.

        Args:
           input_dim: size of the observation space.
           subsampling_rate: subsampling rate that was used for expert trajectories.
           lambd: gradient penalty coefficient for wgan.
           gail_loss: gail loss to use.
        """
        self.input_dim = input_dim
        self.lambd = lambd
        with tf.variable_scope('inverse_inference'):
            self.inverse_ref_step = tf.Variable(
                0, dtype=tf.int64, name='step')
            self.inverse_inference = SLPolicy(input_dim, output_dim)
            self.inverse_optimizer = tf.train.AdamOptimizer()
            self.inverse_optimizer._create_slots(self.inverse_inference.variables)  # pylint: disable=protected-access

    def update_forward_pred(self, batch, original_data=True):
        """Updates the WGAN potential function or GAN discriminator.

        Args:
           batch: A batch from training policy.
           expert_batch: A batch from the expert.
        """
        obs = tf.Variable(
            np.stack(batch.obs).astype('float32'))
        next_obs = tf.Variable(
            np.stack(batch.next_obs).astype('float32'))
        action = tf.Variable(
            np.stack(batch.action).astype('float32'))

        # not using the absorbing state, the last element in the state vector
        #inputs = tf.concat([obs[:, :-1], next_obs[:, :-1]], -1)
        inputs = obs

        with tf.GradientTape() as tape:
            a_pred = self.inverse_inference(inputs)
            mse_loss = tf.losses.mean_squared_error(a_pred, action)
        print(mse_loss)

        grads = tape.gradient(mse_loss, self.inverse_inference.variables)
        self.inverse_optimizer.apply_gradients(
            zip(grads, self.inverse_inference.variables), global_step=self.inverse_ref_step)

    def update_inverse_pred(self, batch, original_data=True):
        """Updates the WGAN potential function or GAN discriminator.

        Args:
           batch: A batch from training policy.
           expert_batch: A batch from the expert.
        """
        obs = contrib_eager_python_tfe.Variable(
            np.stack(batch.obs).astype('float32'))
        next_obs = contrib_eager_python_tfe.Variable(
            np.stack(batch.next_obs).astype('float32'))
        action = contrib_eager_python_tfe.Variable(
            np.stack(batch.action).astype('float32'))

        # not using the absorbing state, the last element in the state vector
        inputs = tf.concat([obs[:, :-1], next_obs[:, :-1]], -1)
        #inputs = obs[:, :-1]

        with tf.GradientTape() as tape:
            a_pred = self.inverse_inference(inputs)
            mse_loss = tf.losses.mean_squared_error(a_pred, action)
        print(mse_loss)

        grads = tape.gradient(mse_loss, self.inverse_inference.variables)
        self.inverse_optimizer.apply_gradients(
            zip(grads, self.inverse_inference.variables), global_step=self.inverse_ref_step)

    def predict(self, tfe_obs, delta_obs):
        delta_obs = np.expand_dims(delta_obs, axis=0)
        #print(tfe_obs[:, :-1].shape, delta_obs.shape)
        next_tfe_obs = tfe_obs[:, :-1] + delta_obs
        inputs = tf.concat([tfe_obs[:, :-1], next_tfe_obs], -1)
        return self.inverse_inference(inputs)

    def forward_predict(self, tfe_obs):
        return self.inverse_inference(tfe_obs)

    @property
    def variables(self):
        """Returns all variables including optimizer variables.

        Returns:
          A dictionary of all variables that are defined in the model.
          variables.
        """
        disc_vars = (
            self.inverse_inference.variables + self.inverse_inference_optimizer.variables()
            + [self.inverse_ref_step])

        return disc_vars
