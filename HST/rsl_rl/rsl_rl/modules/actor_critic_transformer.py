# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal

# a BERT-style transformer block
class Transformer_Block(nn.Module):
    def __init__(self, latent_dim, num_head, dropout_rate) -> None:
        super().__init__()
        self.num_head = num_head
        self.latent_dim = latent_dim
        self.ln_1 = nn.LayerNorm(latent_dim)
        self.attn = nn.MultiheadAttention(latent_dim, num_head, dropout=dropout_rate, batch_first=True)
        self.ln_2 = nn.LayerNorm(latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 4 * latent_dim),
            nn.GELU(),
            nn.Linear(4 * latent_dim, latent_dim),
            nn.Dropout(dropout_rate),
        )
    
    def forward(self, x):
        x = self.ln_1(x)
        x = x + self.attn(x, x, x, need_weights=False)[0]
        x = self.ln_2(x)
        x = x + self.mlp(x)
        
        return x

class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, context_len, latent_dim=128, num_head=4, num_layer=4, dropout_rate=0.1) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_len = context_len
        self.latent_dim = latent_dim
        self.num_head = num_head
        self.num_layer = num_layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.Dropout(dropout_rate),
        )
        self.weight_pos_embed = nn.Embedding(context_len, latent_dim)
        self.attention_blocks = nn.Sequential(
            *[Transformer_Block(latent_dim, num_head, dropout_rate) for _ in range(num_layer)],
        )
        self.output_layer = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, output_dim),
        )
    
    def forward(self, x):
        x = self.input_layer(x)
        x = x + self.weight_pos_embed(torch.arange(x.shape[1], device=x.device))
        x = self.attention_blocks(x)

        # take the last token
        x = x[:, -1, :]
        x = self.output_layer(x)

        return x

class ActorCriticTransformer(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        obs_context_len, 
                        init_noise_std=1.0,
                        **kwargs):
        if kwargs:
            print("ActorCriticTransformer.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCriticTransformer, self).__init__()

        # Policy
        self.actor = Transformer(num_actor_obs, num_actions, obs_context_len)
        self.actor.output_layer[1].weight.data *= 0.01 # init last layer to be 100x smaller

        # Value function
        self.critic = Transformer(num_critic_obs, 1, obs_context_len)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

