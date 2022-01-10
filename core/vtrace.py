# This file taken from
#     https://github.com/deepmind/scalable_agent/blob/
#         cd66d00914d56c8ba2f0615d9cdeefcb169a8d70/vtrace.py
# and modified.

# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions to compute V-trace off-policy actor critic targets.

For details and theory see:

"IMPALA: Scalable Distributed Deep-RL with
Importance Weighted Actor-Learner Architectures"
by Espeholt, Soyer, Munos et al.

See https://arxiv.org/abs/1802.01561 for the full paper.
"""

import collections

import torch
import torch.nn.functional as F
import torch.distributions as tdist

VTraceFromLogitsReturns = collections.namedtuple(
    "VTraceFromLogitsReturns", # 元组名称
    [
        "vs", # 字典的key
        "pg_advantages",
        "log_rhos",
        "behavior_action_log_probs",
        "target_action_log_probs",
    ],
)

# collections.namedtuple(typename, field_names)创建元组，typename是元组名称，field_names是字典的key。
VTraceReturns = collections.namedtuple("VTraceReturns", "vs pg_advantages")

# 计算 policy_logits 和 actions 之间的误差
def action_log_probs(policy_logits, actions):
    """
    policy_logits.shape = (unroll_length, batch_size, num_actions)  torch.Size([10, 24, 6])
    流程：（1）把policy_logits摊平为(unroll_length * batch_size, num_actions)； torch.Size([240, 6])
         （2）对最后一维先后做softmax得到概率分布（概率质量函数）, log运算得到负无穷到0的值  torch.Size([240, 6])
         （3）F.nll_loss(output, label)就是从 output 中拿出 label 对应的那个数，再去掉负号。 shape = (unroll_length, batch_size)  torch.Size([10, 24])
    return torch.Size([10, 24])
    """
    return -F.nll_loss(
        F.log_softmax(torch.flatten(policy_logits, 0, -2), dim=-1),
        torch.flatten(actions),
        reduction="none", # 如果reduction="mean"，则求均值
    ).view_as(actions)

def action_log_probs_continuous(policy_logits, actions):
    """
    policy_logits.shape = (unroll_length, batch_size, 2)
    (1)根据policy_logits得到概率密度函数（正态分布）
    (2)Normal.log_prob(value)是计算value在定义的正态分布中对应的概率的对数，概率是0-1，则对数是负无穷-0.
    (3)计算 negative log likelihood
    """
    unroll_length = policy_logits.shape[0]
    batch_size = policy_logits.shape[1]
    actions = torch.flatten(actions)
    policy_logits_flatten = torch.flatten(policy_logits, 0, -2) # 把policy_logits摊平为(unroll_length * batch_size, 2)
    negative_log_likelihood = torch.zeros(policy_logits_flatten.shape[0], 1, dtype=torch.float32).to("cuda")
    for i in range(policy_logits_flatten.shape[0]):
        prob_dist = tdist.Normal(policy_logits_flatten[i][0], policy_logits_flatten[i][1])
        negative_log_likelihood[i] = prob_dist.log_prob(actions[i])
        # if i == 0:
        #     print(prob_dist.log_prob(actions[i]))
    # print(negative_log_likelihood.shape)
    # print(negative_log_likelihood)
    return negative_log_likelihood.view([unroll_length, batch_size])


def from_logits(
    behavior_policy_logits,
    target_policy_logits,
    actions,
    discounts,
    rewards,
    values,
    bootstrap_value,
    clip_rho_threshold=1.0,
    clip_pg_rho_threshold=1.0,
):
    """V-trace for softmax policies."""

    target_action_log_probs = action_log_probs_continuous(target_policy_logits, actions) # torch.Size([10, 24])
    # print(target_action_log_probs)
    behavior_action_log_probs = action_log_probs_continuous(behavior_policy_logits, actions)
    # print(behavior_action_log_probs)
    log_rhos = target_action_log_probs - behavior_action_log_probs
    vtrace_returns = from_importance_weights(
        log_rhos=log_rhos,
        discounts=discounts,
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_value,
        clip_rho_threshold=clip_rho_threshold,
        clip_pg_rho_threshold=clip_pg_rho_threshold,
    )
    return VTraceFromLogitsReturns(
        log_rhos=log_rhos,
        behavior_action_log_probs=behavior_action_log_probs,
        target_action_log_probs=target_action_log_probs,
        **vtrace_returns._asdict(),
    )


@torch.no_grad()
def from_importance_weights(
    log_rhos,
    discounts,
    rewards,
    values,
    bootstrap_value,
    clip_rho_threshold=1.0,
    clip_pg_rho_threshold=1.0,
):
    """V-trace from log importance weights."""
    with torch.no_grad():
        rhos = torch.exp(log_rhos)
        if clip_rho_threshold is not None:
            clipped_rhos = torch.clamp(rhos, max=clip_rho_threshold)
        else:
            clipped_rhos = rhos

        cs = torch.clamp(rhos, max=1.0)
        # Append bootstrapped value to get [v1, ..., v_t+1]
        values_t_plus_1 = torch.cat(
            [values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0
        )
        deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)

        acc = torch.zeros_like(bootstrap_value)
        result = []
        for t in range(discounts.shape[0] - 1, -1, -1):
            acc = deltas[t] + discounts[t] * cs[t] * acc
            result.append(acc)
        result.reverse()
        vs_minus_v_xs = torch.stack(result)

        # Add V(x_s) to get v_s.
        vs = torch.add(vs_minus_v_xs, values)

        # Advantage for policy gradient.
        broadcasted_bootstrap_values = torch.ones_like(vs[0]) * bootstrap_value
        vs_t_plus_1 = torch.cat(
            [vs[1:], broadcasted_bootstrap_values.unsqueeze(0)], dim=0
        )
        if clip_pg_rho_threshold is not None:
            clipped_pg_rhos = torch.clamp(rhos, max=clip_pg_rho_threshold)
        else:
            clipped_pg_rhos = rhos
        pg_advantages = clipped_pg_rhos * (rewards + discounts * vs_t_plus_1 - values)

        # Make sure no gradients backpropagated through the returned values.
        return VTraceReturns(vs=vs, pg_advantages=pg_advantages)
