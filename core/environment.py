# Copyright (c) Facebook, Inc. and its affiliates.
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
"""The environment class for MonoBeast."""

import torch
import numpy as np


def _format_frame(frame):
    """
    操作前frame.shape = (C, H, W) = <class 'tuple'>: (4, 84, 84)
    操作后frame.shape = (T, B, C, H, W)
    """
    frame = torch.from_numpy(frame)
    return frame.view((1, 1) + frame.shape)  # (...) -> (T,B,...). 给frame加上Time、Batch两个维度


# 封装 gym_env：将gym env包装为pytorch可以理解的数据结构
class Environment:
    def __init__(self, gym_env):
        self.gym_env = gym_env
        self.episode_return = None # 回合奖励
        self.episode_step = None # 回合步数
        self.exploration = 0.01 # 探索


    def initial(self):
        initial_reward = torch.zeros(1, 1)
        # This supports only single-tensor actions ATM.
        initial_last_action = torch.zeros(1, 1, dtype=torch.int64)
        self.episode_return = torch.zeros(1, 1)
        self.episode_step = torch.zeros(1, 1, dtype=torch.int32)
        initial_done = torch.ones(1, 1, dtype=torch.uint8) # ones
        initial_frame = _format_frame(self.gym_env.reset()) # self.gym_env.reset().shape = (4, 84, 84); initial_frame.shape = torch.Size([1, 1, 4, 84, 84])
        # Note: gym.reset(self)：重置环境的状态，返回观测
        return dict(
            frame=initial_frame,
            reward=initial_reward,
            done=initial_done,
            episode_return=self.episode_return,
            episode_step=self.episode_step,
            last_action=initial_last_action,
        )

    # 将UAV环境的state拆分为map、scalars

    def step(self, action):
        """
        episode_step 回合步数
        reward 记录一步奖励；
        """
        frame, reward, done, unused_info = self.gym_env.step(action.item())
        self.episode_step += 1
        self.episode_return += reward
        episode_step = self.episode_step
        episode_return = self.episode_return

        # 简单处理
        frame = _format_frame(frame)
        if done:
            frame = self.gym_env.reset()
        reward = torch.tensor(reward).view(1, 1)
        done = torch.tensor(done).view(1, 1)  # TODO 低版本torch==1.1.0会把bool转型为uint8

        return_dict = dict(
            frame = frame,  # 执行action后的state
            reward = reward,  # 执行action后的reward
            done = done,  # 是否终止状态
            episode_return = episode_return,  # 执行action后的回合奖励
            episode_step = episode_step,  # 该回合的第几步
            last_action = action,  # 刚执行的action
        )

        if done:
            self.episode_return = torch.zeros(1, 1)
            self.episode_step = torch.zeros(1, 1, dtype=torch.int32)

        return return_dict


    def close(self):
        self.gym_env.close()

"""
Gym Atari Pong分析：
state: 图片 84 x 84 x 4，4表示堆叠最近的4个状态
action: 上、下、不动；更高级的游戏中还包括速度，因为滑板的速度会影响球射出去的速度和角度
reward: 挡到球得一分，挡不到球对方得一分，自己扣一分
episode: 一个回合21次机会，最高得分21，最低得分-21
done: 两个玩家只要一方得到21分就结束
"""