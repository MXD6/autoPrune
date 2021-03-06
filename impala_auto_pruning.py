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

import argparse
import logging
import os
import pprint
import threading
import time
import timeit
import traceback
import typing
import sys
import csv
# sys.path.append('./')  # 此时假设此 py 文件和 env 文件夹在同一目录下
# import experiment.AutoPruning # TODO 导入自定义环境，gym.make()会从当前项目中找环境
os.environ["OMP_NUM_THREADS"] = "1"  # Necessary for multithreading.
import torch
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional as F
import torch.distributions as tdist
from core import environment
from core import file_writer
from core import prof
from core import vtrace
from env.auto_pruning_env import AutoPruningEnv


# yapf: disable
parser = argparse.ArgumentParser(description="PyTorch Scalable Agent")

# 调用 add_argument() 方法添加参数
parser.add_argument("--env", type=str, default="AutoPrune-v0",
                    help="Gym environment.")
parser.add_argument("--mode", default="train",
                    choices=["train", "test", "test_render"],
                    help="Training or test mode.")
parser.add_argument("--xpid", default=None,
                    help="Experiment id (default: None).")

# Training settings.
parser.add_argument("--disable_checkpoint", action="store_true",
                    help="Disable saving checkpoint.")
parser.add_argument("--savedir", default="~/logs/torchbeast/autoprune",
                    help="Root dir where experiment data will be saved.")
parser.add_argument("--num_actors", default=4, type=int, metavar="N",
                    help="Number of actors (default: 4).")
parser.add_argument("--total_steps", default=30000, type=int, metavar="T",
                    help="Total environment steps to train for.")
parser.add_argument("--batch_size", default=4, type=int, metavar="B",
                    help="Learner batch size.")
parser.add_argument("--unroll_length", default=45, type=int, metavar="T",
                    help="The unroll length (time dimension).") # 展开长度，时间维度
parser.add_argument("--num_buffers", default=None, type=int,
                    metavar="N", help="Number of shared-memory buffers.") # num_buffers应该大于batch_size和num_actors。
parser.add_argument("--seed", default=2022, type=int,
                    metavar="S", help="Seed")
parser.add_argument("--num_learner_threads", "--num_threads", default=1, type=int,
                    metavar="N", help="Number learner threads.") # learner的数量
parser.add_argument("--disable_cuda", action="store_true",
                    help="Disable CUDA.")
parser.add_argument("--use_lstm", action="store_true",
                    help="Use LSTM in agent model.")

# Loss settings.
parser.add_argument("--entropy_cost", default=0.0006,
                    type=float, help="Entropy cost/multiplier.")
parser.add_argument("--baseline_cost", default=0.5,
                    type=float, help="Baseline cost/multiplier.")
parser.add_argument("--discounting", default=0.99,
                    type=float, help="Discounting factor.")
parser.add_argument("--reward_clipping", default="abs_one",
                    choices=["abs_one", "none"],
                    help="Reward clipping.")

# Optimizer settings.
parser.add_argument("--learning_rate", default=0.00048, # 0.00003
                    type=float, metavar="LR", help="Learning rate.")
parser.add_argument("--alpha", default=0.99, type=float,
                    help="RMSProp smoothing constant.")
parser.add_argument("--momentum", default=0, type=float,
                    help="RMSProp momentum.")
parser.add_argument("--epsilon", default=0.01, type=float,
                    help="RMSProp epsilon.")
parser.add_argument("--grad_norm_clipping", default=40.0, type=float,
                    help="Global gradient norm clip.")
# yapf: enable


logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)

# 用于learner和actor之间数据通信，包括模型参数、轨迹样本
Buffers = typing.Dict[str, typing.List[torch.Tensor]] # typing.Dict['key', 'value']，创建一个字典


def compute_baseline_loss(advantages):
    return 0.5 * torch.sum(advantages ** 2)


def compute_entropy_loss(logits):
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    return torch.sum(policy * log_policy)


def compute_policy_gradient_loss_continuous(logits, actions, advantages):
    """
    logits.shape: torch.Size([unroll_length, batch_size, num_actions])
    actions.shape: torch.Size([unroll_length, batch_size])
    advantages.shape: torch.Size([unroll_length, batch_size])
    """
    logits_flatten = torch.flatten(logits, 0, 1)
    actions = torch.flatten(actions)
    negative_log_likelihood = torch.zeros(logits_flatten.shape[0], 1, dtype=torch.float32).to("cuda")
    for i in range(logits_flatten.shape[0]):
        prob_dist = tdist.Normal(logits_flatten[i][0], logits_flatten[i][1])
        negative_log_likelihood[i] = -1 * prob_dist.log_prob(actions[i])
    negative_log_likelihood = negative_log_likelihood.view_as(advantages)
    return torch.sum(negative_log_likelihood * advantages.detach())

"""
acto：
1.从learner获取network参数并更新
2.在local采集unroll_length个轨迹样本
3.将样本传送给learner
"""
def act(
        flags,  # 参数
        actor_index: int,
        free_queue: mp.SimpleQueue,
        full_queue: mp.SimpleQueue,
        model: torch.nn.Module,
        buffers: Buffers,
        initial_agent_state_buffers,
):
    try:
        logging.info("Actor %i started.", actor_index)
        timings = prof.Timings()  # Keep track of how fast things are.

        # 1.创建gym环境
        gym_env = AutoPruningEnv(actor_index, flags.xpid)
        # 2.封装gym为Environment
        env = environment.Environment(gym_env)
        # 3.Environment初始化
        env_output = env.initial()

        agent_state = model.initial_state(batch_size=1)
        agent_output, unused_state = model(env_output, agent_state)  # 前向传播，调用forward(self, inputs, core_state=()):

        while True:
            index = free_queue.get()
            if index is None:
                break

            # Write old rollout end.  将env_output、agent_output、写入buffers；其中，buffers.type = dict；将agent_state 写入LSTM state buffers；
            for key in env_output:
                buffers[key][index][0, ...] = env_output[key]
            for key in agent_output:
                buffers[key][index][0, ...] = agent_output[key]
            for i, tensor in enumerate(agent_state):
                initial_agent_state_buffers[index][i][...] = tensor

            # 2.Do new rollout. 在local采集unroll_length个样本
            for t in range(flags.unroll_length):
                timings.reset()

                with torch.no_grad(): # torch.no_grad()一个上下文管理器，被该语句 wrap 起来的部分将不会track 梯度。
                    # 1.根据frame、last action、reward, 输出action、policy logits
                    agent_output, agent_state = model(env_output, agent_state)

                timings.time("model")

                # 2.根据action，输出next reward、next frame、last action
                env_output = env.step(agent_output["action"])

                timings.time("step")

                # 3.将样本（action、reward、next state、done、behavior_policy_logits）存入buffers
                for key in env_output:
                    buffers[key][index][t + 1, ...] = env_output[key]
                for key in agent_output:
                    buffers[key][index][t + 1, ...] = agent_output[key]

                timings.time("write")
            full_queue.put(index)

        if actor_index == 0:
            logging.info("Actor %i: %s", actor_index, timings.summary())

    except KeyboardInterrupt:
        pass  # Return silently.
    except Exception as e:
        logging.error("Exception in worker process %i", actor_index)
        traceback.print_exc()
        print()
        raise e


def get_batch(
        flags,
        free_queue: mp.SimpleQueue,
        full_queue: mp.SimpleQueue,
        buffers: Buffers,
        initial_agent_state_buffers,
        timings,
        lock=threading.Lock(), # 进程加锁，是为了进程同步
):
    with lock:
        timings.time("lock") # 统计加锁时间
        indices = [full_queue.get() for _ in range(flags.batch_size)]
        timings.time("dequeue")
    # batch从buffers获取batch_size个数据
    batch = { # batch['frame'] = torch.Size([81, 8, 4, 84, 84]); batch['reward'] = torch.Size([81, 8]); batch['action'] = torch.Size([81, 8])
        key: torch.stack([buffers[key][m] for m in indices], dim=1) for key in buffers # torch.stack()把多个2维的张量凑成一个3维的张量；多个3维的凑成一个4维的张量…以此类推，也就是在增加新的维度进行堆叠。
    }
    # 从 LSTM state buffers 获取 agent_state
    initial_agent_state = (
        torch.cat(ts, dim=1)
        for ts in zip(*[initial_agent_state_buffers[m] for m in indices])
    )
    timings.time("batch")
    for m in indices:
        free_queue.put(m)
    timings.time("enqueue")
    batch = {k: t.to(device=flags.device, non_blocking=True) for k, t in batch.items()}
    initial_agent_state = tuple(
        t.to(device=flags.device, non_blocking=True) for t in initial_agent_state
    )
    timings.time("device")
    return batch, initial_agent_state


"""
learner：
1.从Actor获取样本 batch_size x unroll_length
2.训练Model
3.更新Actor的Model参数
"""
def learn(
        flags,
        actor_model,
        model,
        batch,
        initial_agent_state,
        optimizer,
        scheduler,
        lock=threading.Lock(),# noqa: B008
):
    """Performs a learning (optimization) step."""
    with lock:
        # 输入样本元组（action、reward、next state、done、behavior_policy_logits）输出（next action、target_policy_logits）
        learner_outputs, unused_state = model(batch, initial_agent_state)  # 前向传播

        # Take final value function slice for bootstrapping.
        bootstrap_value = learner_outputs["baseline"][-1]

        # Move from obs[t] -> action[t] to action[t] -> obs[t].
        batch = {key: tensor[1:] for key, tensor in batch.items()}
        learner_outputs = {key: tensor[:-1] for key, tensor in learner_outputs.items()}

        rewards = batch["reward"]
        if flags.reward_clipping == "abs_one":
            clipped_rewards = torch.clamp(rewards, -1, 1) # torch.clamp(input, min, max, out=None) 将输入input张量的每个元素夹紧到区间 [min,max]。
        elif flags.reward_clipping == "none":
            clipped_rewards = rewards

        # ~ 按位取反运算符：对数据的每个二进制位取反, 即把1变为0,把0变为1
        discounts = (~batch["done"]).float() * flags.discounting # batch["done"] dtype=bool

        # 调用vtrace算法
        vtrace_returns = vtrace.from_logits(
            behavior_policy_logits=batch["policy_logits"],
            target_policy_logits=learner_outputs["policy_logits"],
            actions=batch["action"],
            discounts=discounts,
            rewards=clipped_rewards,
            values=learner_outputs["baseline"],
            bootstrap_value=bootstrap_value,
        )

        pg_loss = compute_policy_gradient_loss_continuous(
            learner_outputs["policy_logits"],
            batch["action"],
            vtrace_returns.pg_advantages,
        )
        baseline_loss = flags.baseline_cost * compute_baseline_loss(
            vtrace_returns.vs - learner_outputs["baseline"]
        )
        entropy_loss = flags.entropy_cost * compute_entropy_loss(
            learner_outputs["policy_logits"]
        )

        # 总损失
        total_loss = pg_loss + baseline_loss + entropy_loss

        episode_returns = batch["episode_return"][batch["done"]]
        episode_steps = batch["episode_step"][batch["done"]]

        # 统计一个step的训练结果
        stats = {
            "episode_returns": tuple(episode_returns.cpu().numpy()),
            "mean_episode_return": torch.mean(episode_returns).item(),
            "total_loss": total_loss.item(),
            "pg_loss": pg_loss.item(),
            "baseline_loss": baseline_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "episode_steps": tuple(episode_steps.cpu().numpy()),
            "mean_episode_step": torch.mean(episode_steps * 1.0).item(),
        }

        optimizer.zero_grad()
        total_loss.backward()  # 反向传播，计算梯度
        nn.utils.clip_grad_norm_(model.parameters(), flags.grad_norm_clipping)
        optimizer.step()  # 根据梯度，更新参数
        scheduler.step()

        # 向actor传递新network
        actor_model.load_state_dict(model.state_dict())
        return stats


def create_buffers(flags, obs_shape, num_param_prob_dist) -> Buffers:
    """
    flags: 参数
    obs_shape: the shape of state
    num_param_prob_dist: 概率分布的参数数量；正态分布是2
    """
    T = flags.unroll_length
    specs = dict(
        frame=dict(size=(T + 1, *obs_shape), dtype=torch.uint8),
        reward=dict(size=(T + 1,), dtype=torch.float32),
        done=dict(size=(T + 1,), dtype=torch.bool),
        episode_return=dict(size=(T + 1,), dtype=torch.float32),
        episode_step=dict(size=(T + 1,), dtype=torch.int32),
        policy_logits=dict(size=(T + 1, num_param_prob_dist), dtype=torch.float32),
        baseline=dict(size=(T + 1,), dtype=torch.float32),
        last_action=dict(size=(T + 1,), dtype=torch.int64),
        action=dict(size=(T + 1,), dtype=torch.int64),
    )
    buffers: Buffers = {key: [] for key in specs}
    for _ in range(flags.num_buffers):
        for key in buffers:
            buffers[key].append(torch.empty(**specs[key]).share_memory_())
    return buffers


def train(flags):  # pylint: disable=too-many-branches, too-many-statements
    # 1.set log
    if flags.xpid is None:
        flags.xpid = "autoPrune-%s" % time.strftime("%Y%m%d-%H%M%S")
    plogger = file_writer.FileWriter(
        xpid=flags.xpid, xp_args=flags.__dict__, rootdir=flags.savedir
    )
    checkpointpath = os.path.expandvars(
        os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "model.tar"))
    )

    if flags.num_buffers is None:  # Set sensible default for num_buffers.
        flags.num_buffers = max(2 * flags.num_actors, flags.batch_size)
    if flags.num_actors >= flags.num_buffers:
        raise ValueError("num_buffers should be larger than num_actors")
    if flags.num_buffers < flags.batch_size:
        raise ValueError("num_buffers should be larger than batch_size")

    T = flags.unroll_length
    B = flags.batch_size

    flags.device = None
    if not flags.disable_cuda and torch.cuda.is_available():
        logging.info("Using CUDA.")
        flags.device = torch.device("cuda")
    else:
        logging.info("Not using CUDA.")
        flags.device = torch.device("cpu")


    # 2.create actor network;
    model = Net(observation_shape=(1, 10), num_actions=1, use_lstm=flags.use_lstm)

    # 3.create replay buffers
    buffers = create_buffers(flags, obs_shape=(1, 10), num_param_prob_dist=2)

    model.share_memory()

    # LSTM state 的缓冲池；Add initial RNN state.
    initial_agent_state_buffers = [] # actor 采集每个元组（a、r、s、done）后，learner需要执行model（输入a、r、s）输出a 来更新model，由于model包含 LSTM 部分，所以需要保存 LSTM state
    for _ in range(flags.num_buffers):
        state = model.initial_state(batch_size=1)
        for t in state:
            t.share_memory_()
        initial_agent_state_buffers.append(state)


    # 4.create and start actor's multi processes
    actor_processes = [] # torch.multiprocessing，异步多进程训练模型；使用SimpleQueue通信
    ctx = mp.get_context("spawn") # spawn
    free_queue = ctx.SimpleQueue() # 在进程中的函数不能有返回值，如果需要返回值，需要使用Queue来暂时保存返回值，等进程结束后再统一取出
    full_queue = ctx.SimpleQueue()  # 生产者进程、消费者进程，共用同一个队列进行同步。可实现数据共享，但会造成进程堵塞。

    for i in range(flags.num_actors):
        actor = ctx.Process(
            target=act,
            args=(
                flags,
                i,
                free_queue,
                full_queue,
                model,
                buffers,
                initial_agent_state_buffers,
            ),
        )
        actor.start()
        actor_processes.append(actor)

    # 5.create learner network;
    learner_model = Net(observation_shape=(1, 10), num_actions=1, use_lstm=flags.use_lstm).to(device=flags.device)

    # 6.create optimizer and scheduler
    optimizer = torch.optim.RMSprop(
        learner_model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon, # \epsilon, 加在分母上防止除0
        alpha=flags.alpha, # \alpha 平滑常数, 又叫衰减速率
    )

    def lr_lambda(epoch):
        return 1 - min(epoch * T * B, flags.total_steps) / flags.total_steps

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)  # 将每个参数组的学习率设置为初始 lr 乘以给定函数。


    logger = logging.getLogger("logfile")
    # 设置要写入 logfile 的stat key
    stat_keys = [
        "total_loss",
        "mean_episode_return",
        "pg_loss",
        "baseline_loss",
        "entropy_loss",
        "mean_episode_step",
        "episode_steps",
    ]
    logger.info("# Step\t%s", "\t".join(stat_keys))

    step, stats = 0, {}

    # learner的训练函数：
    def batch_and_learn(i, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal step, stats
        timings = prof.Timings()
        while step < flags.total_steps: # flags.total_steps，训练learner的总步数
            timings.reset()

            # 1.从actors拿到轨迹数据
            batch, agent_state = get_batch(
                flags,
                free_queue,
                full_queue,
                buffers, # buffers.shape和batch.shape相同
                initial_agent_state_buffers,
                timings,
            )
            # 2/3.learner根据轨迹数据更新net
            stats = learn(
                flags, model, learner_model, batch, agent_state, optimizer, scheduler,
            )
            timings.time("learn")
            with lock:
                to_log = dict(step=step)
                to_log.update({k: stats[k] for k in stat_keys})
                plogger.log(to_log)
                step += T * B

        if i == 0:
            logging.info("Batch and learn: %s", timings.summary())

    for m in range(flags.num_buffers):
        free_queue.put(m)

    # 7. create and start learner's multi threads
    threads = []
    for i in range(flags.num_learner_threads):
        thread = threading.Thread(
            target=batch_and_learn, name="batch-and-learn-%d" % i, args=(i,)
        )
        thread.start()
        threads.append(thread)


    # 测试线程
    def test_traning(flags, num_episodes: int = 5):
        returns = []
        while len(returns) < num_episodes:
            # 1.创建gym环境
            gym_env = AutoPruningEnv("TODO")
            # 2.封装gym为Environment
            env = environment.Environment(gym_env)
            # 3.Environment初始化
            observation = env.initial()

            model = Net(observation_shape=(1, 10), num_actions=1, use_lstm=flags.use_lstm)
            model.eval()
            model.load_state_dict(learner_model.state_dict())  # 从内存中加载模型参数
            agent_state = model.initial_state(batch_size=1)

            if flags.mode == "test_render":
                env.gym_env.render()

            while True:
                # 1.策略网络输入state输出action
                agent_outputs, agent_state = model(observation, agent_state)
                # 2.环境输入action输出state、reward
                observation = env.step(agent_outputs["action"])

                if observation["done"].item():
                    returns.append(observation["episode_return"].item())
                    break

            env.close()

        with open(os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "test_training.csv")), "a") as f:
            writer = csv.writer(f)
            writer.writerow([step, num_episodes, sum(returns) / len(returns)])


    def start_test(flags):
        nonlocal step
        count = 1
        # 测试日志
        with open(os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "test_training.csv")), "a") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "average_step", "averate_episode_return"])

        while step < flags.total_steps:
            if step > 25000 * count:
                count += 1
                # eps **= count
                test_traning(flags=flags, num_episodes=5)


    # 8.create and start the test thread
    # test_thread = threading.Thread(
    #     target=start_test, name="test-thread", args=(flags,)
    # )
    # test_thread.start()
    # threads.append(test_thread)


    def checkpoint():
        if flags.disable_checkpoint:
            return
        logging.info("Saving checkpoint to %s", checkpointpath)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "flags": vars(flags),
            },
            checkpointpath,
        )

    # 设置要打印到console的 stat key
    stat_keys_console = [
        "total_loss",
        "mean_episode_return",
        "pg_loss",
        "baseline_loss",
        "entropy_loss",
        "mean_episode_step",
        "episode_returns",
    ]

    # 9.Major Process, responsible for printing logs to the Console
    timer = timeit.default_timer
    lock = threading.Lock()
    try:
        last_checkpoint_time = timer()
        while step < flags.total_steps:
            start_step = step
            start_time = timer()
            # time.sleep(60 * 10) # TODO 设置控制台打印日志的间隔时间

            if len(stats) is not 0:
                with lock:
                    if timer() - last_checkpoint_time > 10 * 20: # 10 * 20:  # Save every 10 min = 10 * 60.
                        checkpoint()
                        last_checkpoint_time = timer()

                    sps = (step - start_step) / (timer() - start_time)
                    if stats.get("episode_returns", None):
                        mean_return = (
                                "Return per episode: %.1f. " % stats["mean_episode_return"]
                        )
                    else:
                        mean_return = ""
                    total_loss = stats.get("total_loss", float("inf"))

                    stats_console = dict(step=step)
                    stats_console.update({k: stats[k] for k in stat_keys_console})
                    logging.info(
                        "Steps %i @ %.1f SPS. Loss %f. %sStats:\n%s",
                        step,
                        sps,
                        total_loss,
                        mean_return,
                        pprint.pformat(stats_console),
                    )
                    stats = {}

    except KeyboardInterrupt:
        return  # Try joining actors then quit.
    else:
        for thread in threads:
            thread.join()
        logging.info("Learning finished after %d steps.", step)
    finally:
        for _ in range(flags.num_actors):
            free_queue.put(None)
        for actor in actor_processes:
            actor.join(timeout=1)

    checkpoint()
    plogger.close()



# def test(flags, num_episodes: int = 1):
#     if flags.xpid is None:
#         checkpointpath = "./latest/model.tar"
#     else:
#         checkpointpath = os.path.expandvars(
#             os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "model.tar"))
#         )
#
#     # 1.创建gym环境
#     gym_env = gym.make(flags.env)
#     # 2.封装gym为Environment
#     env = environment.Environment(gym_env)
#     # 3.Environment初始化
#     observation = env.initial_uav()
#
#     model = Net((4, 63, 63), gym_env.action_space.n, flags.use_lstm)
#     model.eval()
#     checkpoint = torch.load(checkpointpath, map_location="cpu")
#     model.load_state_dict(checkpoint["model_state_dict"]) # 从文件中加载模型参数
#     agent_state = model.initial_state(batch_size=1)
#
#     returns = []
#
#     while len(returns) < num_episodes:
#         if flags.mode == "test_render":
#             env.gym_env.render()
#
#         # 1.策略网络输入state输出action
#         agent_outputs, agent_state = model(observation, agent_state)
#         # 2.环境输入action输出state、reward
#         observation = env.step_uav(agent_outputs["action"])
#
#         if observation["done"].item():
#             returns.append(observation["episode_return"].item())
#             logging.info(
#                 "Episode ended after %d steps. Return: %.1f",
#                 observation["episode_step"].item(),
#                 observation["episode_return"].item(),
#             )
#
#     env.close()
#     logging.info(
#         "Average returns over %i steps: %.1f", num_episodes, sum(returns) / len(returns)
#     )


class AutoPruneNet(nn.Module):

    def __init__(self, observation_shape, num_actions, use_lstm=False):
        super(AutoPruneNet, self).__init__()
        self.use_lstm = use_lstm
        self.observation_shape = observation_shape
        self.num_actions = num_actions

        hidden1 = 400
        hidden2 = 300
        self.fc1 = nn.Linear(observation_shape[1], hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)

        core_output_size = self.fc2.out_features + num_actions + 1 # 1 = reward

        if use_lstm:
            self.core = nn.LSTM(core_output_size, core_output_size, 2) # 2层, 4层

        self.policy = nn.Linear(core_output_size, 2)
        self.baseline = nn.Linear(core_output_size, 1)


    def initial_state(self, batch_size):
        if not self.use_lstm:
            return tuple()
        return tuple(
            torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size)
            for _ in range(2)
        )

    def forward(self, inputs, core_state=()):
        """
        agent 输入 image、remain data、last action、reward，输出 action。
        使用 action、behavior_policy_logits、reward、next state、done、next action、target_policy_logits 更新 agent。
        """
        x = inputs["frame"]  # [T, B, C, H, W].  T = Time, B = Batch
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # 合并T、B维度后，x.shape = [T*B, C, H, W] = [1, 4, 63, 63]
        x = x.float() #  / 255.0

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(T * B, -1)

        last_action = inputs["last_action"].view(T * B, 1).float()  # last_action.shape = torch.Size([1, 1])
        clipped_reward = torch.clamp(inputs["reward"], -1, 1).view(T * B, 1).float()  # 奖励裁剪 clipped_reward.shape = torch.Size([1, 1])

        core_input = torch.cat([x, clipped_reward, last_action], dim=-1)
        """
        torch.cat((a, b), dim=1)按维度1将a、b拼接。
        torch.unbind(input, dim=0)删除input张量的维度0
        """

        if self.use_lstm and len(core_state) is not 0:
            core_input = core_input.view(T, B, -1)
            core_output_list = []
            notdone = (~inputs["done"]).float()
            for input, nd in zip(core_input.unbind(), notdone.unbind()):
                # Reset core state to zero whenever an episode ended.
                # Make `done` broadcastable with (num_layers, B, hidden_size)
                # states:
                nd = nd.view(1, -1, 1)
                core_state = tuple(nd * s for s in core_state)
                output, core_state = self.core(input.unsqueeze(0), core_state)
                core_output_list.append(output)
            core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        else:
            core_output = core_input
            core_state = tuple()

        policy_logits = torch.sigmoid(self.policy(core_output)) # policy_logits.shape = (1, 2) 方差的取值范围是非负数，所以对mu, sigma做sigmoid运算
        # print(policy_logits.shape)
        # print(policy_logits)
        baseline = self.baseline(core_output)

        # 无论 train 还是 test 都得采样
        action = torch.zeros(policy_logits.shape[0], 1)
        for i in range(policy_logits.shape[0]):
            action[i] = tdist.Normal(policy_logits[i][0], policy_logits[i][1]).sample()

        policy_logits = policy_logits.view(T, B, 2)
        baseline = baseline.view(T, B) # baseline.shape = torch.Size([1, 1])
        action = action.view(T, B) # action.shape = torch.Size([1, 1]) TODO bug: 动作应该在0-1之间

        return (
            dict(policy_logits=policy_logits, baseline=baseline, action=action), # policy_logits.type=torch.float32; action.type=torch.int64
            core_state,
        )

Net = AutoPruneNet


def main(flags):
    if flags.mode == "train":
        train(flags)
    # else:
        # test(flags)

if __name__ == "__main__":
    flags = parser.parse_args()
    main(flags)


"""
从项目根目录输入下列命令启动，保证导包正确
python -m impala_auto_pruning --mode train --use_lstm

"""

