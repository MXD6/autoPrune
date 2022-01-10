# from env.auto_pruning_env import AutoPruningEnv
# from env.channel_pruning_env import ChannelPruningEnv
# import torchvision.models as models
# from copy import deepcopy
# import torch
# import argparse
import os

# def get_model_and_checkpoint(model, dataset, checkpoint_path, n_gpu=1):
#     if model == 'mobilenet' and dataset == 'imagenet':
#         from models.mobilenet import MobileNet
#         net = MobileNet(n_class=1000)
#     elif model == 'mobilenetv2' and dataset == 'imagenet':
#         from models.mobilenet_v2 import MobileNetV2
#         net = MobileNetV2(n_class=1000)
#     elif model == 'resnet50' and dataset == 'cifar10': # TODO 看看AMC剪枝细节。残差怎么剪枝。
#         net = models.resnet50(pretrained=False)
#         net.load_state_dict(torch.load('checkpoints/resnet50-19c8e357.pth'))
#         net = net.cuda()
#         if n_gpu > 1:
#             net = torch.nn.DataParallel(net, range(n_gpu))
#         return net, deepcopy(net.state_dict())
#     elif model =='mobilenet' and dataset == 'cifar10':
#         from models.mobilenet import MobileNet
#         net = MobileNet(n_class=10)
#     else:
#         raise NotImplementedError
#
#     sd = torch.load(checkpoint_path)
#     if 'tar' in checkpoint_path: # amc项目的加载方式
#         if 'state_dict' in sd:  # a checkpoint but not a state_dict
#             sd = sd['state_dict']
#     else:
#         sd = sd['net']
#     sd = {k.replace('module.', ''): v for k, v in sd.items()}
#     net.load_state_dict(sd)
#
#     net = net.cuda()
#     if n_gpu > 1:
#
#         net = torch.nn.DataParallel(net, range(n_gpu))
#     return net, deepcopy(net.state_dict())
#
# def parse_args():
#     parser = argparse.ArgumentParser(description='AMC search script')
#
#     parser.add_argument('--job', default='train', type=str, help='support option: train/export')
#     parser.add_argument('--suffix', default=None, type=str, help='suffix to help you remember what experiment you ran')
#     # env
#     parser.add_argument('--model', default='mobilenet', type=str, help='model to prune')
#     parser.add_argument('--dataset', default='imagenet', type=str, help='dataset to use (cifar/imagenet)')
#     parser.add_argument('--data_root', default=None, type=str, help='dataset path')
#     parser.add_argument('--preserve_ratio', default=0.5, type=float, help='preserve ratio of the model')
#     parser.add_argument('--lbound', default=0.2, type=float, help='minimum preserve ratio')
#     parser.add_argument('--rbound', default=1., type=float, help='maximum preserve ratio')
#     parser.add_argument('--reward', default='acc_reward', type=str, help='Setting the reward. You can select acc_reward/acc_flops_reward. flops用来衡量inference latency')
#     parser.add_argument('--acc_metric', default='acc5', type=str, help='use acc1 or acc5')
#     parser.add_argument('--use_real_val', dest='use_real_val', action='store_true')
#     parser.add_argument('--ckpt_path', default=None, type=str, help='manual path of checkpoint')
#     # parser.add_argument('--pruning_method', default='cp', type=str,
#     #                     help='method to prune (fg/cp for fine-grained and channel pruning)')
#     # only for channel pruning
#     parser.add_argument('--n_calibration_batches', default=60, type=int,
#                         help='n_calibration_batches')
#     parser.add_argument('--n_points_per_layer', default=10, type=int,
#                         help='method to prune (fg/cp for fine-grained and channel pruning)')
#     parser.add_argument('--channel_round', default=8, type=int, help='Round channel to multiple of channel_round')
#     # ddpg
#     parser.add_argument('--hidden1', default=300, type=int, help='hidden num of first fully connect layer')
#     parser.add_argument('--hidden2', default=300, type=int, help='hidden num of second fully connect layer')
#     parser.add_argument('--lr_c', default=1e-3, type=float, help='learning rate for actor')
#     parser.add_argument('--lr_a', default=1e-4, type=float, help='learning rate for actor')
#     parser.add_argument('--warmup', default=100, type=int,
#                         help='time without training but only filling the replay memory')
#     parser.add_argument('--discount', default=1., type=float, help='')
#     parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
#     parser.add_argument('--rmsize', default=100, type=int, help='memory size for each layer')
#     parser.add_argument('--window_length', default=1, type=int, help='')
#     parser.add_argument('--tau', default=0.01, type=float, help='moving average for target network')
#     # noise (truncated normal distribution)
#     parser.add_argument('--init_delta', default=0.5, type=float,
#                         help='initial variance of truncated normal distribution')
#     parser.add_argument('--delta_decay', default=0.95, type=float,
#                         help='delta decay during exploration') # 探索因子
#     # training
#     parser.add_argument('--max_episode_length', default=1e9, type=int, help='')
#     parser.add_argument('--output', default='./logs', type=str, help='')
#     parser.add_argument('--debug', dest='debug', action='store_true')
#     parser.add_argument('--init_w', default=0.003, type=float, help='')
#     parser.add_argument('--train_episode', default=800, type=int, help='train iters each timestep')
#     parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
#     parser.add_argument('--seed', default=None, type=int, help='random seed to set')
#     parser.add_argument('--n_gpu', default=1, type=int, help='number of gpu to use')
#     parser.add_argument('--n_worker', default=16, type=int, help='number of data loader worker')
#     parser.add_argument('--data_bsize', default=50, type=int, help='number of data batch size')
#     parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
#     # export
#     parser.add_argument('--ratios', default=None, type=str, help='ratios for pruning')
#     parser.add_argument('--channels', default=None, type=str, help='channels after pruning')
#     parser.add_argument('--export_path', default=None, type=str, help='path for exporting models')
#     parser.add_argument('--use_new_input', dest='use_new_input', action='store_true', help='use new input feature')
#
#     return parser.parse_args()

if __name__ == '__main__':

    # TODO 测试 amc channel pruning 的 extraction information 的速度
    # --job=train --model=mobilenet --dataset=cifar10 --preserve_ratio=0.5 --lbound=0.2  --rbound=1 --reward=acc_reward --data_root=./dataset/cifar10 --ckpt_path=./checkpoints/mobilenetamc_ckpt.pth --seed=2021
    # args = parse_args()
    # model, checkpoint = get_model_and_checkpoint(model="mobilenet", dataset="cifar10", checkpoint_path="./checkpoints/mobilenetamc_ckpt.pth", n_gpu=1)
    # env = ChannelPruningEnv(
    #     model=model, checkpoint=checkpoint, data="cifar10",preserve_ratio=0.5,
    #     n_data_worker=16, batch_size=50, args=args
    # )


    # TODO 测试 auto pruning 的 extraction information 的速度
    # AutoPruningEnv()


    # TODO 读取指定文件夹下的所有文件名
    # path = "J:/DesktopTransfer/科研/1.实验室方向/2.Edge Intelligence & Efficient DNN/2.Auto ML/Neural Architecture Search/Benchmark"
    # datanames = os.listdir(path)
    # for i in datanames:
    #     print(i)

    # TODO open创建csv文件及文件夹
    import os

    path = os.path.expanduser("%s/%s/%s" % ("~/autoPrune/logs", 'autoPrune-20220102-120706', '0'))
    # 创建文件夹
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + '/log.csv', 'w+') as f:
        f.write('MXD')