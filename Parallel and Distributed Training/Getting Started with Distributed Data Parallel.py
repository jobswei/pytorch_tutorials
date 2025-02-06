# DataParallel它是单进程、多线程的，但它只能在单台机器上运行。相比之下，DistributedDataParallel它是多进程的，支持单机和多机训练。

# backend：进程间通信，具体的数据传输
# 在分布式计算和深度学习框架（如 PyTorch）中，backend 具体指的是用于进程间通信的实现方式。不同的后端提供不同的通信协议和性能特性，以满足不同的计算需求。

# 常见的 Backend 类型
    # Gloo：适用于 CPU 的高效通信，支持多种操作，适合在没有 GPU 的环境下进行分布式训练。
    # NCCL：专为 NVIDIA GPU 设计，优化了 GPU 之间的通信，适合大规模的深度学习训练。
    # MPI：消息传递接口，广泛用于高性能计算，支持多种平台和通信模式。
# PyTorch 中的 Rendezvous：机制用于协调和管理多节点多进程的训练任务
    # Rendezvous 。PyTorch 提供了多种后端（如 c10d）来实现这种机制。


# DDP不管模型to了哪个device，只是把模型包起来

import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import time
from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
# 现在，让我们创建一个玩具模块，用 DDP 包装它，并向它提供一些虚拟输入数据。请注意，由于 DDP 将模型状态从等级 0 进程广播到 DDP 构造函数中的所有其他进程，因此您不必担心不同的 DDP 进程从不同的初始模型参数值开始。
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

class ToyMpModel(nn.Module):
    def __init__(self, dev0, dev1):
        super(ToyMpModel, self).__init__()
        self.dev0 = dev0
        self.dev1 = dev1
        self.net1 = torch.nn.Linear(10, 10).to(dev0)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to(dev1)

    def forward(self, x):
        x = x.to(self.dev0)
        x = self.relu(self.net1(x))
        x = x.to(self.dev1)
        return self.net2(x)
    
def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = ToyModel().to(rank)
    print(f"device:{next(model.parameters()).device}\n")
    print(f"rank:{dist.get_rank()}")
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()
    print(f"Finished running basic DDP example on rank {rank}.")

def demo_checkpoint(rank, world_size):
    print(f"Running DDP checkpoint example on rank {rank}.")
    setup(rank, world_size)

    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])


    CHECKPOINT_PATH = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/weiziyu/pytorch_tutorials/work_dirs" + "/model.pth"
    if rank == 0:
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

    # Use a barrier() to make sure that process 1 loads the model after process
    # 0 saves it.
    dist.barrier()
    # TODO: 尽管加了barrier还是要sleep，否则就会报EOFError，很奇怪，可能是OS的问题
    time.sleep(1)
    # configure map_location properly
    # 你存的state_dict没有经过to_cpu，所以是在cuda:0上
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    ddp_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=map_location, weights_only=True))

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)

    loss_fn(outputs, labels).backward()
    optimizer.step()

    # Not necessary to use a dist.barrier() to guard the file deletion below
    # as the AllReduce ops in the backward pass of DDP already served as
    # a synchronization.

    if rank == 0:
        os.remove(CHECKPOINT_PATH)

    cleanup()
    print(f"Finished running DDP checkpoint example on rank {rank}.")

def demo_model_parallel(rank, world_size):
    print(f"Running DDP with model parallel example on rank {rank}.")
    setup(rank, world_size)

    # setup mp_model and devices for this process
    dev0 = rank * 2
    dev1 = rank * 2 + 1
    mp_model = ToyMpModel(dev0, dev1)
    ddp_mp_model = DDP(mp_model)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_mp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    # outputs will be on dev1
    outputs = ddp_mp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(dev1)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()
    print(f"Finished running DDP with model parallel example on rank {rank}.")

def demo_basic_torchrun():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")
    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()
    model = ToyModel().to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(device_id)
    loss_fn(outputs, labels).backward()
    optimizer.step()
    dist.destroy_process_group()
    print(f"Finished running basic DDP example on rank {rank}.")

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

# if __name__ == "__main__":
#     n_gpus = torch.cuda.device_count()
#     assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
#     world_size = n_gpus
#     run_demo(demo_basic, world_size)
    # run_demo(demo_checkpoint, world_size)
    # world_size = n_gpus//2
    # # 与模型并行相结合
    # run_demo(demo_model_parallel, world_size)

# mp.spawn做的事情可以让torchrun来做
# torchrun --nnodes=2 --nproc_per_node=8 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 elastic_ddp.py
if __name__ == "__main__":
    demo_basic_torchrun()