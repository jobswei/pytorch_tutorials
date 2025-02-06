

"""run.py:"""
#!/usr/bin/env python
import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time

# ======= 点对点通信 =======
"""Blocking point-to-point communication."""
def run1(rank, size):
    # 两个进程都以零张量开始，然后进程 0 增加张量并将其发送给进程 1，这样它们最终都得到 1.0。请注意，进程 1 需要分配内存来存储它将收到的数据。
    # 还要注意send/recv是阻塞的：两个进程都阻塞，直到通信完成。另一方面，immediate 是 非阻塞的；脚本继续执行，方法返回一个Work对象，我们可以选择 wait()
    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
    print('Rank ', rank, ' has data ', tensor[0])
    
"""Non-blocking point-to-point communication."""
def run2(rank, size):
    # 使用立即数时，我们必须小心使用发送和接收的张量。由于我们不知道数据何时会传送给另一个进程，因此在req.wait()完成之前，我们不应该修改发送的张量，也不应该访问接收的张量。换句话说，
        # tensor之后写入dist.isend()将导致未定义的行为。
        # tensor从之后读取dist.irecv()将导致未定义的行为，直到req.wait()执行完毕。
    # 然而，req.wait() 执行之后我们可以保证通信已经发生，并且存储的值tensor[0]是 1.0。
    tensor = torch.zeros(1)
    req = None
    if rank == 0:
        time.sleep(5)
        tensor += 1
        # Send the tensor to process 1
        print('Rank 0 started sending')
        req = dist.isend(tensor=tensor, dst=1)
    else:
        # Receive tensor from process 0
        print('Rank 1 started receiving')
        req = dist.irecv(tensor=tensor, src=0)
        # time.sleep(5)
    # 加wait就同步，不加wait就不同步了。
    # 不wait，rank1收不到，打印原tensor。rank0 send的时候 rank2已经关了，所以报错了
    # req.wait()
    print('Rank ', rank, ' has data ', tensor[0])
# ======= 点对点通信 =======


# 操作符 https://pytorch.org/docs/stable/distributed.html#torch.distributed.ReduceOp
# dist.ReduceOp.SUM，
# dist.ReduceOp.PRODUCT，
# dist.ReduceOp.MAX，
# dist.ReduceOp.MIN，
# dist.ReduceOp.BAND，
# dist.ReduceOp.BOR，
# dist.ReduceOp.BXOR，
# dist.ReduceOp.PREMUL_SUM。

# 通信方法
# dist.broadcast(tensor, src, group)：从src复制tensor到所有其他进程。
# dist.reduce(tensor, dst, op, group)：op应用于每个tensor并将结果存储在中dst。
# dist.all_reduce(tensor, op, group)：与reduce相同，但结果存储在所有进程中。
# dist.scatter(tensor, scatter_list, src, group)：复制scatter_list[i]到第i个进程
# dist.gather(tensor, gather_list, dst, group)：从所有进程中复制tensor到dst。
# dist.all_gather(tensor_list, tensor, group)：从所有进程中复制tensor到所有进程上。
# dist.barrier(group)：阻止组内所有进程，直到每个进程都进入此功能。
# dist.all_to_all(output_tensor_list, input_tensor_list, group)：将输入张量列表分散到组中的所有进程，并在输出列表中返回收集的张量列表。
""" All-Reduce example."""
def run3(rank, size):
    """ Simple collective communication. """
    group = dist.new_group([0, 1])
    tensor = torch.ones(1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    print('Rank ', rank, ' has data ', tensor[0])
    
    
def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    world_size = 2
    processes = []
    if "google.colab" in sys.modules:
        print("Running in Google Colab")
        mp.get_context("spawn")
    else:
        mp.set_start_method("spawn")
    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size, run3))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()