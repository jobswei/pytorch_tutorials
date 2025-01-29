import torch
import torch.nn as nn
import torch.optim as optim
import torch.profiler as profiler
import torch.distributed as dist

# 假设您已经初始化了分布式环境
dist.init_process_group(backend='nccl')

# 在每个进程中生成跟踪文件
rank = dist.get_rank()
# 定义一个简单的神经网络模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        with profiler.record_function("NAME"):
            return self.fc2(torch.relu(self.fc1(x)))

# 初始化模型、损失函数和优化器
model = SimpleModel().to("cuda:0")
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 生成一些随机数据
inputs = torch.randn(64, 10).to("cuda:0")  # 64 个样本，每个样本 10 个特征
targets = torch.randn(64, 1).to("cuda:0")   # 64 个目标值

# 使用 PyTorch Profiler 生成跟踪数据
with torch.profiler.profile(
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
    # on_trace_ready=torch.profiler.tensorboard_trace_handler('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/weiziyu/pytorch_tutorials/work_dirs/traces/logs'),
    record_shapes=True,
    profile_memory=True,
) as prof:
    for epoch in range(25):  # 训练 5 个 epoch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        prof.step()

# 导出跟踪数据 。这个和上面那个on_trace_ready做的是同一件事，不能同时出现
prof.export_chrome_trace("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/weiziyu/pytorch_tutorials/work_dirs/traces/trace.json")
