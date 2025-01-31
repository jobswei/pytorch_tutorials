# pytorch_tutorials
学习pytorch

开始于 2025-01-26


# Learn the Basics
[Learn the Basics](https://pytorch.org/tutorials/beginner/basics/intro.html): 自动求导机制，计算图简介

* 自动微分机制：https://pytorch.org/docs/stable/notes/autograd.html
* Functions母函数: https://pytorch.org/docs/stable/generated/torch.autograd.Function.backward.html#torch.autograd.Function.backward
* 损失函数：https://pytorch.org/docs/stable/nn.html#loss-functions
* 优化算法：https://pytorch.org/docs/stable/optim.html
* torch.fx: 捕获修改计算图
* torch.autograd库
* torch.autograd.grad: https://pytorch.org/cppdocs/api/function_namespacetorch_1_1autograd_1ab9fa15dc09a8891c26525fb61d33401a.html
* torch.autograd.backward: https://pytorch.org/cppdocs/api/function_namespacetorch_1_1autograd_1a1403bf65b1f4f8c8506a9e6e5312d030.html
* 看看每一个基础模块中的forward和backward函数

# Deploying PyTorch Models in Production
* [ONNX 简介与部署](): 还没学
* [depoly with flask](https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html#): 借助flask进行web部署
    * 生产环境部署flask: https://flask.palletsprojects.com/en/stable/tutorial/deploy/
    * UI: https://github.com/avinassh/pytorch-flask-api-heroku
* [Introduction to TorchScript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html): TorchScript简介
* [Loading a TorchScript Model in C++](https://pytorch.org/tutorials/advanced/cpp_export.html): C++导入TorchScript Model，编译，运行推理

# Profiling PyTorch
* [profiler 的应用](https://pytorch.org/tutorials/beginner/profiler.html)
    * torch.profiler: https://pytorch.org/docs/main/profiler.html
    * torch profiler recipes: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
    * Profiler API : https://pytorch.org/docs/stable/autograd.html?highlight=profiler#profiler
* [HolisticTraceAnalysis使用](https://pytorch.org/tutorials/beginner/hta_intro_tutorial.html): 分析cuda在训练时的空闲时间，内核占用等

# Fronted APIs

* [Channels Last Memory Format](https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html): 通道最后存储方式，提高性能
* [Forward-mode Automatic Differentiation](https://pytorch.org/tutorials/intermediate/forward_ad_usage.html): 前向自动微分，forward的同时计算梯度，适合
    * functorch: https://github.com/pytorch/functorch
    * forward-mode AD: https://pytorch.org/docs/main/notes/extending.html#forward-mode-ad

* [composing function transforms](https://pytorch.org/functorch/stable/notebooks/jacobians_hessians.html): 雅可比、海森矩阵计算，正向方法与逆向方法

* [Model ensembling](https://pytorch.org/tutorials/intermediate/ensembling.html): torch.vmap加快集成模型速度

* [Per-sample-gradients](https://pytorch.org/tutorials/intermediate/per_sample_grads.html): 单个样本梯度计算，vmap应用举例

* [Using the PyTorch C++ Fronted](https://pytorch.org/tutorials/advanced/cpp_frontend.html): C++的torch库使用
    * C++ API: https://pytorch.org/cppdocs/
    * torch::nn内置模块完整列表：https://pytorch.org/cppdocs/api/namespace_torch__nn.html
    * https://pytorch.org/cppdocs/api/classtorch_1_1nn_1_1_module.html#exhale-class-classtorch-1-1nn-1-1-module
* [Dynamic Parallelism in TorchScript](https://pytorch.org/tutorials/advanced/torch-script-parallelism.html): 使用torch.jit.fork使计算并行加速
* [Autograd in C++ Frontend](https://pytorch.org/tutorials/advanced/cpp_autograd.html)
    * C++ tensor API :https://pytorch.org/cppdocs/api/classat_1_1_tensor.html