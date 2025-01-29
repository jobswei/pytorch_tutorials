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

