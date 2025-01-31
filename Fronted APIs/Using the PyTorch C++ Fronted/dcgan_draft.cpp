
#include <torch/torch.h>
#include <iostream>


struct Net : torch::nn::Module {
// 为什么子模块是在构造函数的初始化列表中创建的，而参数是在构造函数主体内创建的。这有一个很好的理由，我们将在下面的 C++ 前端所有权模型部分中讨论这一点
    Net(int64_t N, int64_t M)
        : linear(register_module("linear", torch::nn::Linear(N,M))){
        another_bias = register_parameter("b",torch::randn(M));
    }

    torch::Tensor forward(torch::Tensor input){
        return linear(input) + another_bias;
    }

    torch::Tensor another_bias;
    torch::nn::Linear linear;
};


int main() {
    Net net(4,5);
    std::cout<<"===== 模型参数 ====="<<std::endl;
    for (const auto& pair : net.named_parameters()){
        std::cout << pair.key() << ": " << pair.value() << std::endl;
    }
    std::cout<<"===== 前向 ====="<<std::endl;
    std::cout << net.forward(torch::ones({2, 4})) << std::endl;
    return 0;
}