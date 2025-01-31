struct Net : torch::nn::Module { };


void a(Net net) { } // 值传递，不会影响外面
void b(Net& net) { } // 引用传递，会影响外面
void c(Net* net) { } // 指针传递，会影响外面

void d(std::shared_ptr<Net> net) { }

int main() {
  Net net;
  a(net);
  a(std::move(net));// std::move主要作用是将一个对象的值转移给另一个对象，而不是进行复制，从而提高性能，尤其是在处理大型对象时
  b(net);
  c(&net);

  auto net = std::make_shared<Net>();
  d(net);
}

// 为了使用linear子模块，我们希望将其直接存储在我们的类中。但是，我们还希望模块基类知道并可以访问此子模块。为此，它必须存储对此子模块的引用。此时，我们已经到了共享所有权的需要。类 torch::nn::Module和具体Net类都需要对子模块的引用。因此，基类将模块存储为 shared_ptrs，因此具体类也必须如此
struct Net : torch::nn::Module {
  Net(int64_t N, int64_t M)
    : linear(register_module("linear", torch::nn::Linear(N, M)))
  { }
  torch::nn::Linear linear;
};
// shared_ptr但是等一下！我在上面的代码中没有看到任何提及！这是为什么呢？因为std::shared_ptr<MyModule>要输入的内容实在是太多了。为了让我们的研究人员保持高效，我们想出了一个精心设计的方案来隐藏提及shared_ptr——这通常是为了值语义而保留的——同时保留引用语义。要了解这是如何工作的，我们可以看一下torch::nn::Linear核心库中模块的简化定义（完整定义在这里）：
struct LinearImpl : torch::nn::Module {
  LinearImpl(int64_t in, int64_t out);

  Tensor forward(const Tensor& input);

  Tensor weight, bias;
};

TORCH_MODULE(Linear);
// 简而言之：模块不叫Linear，而是LinearImpl。然后，宏 TORCH_MODULE定义实际的Linear类。这个“生成的”类实际上是 的包装器std::shared_ptr<LinearImpl>。它是一个包装器而不是简单的 typedef，因此，除其他外，构造函数仍按预期工作，即，您仍然可以编写torch::nn::Linear(3, 4) 而不是std::make_shared<LinearImpl>(3, 4)。我们将宏创建的类称为模块持有者。与（共享）指针一样，您可以使用箭头运算符（如model->forward(...) ）访问底层对象。最终结果是所有权模型与 Python API 非常相似。引用语义成为默认设置，但无需额外输入std::shared_ptr 或std::make_shared<Net>。
// TORCH_MODULE应用举例
struct NetImpl : torch::nn::Module {};
TORCH_MODULE(Net);

void a(Net net) { }

int main() {
  Net net;
  a(net);
}

// 默认构造的 std::shared_ptr是“空的”，即包含一个空指针。什么是默认构造的Linear或Net？嗯，这是一个棘手的选择。我们可以说它应该是一个空的 (null) std::shared_ptr<LinearImpl>。但是，回想一下Linear(3, 4) 和 std::make_shared<LinearImpl>(3, 4)是一样的。这意味着如果我们决定Linear linear应该是一个空指针，那么就无法构造一个不接受任何构造函数参数或将所有参数都设置为默认值的模块。因此，在当前 API 中，默认构造的模块持有者（如Linear()）会调用底层模块的默认构造函数（LinearImpl()）。如果底层模块没有默认构造函数，则会收到编译器错误。要构造空持有者，可以nullptr传递给持有者的构造函数。;

// 实际上，这意味着你可以像前面所示的那样使用子模块，其中模块在初始化列表中注册和构造
struct Net : torch::nn::Module {
  Net(int64_t N, int64_t M)
    : linear(register_module("linear", torch::nn::Linear(N, M)))
  { }
  torch::nn::Linear linear;
};
// 或者你可以先用空指针构造持有者，然后在构造函数中分配给它（对于 Pythonistas 来说更熟悉）：
struct Net : torch::nn::Module {
  Net(int64_t N, int64_t M) {
    linear = register_module("linear", torch::nn::Linear(N, M));
  }
  torch::nn::Linear linear{nullptr}; // construct an empty holder
};