
#include <torch/torch.h>
#include <iostream>
const int kNoiseSize = 100;
const int kBatchSize = 64;
const int kNumberOfEpochs = 5;

namespace torch{
    struct DCGANGeneratorImpl : nn::Module {
        DCGANGeneratorImpl(int kNoiseSize)
            : conv1(nn::ConvTranspose2dOptions(kNoiseSize, 256, 4)
                        .bias(false)),
                batch_norm1(256),
                conv2(nn::ConvTranspose2dOptions(256, 128, 3)
                        .stride(2)
                        .padding(1)
                        .bias(false)),
                batch_norm2(128),
                conv3(nn::ConvTranspose2dOptions(128, 64, 4)
                        .stride(2)
                        .padding(1)
                        .bias(false)),
                batch_norm3(64),
                conv4(nn::ConvTranspose2dOptions(64, 1, 4)
                        .stride(2)
                        .padding(1)
                        .bias(false))
        {
            // register_module() is needed if we want to use the parameters() method later on
            register_module("conv1", conv1);
            register_module("conv2", conv2);
            register_module("conv3", conv3);
            register_module("conv4", conv4);
            register_module("batch_norm1", batch_norm1);
            register_module("batch_norm2", batch_norm2);
            register_module("batch_norm3", batch_norm3);
        }

        torch::Tensor forward(torch::Tensor x) {
            x = torch::relu(batch_norm1(conv1(x)));
            x = torch::relu(batch_norm2(conv2(x)));
            x = torch::relu(batch_norm3(conv3(x)));
            x = torch::tanh(conv4(x));
            return x;
        }

        nn::ConvTranspose2d conv1, conv2, conv3, conv4;
        nn::BatchNorm2d batch_norm1, batch_norm2, batch_norm3;
    };
    TORCH_MODULE(DCGANGenerator);

    nn::Sequential discriminator(
        // Layer 1
        nn::Conv2d(
            nn::Conv2dOptions(1, 64, 4).stride(2).padding(1).bias(false)),
        nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
        // Layer 2
        nn::Conv2d(
            nn::Conv2dOptions(64, 128, 4).stride(2).padding(1).bias(false)),
        nn::BatchNorm2d(128),
        nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
        // Layer 3
        nn::Conv2d(
            nn::Conv2dOptions(128, 256, 4).stride(2).padding(1).bias(false)),
        nn::BatchNorm2d(256),
        nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
        // Layer 4
        nn::Conv2d(
            nn::Conv2dOptions(256, 1, 3).stride(1).padding(0).bias(false)),
        nn::Sigmoid());

}

torch::DCGANGenerator generator(kNoiseSize);
torch::nn::Sequential discriminator = torch::discriminator;

// dataset的读取报segment default 待解决
auto dataset = torch::data::datasets::MNIST("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/weiziyu/pytorch_tutorials/Fronted APIs/Using the PyTorch C++ Fronted/MNIST")
    .map(torch::data::transforms::Normalize<>(0.5, 0.5))
    .map(torch::data::transforms::Stack<>());

// auto data_loader = torch::data::make_data_loader(
//     std::move(dataset),
//     torch::data::DataLoaderOptions().batch_size(kBatchSize).workers(2));

torch::optim::Adam generator_optimizer(
    generator->parameters(), torch::optim::AdamOptions(2e-4).betas(std::make_tuple(0.5, 0.5)));
torch::optim::Adam discriminator_optimizer(
    discriminator->parameters(), torch::optim::AdamOptions(5e-4).betas(std::make_tuple(0.5, 0.5)));

int main(){

    // for (torch::data::Example<>& batch : *data_loader) {
    //     std::cout << "Batch size: " << batch.data.size(0) << " | Labels: ";
    //     for (int64_t i = 0; i < batch.data.size(0); ++i) {
    //         std::cout << batch.target[i].item<int64_t>() << " ";
    //     }
    //     std::cout << std::endl;
    //     break;
    //     }

    // torch::Device device = torch::kCPU;
    // if (torch::cuda::is_available()) {
    //     std::cout << "CUDA is available! Training on GPU." << std::endl;
    //     device = torch::kCUDA;
    // }

    // generator -> to(device);
    // discriminator -> to(device);

    // for (int64_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
    //     int64_t batch_index = 0;
    //     for (torch::data::Example<>& batch : *data_loader) {
    //         // Train discriminator with real images.
    //         discriminator->zero_grad();
    //         torch::Tensor real_images = batch.data.to(device);
    //         torch::Tensor real_labels = torch::empty(batch.data.size(0),device).uniform_(0.8, 1.0);
    //         torch::Tensor real_output = discriminator->forward(real_images).reshape(real_labels.sizes());
    //         torch::Tensor d_loss_real = torch::binary_cross_entropy(real_output, real_labels);
    //         d_loss_real.backward();

    //         // Train discriminator with fake images.
    //         torch::Tensor noise = torch::randn({batch.data.size(0), kNoiseSize, 1, 1},device);
    //         torch::Tensor fake_images = generator->forward(noise);
    //         torch::Tensor fake_labels = torch::zeros(batch.data.size(0),device);
    //         torch::Tensor fake_output = discriminator->forward(fake_images.detach()).reshape(fake_labels.sizes());
    //         torch::Tensor d_loss_fake = torch::binary_cross_entropy(fake_output, fake_labels);
    //         d_loss_fake.backward();

    //         torch::Tensor d_loss = d_loss_real + d_loss_fake;
    //         discriminator_optimizer.step();

    //         // Train generator.
    //         generator->zero_grad();
    //         fake_labels.fill_(1);
    //         fake_output = discriminator->forward(fake_images).reshape(fake_labels.sizes());
    //         torch::Tensor g_loss = torch::binary_cross_entropy(fake_output, fake_labels);
    //         g_loss.backward();
    //         generator_optimizer.step();

    //         std::printf(
    //             "\r[%2ld/%2ld][%3ld] D_loss: %.4f | G_loss: %.4f",
    //             epoch,
    //             kNumberOfEpochs,
    //             ++batch_index,
    //             d_loss.item<float>(),
    //             g_loss.item<float>());
    //     }
    // }
}