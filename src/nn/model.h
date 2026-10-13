// Residual policy/value network (AlphaZero / KataGo style) implemented with LibTorch.
//
// Heads:
//   policy     : 82 logits (81 points + pass)
//   value      : tanh scalar, expected result for the player to move (-1 loss .. +1 win)
//   score      : expected final score lead for the player to move, in units of 10 points
//   ownership  : 81 tanh values, +1 = owned by player to move at the end of the game
#pragma once

#include <torch/torch.h>

#include <string>

#include "nn/features.h"

namespace nn {

struct NetConfig {
  int blocks = 6;
  int channels = 96;
  int head_channels = 32;
};

struct NetOutput {
  torch::Tensor policy_logits;  // [B, 82]
  torch::Tensor value;          // [B]
  torch::Tensor score;          // [B]  (score lead / 10)
  torch::Tensor ownership;      // [B, 81]
};

struct ResBlockImpl : torch::nn::Module {
  explicit ResBlockImpl(int ch);
  torch::Tensor forward(const torch::Tensor& x);
  torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
  torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
};
TORCH_MODULE(ResBlock);

// Residual block with KataGo-style global pooling bias: channels of the first
// conv are pooled (mean + max) over the board and fed back as a per-channel
// bias, which helps the network reason about whole-board properties (ko, life).
struct GPoolBlockImpl : torch::nn::Module {
  explicit GPoolBlockImpl(int ch);
  torch::Tensor forward(const torch::Tensor& x);
  torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
  torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
  torch::nn::Linear gpool_fc{nullptr};
};
TORCH_MODULE(GPoolBlock);

struct GoNetImpl : torch::nn::Module {
  explicit GoNetImpl(const NetConfig& cfg);
  NetOutput forward(const torch::Tensor& x);

  NetConfig cfg;
  torch::nn::Conv2d stem{nullptr};
  torch::nn::BatchNorm2d stem_bn{nullptr};
  torch::nn::Sequential trunk{nullptr};
  // policy head
  torch::nn::Conv2d p_conv{nullptr};
  torch::nn::BatchNorm2d p_bn{nullptr};
  torch::nn::Linear p_gpool{nullptr};
  torch::nn::Conv2d p_out{nullptr};
  torch::nn::Linear p_pass{nullptr};
  // value / score / ownership head
  torch::nn::Conv2d v_conv{nullptr};
  torch::nn::BatchNorm2d v_bn{nullptr};
  torch::nn::Linear v_fc{nullptr};
  torch::nn::Linear v_out{nullptr};
  torch::nn::Linear s_out{nullptr};
  torch::nn::Conv2d o_out{nullptr};
};
TORCH_MODULE(GoNet);

// Saves weights + architecture into a single file; load reconstructs the net.
void save_model(GoNet& net, const std::string& path);
GoNet load_model(const std::string& path, const torch::Device& device);
// Reads only the architecture stored in a model file.
NetConfig read_model_config(const std::string& path);

int64_t count_parameters(GoNet& net);

}  // namespace nn
