#include "nn/model.h"

#include <filesystem>
#include <stdexcept>

namespace nn {

namespace F = torch::nn::functional;
using torch::nn::BatchNorm2d;
using torch::nn::BatchNorm2dOptions;
using torch::nn::Conv2d;
using torch::nn::Conv2dOptions;
using torch::nn::Linear;

namespace {
Conv2d conv(int in, int out, int k) {
  return Conv2d(Conv2dOptions(in, out, k).padding(k / 2).bias(false));
}
torch::Tensor global_pool(const torch::Tensor& x) {
  // [B, C, H, W] -> [B, 2C] (mean, max)
  auto mean = x.mean({2, 3});
  auto mx = std::get<0>(x.flatten(2).max(2));
  return torch::cat({mean, mx}, 1);
}
}  // namespace

ResBlockImpl::ResBlockImpl(int ch) {
  conv1 = register_module("conv1", conv(ch, ch, 3));
  bn1 = register_module("bn1", BatchNorm2d(BatchNorm2dOptions(ch)));
  conv2 = register_module("conv2", conv(ch, ch, 3));
  bn2 = register_module("bn2", BatchNorm2d(BatchNorm2dOptions(ch)));
}

torch::Tensor ResBlockImpl::forward(const torch::Tensor& x) {
  auto h = torch::relu(bn1(conv1(x)));
  h = bn2(conv2(h));
  return torch::relu(x + h);
}

GPoolBlockImpl::GPoolBlockImpl(int ch) {
  conv1 = register_module("conv1", conv(ch, ch, 3));
  bn1 = register_module("bn1", BatchNorm2d(BatchNorm2dOptions(ch)));
  gpool_fc = register_module("gpool_fc", Linear(2 * ch, ch));
  conv2 = register_module("conv2", conv(ch, ch, 3));
  bn2 = register_module("bn2", BatchNorm2d(BatchNorm2dOptions(ch)));
}

torch::Tensor GPoolBlockImpl::forward(const torch::Tensor& x) {
  auto h = torch::relu(bn1(conv1(x)));
  auto g = gpool_fc(global_pool(h));
  h = h + g.unsqueeze(2).unsqueeze(3);
  h = bn2(conv2(h));
  return torch::relu(x + h);
}

GoNetImpl::GoNetImpl(const NetConfig& c) : cfg(c) {
  const int ch = cfg.channels, hc = cfg.head_channels;
  stem = register_module("stem", conv(kInputPlanes, ch, 3));
  stem_bn = register_module("stem_bn", BatchNorm2d(BatchNorm2dOptions(ch)));
  trunk = torch::nn::Sequential();
  for (int i = 0; i < cfg.blocks; ++i) {
    // Every third block (starting with the 3rd) uses global pooling.
    if (i % 3 == 2) trunk->push_back(GPoolBlock(ch));
    else trunk->push_back(ResBlock(ch));
  }
  register_module("trunk", trunk);

  p_conv = register_module("p_conv", conv(ch, hc, 1));
  p_bn = register_module("p_bn", BatchNorm2d(BatchNorm2dOptions(hc)));
  p_gpool = register_module("p_gpool", Linear(2 * hc, hc));
  p_out = register_module("p_out", Conv2d(Conv2dOptions(hc, 1, 1)));
  p_pass = register_module("p_pass", Linear(2 * hc, 1));

  v_conv = register_module("v_conv", conv(ch, hc, 1));
  v_bn = register_module("v_bn", BatchNorm2d(BatchNorm2dOptions(hc)));
  v_fc = register_module("v_fc", Linear(2 * hc, 128));
  v_out = register_module("v_out", Linear(128, 1));
  s_out = register_module("s_out", Linear(128, 1));
  o_out = register_module("o_out", Conv2d(Conv2dOptions(hc, 1, 1)));
}

NetOutput GoNetImpl::forward(const torch::Tensor& x) {
  auto h = torch::relu(stem_bn(stem(x)));
  h = trunk->forward(h);

  // Policy
  auto p = torch::relu(p_bn(p_conv(h)));
  auto pg = global_pool(p);
  p = p + p_gpool(pg).unsqueeze(2).unsqueeze(3);
  auto board_logits = p_out(torch::relu(p)).flatten(1);  // [B, 81]
  auto pass_logit = p_pass(pg);                          // [B, 1]
  NetOutput out;
  out.policy_logits = torch::cat({board_logits, pass_logit}, 1);

  // Value / score / ownership
  auto v = torch::relu(v_bn(v_conv(h)));
  out.ownership = torch::tanh(o_out(v).flatten(1));
  auto vf = torch::relu(v_fc(global_pool(v)));
  out.value = torch::tanh(v_out(vf)).squeeze(1);
  out.score = s_out(vf).squeeze(1);
  return out;
}

void save_model(GoNet& net, const std::string& path) {
  torch::serialize::OutputArchive archive;
  net->save(archive);
  archive.write("meta_blocks", torch::tensor(int64_t(net->cfg.blocks)));
  archive.write("meta_channels", torch::tensor(int64_t(net->cfg.channels)));
  archive.write("meta_head_channels", torch::tensor(int64_t(net->cfg.head_channels)));
  archive.write("meta_input_planes", torch::tensor(int64_t(kInputPlanes)));
  // Write to a temp file then rename so an interrupted save never corrupts the model.
  const std::string tmp = path + ".tmp";
  archive.save_to(tmp);
  std::filesystem::rename(tmp, path);
}

NetConfig read_model_config(const std::string& path) {
  torch::serialize::InputArchive archive;
  archive.load_from(path, torch::kCPU);
  torch::Tensor t;
  NetConfig cfg;
  archive.read("meta_blocks", t);
  cfg.blocks = int(t.item<int64_t>());
  archive.read("meta_channels", t);
  cfg.channels = int(t.item<int64_t>());
  archive.read("meta_head_channels", t);
  cfg.head_channels = int(t.item<int64_t>());
  archive.read("meta_input_planes", t);
  if (t.item<int64_t>() != kInputPlanes)
    throw std::runtime_error("Model " + path + " was built for a different input encoding");
  return cfg;
}

GoNet load_model(const std::string& path, const torch::Device& device) {
  NetConfig cfg = read_model_config(path);
  GoNet net(cfg);
  torch::serialize::InputArchive archive;
  archive.load_from(path, torch::kCPU);
  net->load(archive);
  net->to(device);
  return net;
}

int64_t count_parameters(GoNet& net) {
  int64_t n = 0;
  for (const auto& p : net->parameters()) n += p.numel();
  return n;
}

}  // namespace nn
