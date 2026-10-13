#include "nn/evaluator.h"

#include <cmath>
#include <iostream>

namespace nn {

using namespace go;

torch::Device pick_device(const std::string& name) {
  if (name == "cpu") return torch::Device(torch::kCPU);
  if (name.rfind("cuda", 0) == 0) {
    if (!torch::cuda::is_available()) {
      std::cerr << "[warn] CUDA requested but not available, falling back to CPU\n";
      return torch::Device(torch::kCPU);
    }
    return torch::Device(name);
  }
  return torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
}

Evaluator::Evaluator(GoNet net, torch::Device device) : net_(std::move(net)), device_(device) {
  net_->to(device_);
  net_->eval();
}

std::shared_ptr<Evaluator> Evaluator::create(const std::string& model_path, const torch::Device& device,
                                             const NetConfig& fallback) {
  if (!model_path.empty()) return std::make_shared<Evaluator>(load_model(model_path, device), device);
  return std::make_shared<Evaluator>(GoNet(fallback), device);
}

void Evaluator::set_net(GoNet net) {
  std::lock_guard<std::mutex> lock(mu_);
  net_ = std::move(net);
  net_->to(device_);
  net_->eval();
}

std::vector<Evaluation> Evaluator::evaluate(const std::vector<Snapshot>& positions, int sym) {
  const int64_t B = int64_t(positions.size());
  std::vector<Evaluation> results(positions.size());
  if (B == 0) return results;

  std::vector<int> syms(positions.size());
  auto input = torch::empty({B, kInputPlanes, N, N}, torch::kFloat32);
  float* in = input.data_ptr<float>();
  {
    std::lock_guard<std::mutex> lock(mu_);
    for (int64_t i = 0; i < B; ++i) syms[i] = sym >= 0 ? sym : int(rng_() % 8);
  }
  for (int64_t i = 0; i < B; ++i) encode(positions[i], syms[i], in + i * kInputSize);

  torch::Tensor logits, value, score, own;
  {
    std::lock_guard<std::mutex> lock(mu_);
    torch::NoGradGuard no_grad;
    auto out = net_->forward(input.to(device_, /*non_blocking=*/true));
    logits = out.policy_logits.to(torch::kCPU).contiguous();
    value = out.value.to(torch::kCPU).contiguous();
    score = out.score.to(torch::kCPU).contiguous();
    own = out.ownership.to(torch::kCPU).contiguous();
    evals_ += B;
  }
  const float* lp = logits.data_ptr<float>();
  const float* vp = value.data_ptr<float>();
  const float* sp = score.data_ptr<float>();
  const float* op = own.data_ptr<float>();

  for (int64_t i = 0; i < B; ++i) {
    Evaluation& e = results[i];
    const Snapshot& s = positions[i];
    const int sy = syms[i];
    // Network output is in the symmetric frame: map each real action to it.
    float maxl = -1e30f;
    for (int a = 0; a < NUM_ACTIONS; ++a) {
      if (!s.legal[a]) continue;
      maxl = std::max(maxl, lp[i * NUM_ACTIONS + sym_action(sy, a)]);
    }
    double total = 0;
    for (int a = 0; a < NUM_ACTIONS; ++a) {
      if (!s.legal[a]) {
        e.policy[a] = 0.f;
        continue;
      }
      float p = std::exp(lp[i * NUM_ACTIONS + sym_action(sy, a)] - maxl);
      e.policy[a] = p;
      total += p;
    }
    for (int a = 0; a < NUM_ACTIONS; ++a) e.policy[a] = total > 0 ? float(e.policy[a] / total) : 0.f;
    e.value = vp[i];
    e.score = sp[i] * 10.f;
    for (int a = 0; a < NN; ++a) e.ownership[a] = op[i * NN + sym_action(sy, a)];
  }
  return results;
}

Evaluation Evaluator::evaluate_all_syms(const Snapshot& position) {
  Evaluation avg{};
  avg.policy.fill(0.f);
  avg.ownership.fill(0.f);
  for (int s = 0; s < 8; ++s) {
    auto e = evaluate({position}, s)[0];
    for (int a = 0; a < NUM_ACTIONS; ++a) avg.policy[a] += e.policy[a] / 8.f;
    for (int a = 0; a < NN; ++a) avg.ownership[a] += e.ownership[a] / 8.f;
    avg.value += e.value / 8.f;
    avg.score += e.score / 8.f;
  }
  return avg;
}

}  // namespace nn
