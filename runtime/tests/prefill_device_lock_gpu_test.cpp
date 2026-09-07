// Regression for the model-level capture exclusion contract. Even an ineligible
// prefill must enter under device_mutex(), before inspecting session-owned state.
// No weights/checkpoint needed; full graph execution is covered by integration runs.
#include "sparkinfer/models/qwen35.h"
#include "sparkinfer/kv_cache.h"
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <future>
#include <mutex>

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::puts("[SKIP] prefill device lock needs a CUDA device");
        return 0;
    }
    sparkinfer::Qwen35Config cfg;
    cfg.vocab = 128; cfg.hidden = 128; cfg.n_layers = 1;
    cfg.n_q_heads = cfg.n_kv_heads = 1; cfg.head_dim = 128;
    cfg.n_experts = cfg.top_k = 1; cfg.n_shared = 0; cfg.moe_ffn = 128;
    cfg.max_seq = 128;
    sparkinfer::KVCacheConfig kc;
    kc.num_layers = kc.num_kv_heads = 1; kc.head_dim = 128; kc.block_size = 16;
    sparkinfer::KVCacheManager kv(kc, 4ull << 20);
    sparkinfer::Qwen35Model model(cfg, &kv, nullptr);
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return 1;
    std::unique_lock<std::recursive_mutex> held(model.device_mutex());
    std::promise<void> entered;
    auto entered_future = entered.get_future();
    auto call = std::async(std::launch::async, [&] {
        const auto err = cudaSetDevice(device);
        entered.set_value();
        if (err != cudaSuccess) return -2;
        return model.prefill_batched(nullptr, 0);
    });
    entered_future.wait();
    const bool excluded = call.wait_for(std::chrono::milliseconds(100)) == std::future_status::timeout;
    held.unlock();
    const int result = call.get();
    if (!excluded || result != -1) {
        std::fprintf(stderr, "[FAIL] prefill exclusion=%d result=%d\n", excluded, result);
        return 1;
    }
    held.lock();
    if (model.prefill_batched(nullptr, 0) != -1) return 1;
    held.unlock();
    std::puts("[PASS] prefill blocks behind device mutex; recursive entry returns");
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}
