// Exercise real dense prefill with a small deterministic BF16 hybrid model.
// No checkpoint download. Repeated passes cover arena reuse / graph replay,
// changed prompt lengths, GDN and full attention, and multi-chunk raw FFN output.
#include "../src/models/qwen35_prefill.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

using namespace sparkinfer;

static void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}

struct DeviceStorage {
    std::vector<void*> owned;
    ~DeviceStorage() { for (void* p : owned) cudaFree(p); }
    template<class T> T* alloc(size_t n) {
        T* p = nullptr;
        check(cudaMalloc(&p, n * sizeof(T)));
        owned.push_back(p);
        check(cudaMemset(p, 0, n * sizeof(T)));
        return p;
    }
    void* bf16(size_t n, float scale, bool ones = false) {
        std::vector<unsigned short> host(n);
        for (size_t i = 0; i < n; ++i) {
            const float f = ones ? 1.f : scale * (int((i * 1664525u + 1013904223u) % 1009) - 504) / 504.f;
            unsigned bits;
            std::memcpy(&bits, &f, sizeof(bits));
            host[i] = static_cast<unsigned short>(bits >> 16);
        }
        void* p = alloc<unsigned short>(n);
        check(cudaMemcpy(p, host.data(), n * sizeof(host[0]), cudaMemcpyHostToDevice));
        return p;
    }
};

int main() try {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) {
        std::puts("[SKIP] dense prefill scratch needs a CUDA device");
        return 0;
    }
    // Test the raw-output fallbacks too, not just the checkpoint-native fused
    // residual paths. Avoid atomics in this repeatability test.
    setenv("SPARKINFER_PREFILL_I8", "0", 1);
    setenv("SPARKINFER_PREFILL_SKINNY_SPLITK", "0", 1);
    setenv("SPARKINFER_PREFILL_FFN_CHUNK", "1024", 1);
    DeviceStorage mem;
    Qwen35Config c;
    c.vocab = 512; c.hidden = 512; c.n_layers = 2;
    c.n_q_heads = 2; c.n_kv_heads = 1; c.head_dim = 256;
    c.n_experts = c.top_k = 1; c.n_shared = 0; c.moe_ffn = 768;
    c.hybrid = c.dense_ffn = true; c.full_attn_interval = 2;
    c.linear_q_heads = 2; c.linear_v_heads = 4; c.linear_head_dim = 128;
    c.max_seq = 4096;
    const int H = c.hidden, Q = 512, K = 256, LQ = 256, LV = 512, LQKV = 1024;
    Qwen35Weights w;
    w.embed_tokens = mem.bf16(c.vocab * H, .3f);
    w.lm_head = mem.bf16(c.vocab * H, .02f);
    w.final_norm = mem.bf16(H, 0, true);
    w.layers.resize(c.n_layers);
    for (int layer = 0; layer < c.n_layers; ++layer) {
        auto& t = w.layers[layer];
        t.input_norm = t.post_attn_norm = mem.bf16(H, 0, true);
        t.gate_q = mem.bf16(H * c.moe_ffn, .02f);
        t.up_q = mem.bf16(H * c.moe_ffn, .03f);
        t.down_q = mem.bf16(H * c.moe_ffn, .01f);
        t.linear_attn = layer == 0;
        if (t.linear_attn) {
            t.wqkv = mem.bf16(H * LQKV, .03f);
            t.wqkv_gate = mem.bf16(H * LV, .02f);
            t.ssm_conv = mem.bf16(c.linear_conv_kernel * LQKV, .1f);
            t.ssm_alpha = t.ssm_beta = mem.bf16(H * c.linear_v_heads, .01f);
            t.ssm_dt = mem.bf16(c.linear_v_heads, .1f);
            auto* decay = mem.alloc<unsigned short>(c.linear_v_heads);
            const std::vector<unsigned short> negative_one(c.linear_v_heads, 0xbf80);
            check(cudaMemcpy(decay, negative_one.data(), negative_one.size() * sizeof(unsigned short),
                             cudaMemcpyHostToDevice));
            t.ssm_a = decay;
            t.ssm_norm = mem.bf16(c.linear_head_dim, 0, true);
            t.ssm_out = mem.bf16(LV * H, .02f);
        } else {
            t.q_has_gate = true;
            // The shared prefill requires wide=2*Q >= LQKV.
            t.wq = mem.bf16(H * LQKV, .03f);
            t.wk = t.wv = mem.bf16(H * K, .02f);
            t.wo = mem.bf16(Q * H, .02f);
            t.q_norm = t.k_norm = mem.bf16(c.head_dim, 0, true);
        }
    }
    KVCacheConfig kc{c.n_layers, c.n_kv_heads, c.head_dim};
    kc.layer_slot = hybrid_kv_layer_slots(c.n_layers, true, c.full_attn_interval);
    KVCacheManager kv(kc, 32ull << 20);
    if (!kv.allocate(1, c.max_seq)) throw std::runtime_error("KV allocation failed");
    cudaStream_t stream;
    check(cudaStreamCreate(&stream));
    int* host_id = nullptr;
    check(cudaMallocHost(&host_id, sizeof(int)));
    const size_t state_n = (size_t)c.n_layers * c.linear_v_heads * 128 * 128;
    auto* state = mem.alloc<float>(state_n);
    auto* conv = mem.alloc<unsigned short>((size_t)c.n_layers * (c.linear_conv_kernel - 1) * LQKV);
    auto* logits = mem.alloc<float>(c.vocab);
    auto* out_id = mem.alloc<int>(1);
    Qwen35PrefillCtx ctx{c, w, &kv, stream, nullptr, nullptr, 1, state, conv,
                         logits, out_id, host_id, true, nullptr, Q, K, LQ, LV, LQKV,
                         nullptr, nullptr, nullptr, 1, nullptr, 0, nullptr, 0};
    for (int n : {128, 2048, 1025, 128}) {
        std::vector<int> ids(n);
        for (int i = 0; i < n; ++i) ids[i] = (i * 17 + 3) % c.vocab;
        std::vector<float> reference;
        for (int rep = 0; rep < 3; ++rep) {
            const int seed = prefill_batched_run(ctx, ids.data(), n);
            check(cudaDeviceSynchronize());
            if (seed < 0 || seed >= c.vocab) throw std::runtime_error("invalid prefill seed");
            std::vector<float> values(c.vocab + state_n);
            check(cudaMemcpy(values.data(), logits, c.vocab * sizeof(float), cudaMemcpyDeviceToHost));
            check(cudaMemcpy(values.data() + c.vocab, state, state_n * sizeof(float), cudaMemcpyDeviceToHost));
            bool nonzero = false;
            for (float v : values) {
                if (!std::isfinite(v)) throw std::runtime_error("nonfinite prefill output");
                nonzero |= v != 0;
            }
            if (!nonzero) throw std::runtime_error("empty prefill output");
            if (rep && std::memcmp(reference.data(), values.data(), values.size() * sizeof(float)))
                throw std::runtime_error("repeated prefill changed logits or GDN state");
            reference = values;
        }
        // Emit a reproducible fingerprint for before/after comparisons of the
        // same fixture against separately built runtime libraries.
        unsigned long long hash = 14695981039346656037ull;
        for (unsigned char b : std::vector<unsigned char>(
                 reinterpret_cast<unsigned char*>(reference.data()),
                 reinterpret_cast<unsigned char*>(reference.data() + reference.size())))
            hash = (hash ^ b) * 1099511628211ull;
        std::printf("[PASS] n=%d repeated logits/state hash=%016llx\n", n, hash);
    }
    check(cudaFreeHost(host_id));
    check(cudaStreamDestroy(stream));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "[FAIL] dense prefill scratch: %s\n", e.what());
    return 1;
}
