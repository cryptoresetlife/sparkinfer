// Exercise the actual single/paired GEMV entry points, not a duplicate decoder.
// Four-nibble combinations are exhaustive in both halves of the packed word;
// one-hot activation rows expose each decoded value without rounding ambiguity.
#include "sparkinfer/kernels/gemm.h"
#include "sparkinfer/kernels/qtype.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

using namespace sparkinfer::kernels;
static void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
struct Buffer {
    void* data = nullptr;
    explicit Buffer(size_t bytes) { check(cudaMalloc(&data, bytes)); }
    ~Buffer() { cudaFree(data); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

static unsigned packed_code(unsigned n, bool negative) {
    return (n | ((65535u - n) << 16)) ^ (negative ? 0x88888888u : 0u);
}

static std::vector<unsigned char> weights(bool negative, int N, int K) {
    const size_t groups = size_t(N) * (K / 16);
    std::vector<unsigned char> w(SI_NVFP4_HDR + groups + size_t(N) * K / 2, 0);
    const float global_scale = 1.f;
    std::memcpy(w.data(), &global_scale, sizeof(global_scale));
    std::memset(w.data() + SI_NVFP4_HDR, 0x38, groups); // ue4m3(1)
    for (unsigned n = 0; n < N; ++n) {
        const unsigned p = packed_code(n, negative);
        for (int g = 0; g < K / 16; ++g) for (int b = 0; b < 4; ++b)
            w[SI_NVFP4_HDR + groups + size_t(n) * (K / 2) + g * 8 + b] = (p >> (8 * b)) & 255u;
    }
    return w;
}

static void verify(const Buffer& device, int rows, bool negative, int N, int K) {
    constexpr float magnitude[] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    std::vector<__nv_bfloat16> values(size_t(rows) * N);
    check(cudaMemcpy(values.data(), device.data, values.size() * sizeof(values[0]), cudaMemcpyDeviceToHost));
    for (int r = 0; r < rows; ++r) for (unsigned n = 0; n < N; ++n) {
        unsigned code = (packed_code(n, negative) >> (4 * r)) & 15u;
        float expected = magnitude[code & 7u] * ((code & 8u) ? -1.f : 1.f) * (K / 16);
        expected = __bfloat162float(__float2bfloat16(expected));
        float actual = __bfloat162float(values[size_t(r) * N + n]);
        if (actual != expected) {
            std::fprintf(stderr, "row=%d n=%u code=%u expected=%g got=%g\n", r, n, code, expected, actual);
            throw std::runtime_error("NVFP4 decode mismatch");
        }
    }
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::puts("[SKIP] no CUDA device");
        return 77;
    }
    try {
        // K=16 exhausts the decoder; 4112 exercises grouped trips plus the tail.
        // N=64 additionally exercises the S=8 split-K path.
        const int shapes[][2] = {{65536, 16}, {4096, 4112}, {64, 4112}};
        for (const auto& shape : shapes) {
        const int N = shape[0], K = shape[1];
        auto w0 = weights(false, N, K), w1 = weights(true, N, K);
        Buffer dw0(w0.size()), dw1(w1.size()), dx(8 * K), ds(8 * (K / 16) * sizeof(float));
        Buffer y0(size_t(8) * N * sizeof(__nv_bfloat16)), y1(size_t(8) * N * sizeof(__nv_bfloat16));
        std::vector<signed char> x(8 * K, 0);
        std::vector<float> scales(8 * (K / 16), 1.f);
        for (int r = 0; r < 8; ++r) for (int g = 0; g < K / 16; ++g)
            x[r * K + g * 16 + r] = 1;
        check(cudaMemcpy(dw0.data, w0.data(), w0.size(), cudaMemcpyHostToDevice));
        check(cudaMemcpy(dw1.data, w1.data(), w1.size(), cudaMemcpyHostToDevice));
        check(cudaMemcpy(dx.data, x.data(), x.size(), cudaMemcpyHostToDevice));
        check(cudaMemcpy(ds.data, scales.data(), scales.size() * sizeof(float), cudaMemcpyHostToDevice));
        for (int rows = 1; rows <= 8; ++rows) {
            if (!launch_gemv_nvfp4_rows_dp4a(dx.data, ds.data, dw0.data, y0.data, rows, N, K, nullptr))
                throw std::runtime_error("single launch declined");
            verify(y0, rows, false, N, K);
            if (!launch_gemv_nvfp4_rows_dp4a2(dx.data, ds.data, dw0.data, dw1.data,
                                           y0.data, y1.data, rows, N, K, nullptr))
                throw std::runtime_error("paired launch declined");
            verify(y0, rows, false, N, K);
            verify(y1, rows, true, N, K);
        }
        }
        std::puts("PASS: 65536 packed patterns, grouped/tail paths, single/paired GEMV, activation rows 1..8");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
