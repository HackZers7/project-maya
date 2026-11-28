// src/hy4/hy4_ablate.cuh - HY4_ABLATE=<file>: refusal directions projected out of the trunk's hyper-connection
// streams (research; off unless the variable is set).  alesha-pro's hy4f HY4F_ABLATE (Maya's hy4-flash branch)
// ported onto maya-hy4.
//
// The file is an int32 header {points, hc, k, E} and then points * hc * k * E float32 values: point p holds stream
// s's k orthonormal vectors at V + (p * hc + s) * k * E.  The points are the streams the graph reads directly:
// 0..points - 2 the stream entering the layer's attention (that layer's hc_pre), points - 1 the stream leaving the
// last one (the head's hc_head).  At every point each stream loses its components along its own vectors,
// x -= scale (x . v) v (k::hc_ablate in hy4_kernels.cu), so the refusal direction stays out of what every layer
// reads - Hy4's four streams carry it differently, so the file has its own vectors for every layer and every stream
// (the weights stay stock).  The MTP draft block runs on plain residuals and is left alone (a draft only has to be a
// good guess).  HY4_ABLATE_SCALE (default 1) is the strength.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

namespace hy4_ablate {

struct State {
    bool on = false;
    int points = 0, hc = 0, k = 0, E = 0;   // the file's header, checked against the model's geometry at the load
    float scale = 1.0f;
    std::vector<float> host;
    std::mutex mu;
    float* dev[64] = {};
};

inline State& state() {
    static State s;
    return s;
}

inline bool on() { return state().on; }

/// Reads HY4_ABLATE (the file) and checks its header against the model's: points = layers + 1, the model's streams
/// and hidden size.  false: the file is not what this model needs - the load fails (the engine would run the stock
/// model silently otherwise).  A previous setup's device copies are freed: a re-read (the trial load, a changed
/// file) starts the table over.
inline bool setup(int points, int hc, int E, std::string& err) {
    State& a = state();
    a.on = false;
    for (int d = 0; d < 64; ++d)
        if (a.dev[d] != nullptr && cudaSetDevice(d) == cudaSuccess) {
            cudaFree(a.dev[d]);
            a.dev[d] = nullptr;
        }
    const char* path = getenv("HY4_ABLATE");
    if (path == nullptr || !path[0]) return true;
    std::FILE* f = std::fopen(path, "rb");
    if (f == nullptr) {
        err = std::string("hy4 ablate: cannot open HY4_ABLATE=") + path;
        return false;
    }
    int32_t h[4] = {0, 0, 0, 0};
    bool ok = std::fread(h, sizeof(h), 1, f) == 1 && h[0] == points && h[1] == hc && h[2] > 0 && h[2] <= 64 &&
              h[3] == E;
    if (ok) {
        a.host.resize((size_t) h[0] * h[1] * h[2] * h[3]);
        ok = std::fread(a.host.data(), sizeof(float), a.host.size(), f) == a.host.size();
    }
    std::fclose(f);
    if (!ok) {
        err = std::string("hy4 ablate: ") + path + ": wrong header or size (layers + 1, hc, k, E); the model needs " +
              std::to_string(points) + ", " + std::to_string(hc) + ", k, " + std::to_string(E);
        return false;
    }
    a.points = h[0];
    a.hc = h[1];
    a.k = h[2];
    a.E = h[3];
    if (const char* sc = getenv("HY4_ABLATE_SCALE")) a.scale = (float) std::atof(sc);
    a.on = true;
    std::fprintf(stderr, "hy4 ablate: %d directions per stream at %d points from %s, scale %.2f\n", a.k, a.points,
                 path, a.scale);
    return true;
}

/// The file's rows of `point` on the CURRENT device (uploaded on first use), or nullptr when off or of no point.
inline const float* vec(int point) {
    State& a = state();
    if (!a.on || point < 0 || point >= a.points) return nullptr;
    int d = 0;
    if (cudaGetDevice(&d) != cudaSuccess || d < 0 || d >= 64) return nullptr;
    std::lock_guard<std::mutex> lk(a.mu);
    if (a.dev[d] == nullptr) {
        float* p = nullptr;
        if (cudaMalloc((void**) &p, a.host.size() * sizeof(float)) != cudaSuccess ||
            cudaMemcpy(p, a.host.data(), a.host.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
            std::fprintf(stderr, "hy4 ablate: no memory for the directions on CUDA%d - ablation OFF\n", d);
            a.on = false;
            return nullptr;
        }
        a.dev[d] = p;
    }
    return a.dev[d] + (size_t) point * a.hc * a.k * a.E;
}

}  // namespace hy4_ablate
