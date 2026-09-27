// time of one layer's cold expert FFN (up, gate, swiglu, down with the MUL_MAT_ID skip flag) on the CPU backend at
// decode (1 token), for 0..8 active experts, Qwen3-30B-A3B shapes; shows the fixed cost per layer against the cost per
// expert. usage: llama-cold-ffn-bench [repack] [n_threads ...]
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static double bench(ggml_backend_t be, ggml_cgraph * gf, int iters) {
    for (int i = 0; i < 20; i++) ggml_backend_graph_compute(be, gf);
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; i++) ggml_backend_graph_compute(be, gf);
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
}

int main(int argc, char ** argv) {
    bool repack = false;
    std::vector<int> threads;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "repack") == 0) {
            repack = true;
        } else {
            threads.push_back(std::stoi(argv[i]));
        }
    }
    if (threads.empty()) {
        threads = { 1, 2, 4 };
    }
    ggml_backend_load_all();
    const int64_t n_embd = 2048, n_ff = 768, n_exp = 128, K = 8;
    ggml_init_params ip = { ggml_tensor_overhead()*64 + ggml_graph_overhead()*4, NULL, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * up   = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_K, n_embd, n_ff, n_exp);
    ggml_tensor * gate = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_K, n_embd, n_ff, n_exp);
    ggml_tensor * down = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_K, n_ff, n_embd, n_exp);
    ggml_context * ctx2 = ggml_init(ip);
    ggml_tensor * x    = ggml_new_tensor_3d(ctx2, GGML_TYPE_F32, n_embd, 1, 1);
    ggml_tensor * ids  = ggml_new_tensor_2d(ctx2, GGML_TYPE_I32, K, 1);

    ggml_backend_t be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_dev_t dev = ggml_backend_get_device(be);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    auto set_n_threads = (void (*)(ggml_backend_t, int)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    ggml_backend_buffer_type_t wbuft = ggml_backend_dev_buffer_type(dev);
    if (repack) {
        auto get_extra = (ggml_backend_buffer_type_t *(*)(ggml_backend_dev_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
        if (get_extra && get_extra(dev) && get_extra(dev)[0]) {
            wbuft = get_extra(dev)[0];
        }
    }
    printf("weights in %s\n", ggml_backend_buft_name(wbuft));
    ggml_backend_alloc_ctx_tensors_from_buft(ctx, wbuft);
    ggml_backend_alloc_ctx_tensors(ctx2, be);

    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0, 0.02f);
    for (ggml_tensor * w : { up, gate, down }) {
        std::vector<float> f(ggml_nelements(w));
        for (auto & v : f) v = nd(rng);
        std::vector<uint8_t> q(ggml_nbytes(w));
        ggml_quantize_chunk(w->type, f.data(), q.data(), 0, ggml_nrows(w), w->ne[0], nullptr);
        ggml_backend_tensor_set(w, q.data(), 0, q.size());
    }
    std::vector<float> xv(n_embd);
    for (auto & v : xv) v = nd(rng);
    ggml_backend_tensor_set(x, xv.data(), 0, ggml_nbytes(x));

    // cold FFN graph, as build_moe_ffn: up, gate, swiglu, down with skip flag
    auto make_graph = [&](ggml_context * c, bool skip) {
        ggml_tensor * u = ggml_mul_mat_id(c, up, x, ids);   ggml_mul_mat_id_set_skip(u, skip);
        ggml_tensor * g = ggml_mul_mat_id(c, gate, x, ids); ggml_mul_mat_id_set_skip(g, skip);
        ggml_tensor * a = ggml_swiglu_split(c, g, u);
        ggml_tensor * d = ggml_mul_mat_id(c, down, a, ids); ggml_mul_mat_id_set_skip(d, skip);
        ggml_cgraph * gf = ggml_new_graph(c);
        ggml_build_forward_expand(gf, d);
        return gf;
    };
    ggml_init_params ip3 = { ggml_tensor_overhead()*64 + ggml_graph_overhead()*4, NULL, true };
    ggml_context * cg = ggml_init(ip3);
    ggml_cgraph * gf = make_graph(cg, true);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    ggml_gallocr_alloc_graph(ga, gf);

    // empty-ish graph: one tiny op, to see the launch and barrier floor
    ggml_context * ce = ggml_init(ip3);
    ggml_cgraph * ge = ggml_new_graph(ce);
    ggml_build_forward_expand(ge, ggml_scale(ce, x, 1.0f));
    ggml_gallocr_t ga2 = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    ggml_gallocr_alloc_graph(ga2, ge);

    printf("expert bytes (up+gate+down) %.2f MiB\n", 3.0*ggml_nbytes(up)/n_exp/1024/1024);
    printf("%8s %8s %12s %12s\n", "threads", "active", "us/layer", "GB/s");
    for (int nt : threads) {
        if (set_n_threads) {
            set_n_threads(be, nt);
        }
        printf("%8d %8s %12.1f %12s   (1-op graph)\n", nt, "-", bench(be, ge, 2000), "-");
        for (int k : { 0, 1, 2, 3, 4, 8 }) {
            std::vector<int32_t> idv(K, -1);
            for (int i = 0; i < k; i++) idv[i] = (int32_t) (i * 13 % n_exp);
            ggml_backend_tensor_set(ids, idv.data(), 0, ggml_nbytes(ids));
            const double us = bench(be, gf, 300);
            const double bytes = k * 3.0 * ggml_nbytes(up) / n_exp;
            printf("%8d %8d %12.1f %12.2f\n", nt, k, us, k ? bytes / (us * 1e3) : 0.0);
        }
    }
    return 0;
}
