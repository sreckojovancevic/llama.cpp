#include "moe-placement.h"

#include "fit.h"
#include "gguf.h"
#include "json.h"
#include "log.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

using json = common_json;

static constexpr int64_t MiB = 1024*1024;

int64_t common_parse_size(const std::string & s) {
    size_t pos = 0;
    double v = 0.0;
    try {
        v = std::stod(s, &pos);
    } catch (const std::exception &) {
        throw std::invalid_argument("invalid size: " + s);
    }
    std::string suf = s.substr(pos);
    std::transform(suf.begin(), suf.end(), suf.begin(), [](unsigned char c) { return (char) std::toupper(c); });
    int64_t mult;
    if (suf.empty() || suf == "B") {
        mult = 1;
    } else if (suf == "K" || suf == "KB" || suf == "KIB") {
        mult = 1ll << 10;
    } else if (suf == "M" || suf == "MB" || suf == "MIB") {
        mult = 1ll << 20;
    } else if (suf == "G" || suf == "GB" || suf == "GIB") {
        mult = 1ll << 30;
    } else {
        throw std::invalid_argument("invalid size suffix: " + s);
    }
    if (v < 0) {
        throw std::invalid_argument("negative size: " + s);
    }
    return (int64_t) std::llround(v*mult);
}

void common_moe_placement_load_file(const std::string & path, common_params_moe_placement & moe) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("failed to open moe placement file: " + path);
    }
    std::stringstream ss;
    ss << f.rdbuf();
    const json j = json::parse(ss.str());

    moe.enabled = true;
    moe.hot.clear();
    moe.ranking.clear();
    moe.profile      = j.contains("profile") ? j.at("profile").get<std::string>() : "";
    moe.n_selections = 0;

    const json & layers = j.at("layers");
    for (size_t i = 0; i < layers.size(); i++) {
        const json & layer = layers.at(i);
        const int il = layer.at("layer").get<int>();
        if (layer.contains("experts")) {
            const json & experts = layer.at("experts");
            for (size_t k = 0; k < experts.size(); k++) {
                const json & e = experts.at(k);
                common_moe_expert_rank r;
                r.layer = il;
                r.id    = e.at("id").get<int>();
                r.count = e.at("count").get<long long>();
                r.bytes = e.at("bytes").get<long long>();
                moe.ranking.push_back(r);
                moe.n_selections += r.count;
            }
        } else {
            for (int e : layer.at("hot").get<std::vector<int>>()) {
                moe.hot.push_back(il);
                moe.hot.push_back(e);
            }
        }
    }
    if (!moe.ranking.empty() && !moe.hot.empty()) {
        throw std::runtime_error("moe placement file mixes \"hot\" and \"experts\" layers: " + path);
    }
}

// available host memory: Linux MemAvailable, Windows min(available physical, available commit)
static int64_t host_available_memory() {
#if defined(_WIN32)
    MEMORYSTATUSEX st;
    st.dwLength = sizeof(st);
    if (GlobalMemoryStatusEx(&st)) {
        return (int64_t) std::min(st.ullAvailPhys, st.ullAvailPageFile);
    }
#else
    std::ifstream f("/proc/meminfo");
    std::string key;
    long long kb = 0;
    std::string unit;
    while (f >> key >> kb >> unit) {
        if (key == "MemAvailable:") {
            return (int64_t) kb*1024;
        }
    }
#endif
    size_t free  = 0;
    size_t total = 0;
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu) {
        ggml_backend_dev_memory(cpu, &free, &total);
    }
    return (int64_t) free;
}

// bytes of one expert (gate + up + down) per layer from the model file; layers whose tensors are not in this file are missing
static std::map<int, int64_t> expert_bytes_from_gguf(const std::string & path) {
    std::map<int, int64_t> ret;
    gguf_init_params ip = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context * ctx = gguf_init_from_file(path.c_str(), ip);
    if (!ctx) {
        return ret;
    }
    const int64_t kid_arch = gguf_find_key(ctx, "general.architecture");
    const std::string arch = kid_arch >= 0 ? gguf_get_val_str(ctx, kid_arch) : "";
    const int64_t kid_nexp = gguf_find_key(ctx, (arch + ".expert_count").c_str());
    const int64_t kid_nblk = gguf_find_key(ctx, (arch + ".block_count").c_str());
    const int64_t n_expert = kid_nexp >= 0 ? (int64_t) gguf_get_val_u32(ctx, kid_nexp) : 0;
    const int64_t n_block  = kid_nblk >= 0 ? (int64_t) gguf_get_val_u32(ctx, kid_nblk) : 0;
    for (int il = 0; il < n_block && n_expert > 0; il++) {
        int64_t sum = 0;
        bool found = true;
        for (const char * t : { "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps" }) {
            const std::string name = "blk." + std::to_string(il) + "." + t + ".weight";
            const int64_t id = gguf_find_tensor(ctx, name.c_str());
            if (id < 0) {
                found = false;
                break;
            }
            sum += (int64_t) gguf_get_tensor_size(ctx, id);
        }
        if (found) {
            ret[il] = sum / n_expert;
        }
    }
    gguf_free(ctx);
    return ret;
}

void common_moe_placement_resolve(common_params & params, llama_model_params & mparams, const llama_context_params & cparams) {
    common_params_moe_placement & moe = params.moe;
    if (!moe.enabled || moe.ranking.empty()) {
        if (moe.enabled && moe.ram_pin) {
            LOG_WRN("%s: --moe-ram-pin needs a ranking file (analyze.py --emit-ranking), ignored\n", __func__);
            moe.ram_pin = false;
            mparams.moe_ram_pin = false;
        }
        return;
    }

    auto & R = moe.ranking;

    // expert bytes: trust the model file over the ranking file
    const auto gguf_bytes = expert_bytes_from_gguf(params.model.path);
    bool mismatch = false;
    for (auto & r : R) {
        auto it = gguf_bytes.find(r.layer);
        if (it != gguf_bytes.end()) {
            mismatch |= it->second != r.bytes;
            r.bytes = it->second;
        }
    }
    if (mismatch) {
        LOG_WRN("%s: expert sizes in the ranking file differ from the model file, the ranking may be from another model or quant\n", __func__);
    }

    // order: count per byte, then unseen experts spread over the layers (as analyze.py section 4)
    std::map<int, int> unseen_in_layer;
    std::vector<std::pair<const common_moe_expert_rank *, int>> order;
    for (const auto & r : R) {
        order.push_back({ &r, r.count == 0 ? ++unseen_in_layer[r.layer] : 0 });
    }
    std::stable_sort(order.begin(), order.end(), [](const auto & a, const auto & b) {
        const double sa = (double) a.first->count / (double) std::max<int64_t>(a.first->bytes, 1);
        const double sb = (double) b.first->count / (double) std::max<int64_t>(b.first->bytes, 1);
        if (sa != sb) {
            return sa > sb;
        }
        return a.second < b.second;
    });

    // probe: the top expert of each layer is hot, so every layer has both buckets as in the final graph
    std::map<int, const common_moe_expert_rank *> top;
    for (const auto & r : R) {
        auto & t = top[r.layer];
        if (!t || r.count > t->count) {
            t = &r;
        }
    }
    std::vector<int32_t> probe;
    int64_t probe_bytes = 0;
    for (const auto & [il, r] : top) {
        probe.push_back(il);
        probe.push_back(r->id);
        probe_bytes += r->bytes;
    }

    llama_model_params mp = mparams;
    mp.moe_placement = true;
    mp.moe_hot       = probe.data();
    mp.n_moe_hot     = probe.size() / 2;
    mp.moe_ram_pin   = false;
    mp.moe_warm      = nullptr;
    mp.n_moe_warm    = 0;

    std::vector<ggml_backend_dev_t> devs;
    uint32_t hp_ngl = 0;
    uint32_t hp_nct = 0;
    uint32_t hp_nex = 0;
    const auto dmd = common_get_device_memory_data(params.model.path.c_str(), &mp, &cparams, devs, hp_ngl, hp_nct, hp_nex,
            params.verbosity >= LOG_LEVEL_DEBUG ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_ERROR);

    int64_t vram_budget = 0;
    std::string budget_desc = "no GPU device, all experts stay on the CPU";
    if (!devs.empty()) {
        const auto & d = dmd[0];
        const int64_t trunk = (int64_t) d.model - probe_bytes;
        vram_budget = std::max<int64_t>(0, d.free - trunk - (int64_t) d.context - (int64_t) d.compute - moe.vram_margin);
        budget_desc = string_format("%s: free %lld MiB - trunk %lld MiB - KV %lld MiB (n_ctx %u) - compute %lld MiB - margin %lld MiB = %lld MiB",
                ggml_backend_dev_name(devs[0]), (long long) (d.free/MiB), (long long) (trunk/MiB), (long long) (d.context/MiB), cparams.n_ctx,
                (long long) (d.compute/MiB), (long long) (moe.vram_margin/MiB), (long long) (vram_budget/MiB));
        if (devs.size() > 1) {
            LOG_WRN("%s: %zu devices; the VRAM budget is only computed for %s\n", __func__, devs.size(), ggml_backend_dev_name(devs[0]));
        }
        if (cparams.n_ctx == 0) {
            LOG_WRN("%s: n_ctx = 0 (training context); set -c to the context you use, the KV cache takes VRAM from the experts\n", __func__);
        }
    }

    // hot tier: greedy, an expert that does not fit is skipped and smaller ones may still fit
    std::vector<char> is_hot(order.size(), 0);
    int64_t used = 0;
    moe.hot.clear();
    for (size_t i = 0; i < order.size(); i++) {
        const auto * r = order[i].first;
        if (used + r->bytes <= vram_budget) {
            used += r->bytes;
            is_hot[i] = 1;
            moe.hot.push_back(r->layer);
            moe.hot.push_back(r->id);
        }
    }

    // RAM tier
    moe.warm.clear();
    int64_t ram_budget = 0;
    std::string ram_desc;
    if (moe.ram_pin) {
        if (moe.ram_pin_bytes >= 0) {
            ram_budget = moe.ram_pin_bytes;
            ram_desc   = "set by --moe-ram-pin";
        } else {
            // host memory of the model without the cold experts (the probe copied them), KV and compute on the host
            int64_t all_experts = 0;
            for (const auto & r : R) {
                all_experts += r.bytes;
            }
            const auto & h = dmd.back();
            const int64_t host_other = std::max<int64_t>(0, (int64_t) h.model - (all_experts - probe_bytes)) + (int64_t) h.context + (int64_t) h.compute;
            const int64_t avail = host_available_memory();
            constexpr int64_t reserve = 2048*MiB;
            ram_budget = std::max<int64_t>(0, avail - host_other - reserve);
            ram_desc = string_format("auto: available %lld MiB - other host memory %lld MiB - reserve %lld MiB",
                    (long long) (avail/MiB), (long long) (host_other/MiB), (long long) (reserve/MiB));
        }
        int64_t ram_used = 0;
        for (size_t i = 0; i < order.size(); i++) {
            const auto * r = order[i].first;
            if (!is_hot[i] && ram_used + r->bytes <= ram_budget) {
                ram_used += r->bytes;
                is_hot[i] = 2;
                moe.warm.push_back(r->layer);
                moe.warm.push_back(r->id);
            }
        }
    }

    // report
    struct tier { size_t n = 0; int64_t bytes = 0; int64_t count = 0; };
    tier tiers[3];
    for (size_t i = 0; i < order.size(); i++) {
        const int t = is_hot[i] == 1 ? 0 : is_hot[i] == 2 ? 1 : 2;
        tiers[t].n++;
        tiers[t].bytes += order[i].first->bytes;
        tiers[t].count += order[i].first->count;
    }
    const double nsel = (double) std::max<int64_t>(moe.n_selections, 1);
    LOG_INF("%s: moe placement from ranking (profile %s, %lld selections, %zu experts)\n", __func__,
            moe.profile.c_str(), (long long) moe.n_selections, R.size());
    LOG_INF("%s:   VRAM budget: %s\n", __func__, budget_desc.c_str());
    if (moe.ram_pin) {
        LOG_INF("%s:   RAM budget: %lld MiB (%s)\n", __func__, (long long) (ram_budget/MiB), ram_desc.c_str());
    }
    LOG_INF("%s:   %-12s %8s %12s %22s\n", __func__, "tier", "experts", "MiB", "share of selections");
    const char * names[3] = { "VRAM (hot)", "RAM (pinned)", moe.ram_pin ? "disk (mmap)" : "RAM (cold)" };
    for (int t = 0; t < 3; t++) {
        if (t == 1 && !moe.ram_pin) {
            continue;
        }
        LOG_INF("%s:   %-12s %8zu %12.1f %21.1f%%\n", __func__, names[t], tiers[t].n, tiers[t].bytes/(double) MiB, 100.0*tiers[t].count/nsel);
    }

    mparams.moe_placement = true;
    mparams.moe_hot       = moe.hot.data();
    mparams.n_moe_hot     = moe.hot.size() / 2;
    mparams.moe_ram_pin   = moe.ram_pin;
    mparams.moe_warm      = moe.warm.data();
    mparams.n_moe_warm    = moe.warm.size() / 2;
}
