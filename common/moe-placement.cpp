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
        if (!moe.devices.empty()) {
            throw std::invalid_argument("--moe-devices needs --moe-placement with a ranking file (analyze.py --emit-ranking)");
        }
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

    // hot devices: --moe-devices in the listed order, or one slot for the first model device
    std::vector<ggml_backend_dev_t> hot_devs;
    for (const auto & name : moe.devices) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_name(name.c_str());
        if (!dev || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            throw std::invalid_argument("--moe-devices: invalid device " + name + " (see --list-devices)");
        }
        hot_devs.push_back(dev);
    }
    const int n_slots = hot_devs.empty() ? 1 : (int) hot_devs.size();
    if (moe.vram_margin.size() != 1 && (int) moe.vram_margin.size() != n_slots) {
        throw std::invalid_argument("--moe-vram-margin: give one value or one per --moe-devices entry");
    }

    // probe: expert number s (by count) of every layer is hot on slot s, so every layer has all buckets as in the
    // final graph and the compute buffers of every device are measured
    std::map<int, std::vector<const common_moe_expert_rank *>> by_layer;
    for (const auto & r : R) {
        by_layer[r.layer].push_back(&r);
    }
    std::vector<int32_t> probe;
    std::vector<int32_t> probe_dev;
    std::vector<int64_t> probe_bytes(n_slots, 0);
    for (auto & [il, v] : by_layer) {
        std::stable_sort(v.begin(), v.end(), [](const auto * x, const auto * y) { return x->count > y->count; });
        for (int sl = 0; sl < n_slots && sl < (int) v.size(); sl++) {
            probe.push_back(il);
            probe.push_back(v[sl]->id);
            probe_dev.push_back(sl);
            probe_bytes[sl] += v[sl]->bytes;
        }
    }

    moe.dev_ptrs.clear();
    if (!hot_devs.empty()) {
        moe.dev_ptrs = hot_devs;
        moe.dev_ptrs.push_back(nullptr);
    }

    llama_model_params mp = mparams;
    mp.moe_placement = true;
    mp.moe_hot       = probe.data();
    mp.n_moe_hot     = probe.size() / 2;
    mp.moe_ram_pin   = false;
    mp.moe_warm      = nullptr;
    mp.n_moe_warm    = 0;
    mp.moe_devices   = moe.dev_ptrs.empty() ? nullptr : moe.dev_ptrs.data();
    mp.moe_hot_dev   = probe_dev.data();

    std::vector<ggml_backend_dev_t> devs;
    uint32_t hp_ngl = 0;
    uint32_t hp_nct = 0;
    uint32_t hp_nex = 0;
    const auto dmd = common_get_device_memory_data(params.model.path.c_str(), &mp, &cparams, devs, hp_ngl, hp_nct, hp_nex,
            params.verbosity >= LOG_LEVEL_DEBUG ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_ERROR);

    if (hot_devs.empty() && devs.size() > 1) {
        LOG_WRN("%s: %zu devices; the VRAM budget is only computed for %s (use --moe-devices for more)\n", __func__, devs.size(), ggml_backend_dev_name(devs[0]));
    }
    if (!devs.empty() && cparams.n_ctx == 0) {
        LOG_WRN("%s: n_ctx = 0 (training context); set -c to the context you use, the KV cache takes VRAM from the experts\n", __func__);
    }

    // budget per slot: free - (model - probe experts) - KV - compute - margin, from the device's share of the dry run
    std::vector<int64_t>     vram_budget(n_slots, 0);
    std::vector<std::string> slot_name(n_slots, "-");
    std::vector<std::string> budget_desc(n_slots, "no GPU device, all experts stay on the CPU");
    for (int sl = 0; sl < n_slots; sl++) {
        ggml_backend_dev_t dev = hot_devs.empty() ? (devs.empty() ? nullptr : devs[0]) : hot_devs[sl];
        if (!dev) {
            continue;
        }
        const auto it = std::find(devs.begin(), devs.end(), dev);
        if (it == devs.end()) {
            throw std::invalid_argument(std::string("--moe-devices: device ") + ggml_backend_dev_name(dev) + " is not used by the model (check -dev / --rpc)");
        }
        const auto & d = dmd[it - devs.begin()];
        const int64_t trunk  = (int64_t) d.model - probe_bytes[sl];
        const int64_t margin = moe.vram_margin[moe.vram_margin.size() == 1 ? 0 : sl];
        vram_budget[sl] = std::max<int64_t>(0, d.free - trunk - (int64_t) d.context - (int64_t) d.compute - margin);
        slot_name[sl]   = ggml_backend_dev_name(dev);
        budget_desc[sl] = string_format("%s: free %lld MiB - trunk %lld MiB - KV %lld MiB (n_ctx %u) - compute %lld MiB - margin %lld MiB = %lld MiB",
                slot_name[sl].c_str(), (long long) (d.free/MiB), (long long) (trunk/MiB), (long long) (d.context/MiB), cparams.n_ctx,
                (long long) (d.compute/MiB), (long long) (margin/MiB), (long long) (vram_budget[sl]/MiB));
    }

    // hot tiers: devices in order, each filled greedily from the hottest unassigned experts; an expert that does not
    // fit is skipped and smaller ones may still fit
    // assign: >= 0 hot slot, -1 cold, -2 RAM tier
    std::vector<int> assign(order.size(), -1);
    moe.hot.clear();
    moe.hot_dev.clear();
    for (int sl = 0; sl < n_slots; sl++) {
        int64_t used = 0;
        for (size_t i = 0; i < order.size(); i++) {
            const auto * r = order[i].first;
            if (assign[i] == -1 && used + r->bytes <= vram_budget[sl]) {
                used += r->bytes;
                assign[i] = sl;
                moe.hot.push_back(r->layer);
                moe.hot.push_back(r->id);
                moe.hot_dev.push_back(sl);
            }
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
            int64_t probe_all = 0;
            for (int64_t pb : probe_bytes) {
                probe_all += pb;
            }
            const int64_t host_other = std::max<int64_t>(0, (int64_t) h.model - (all_experts - probe_all)) + (int64_t) h.context + (int64_t) h.compute;
            const int64_t avail = host_available_memory();
            constexpr int64_t reserve = 2048*MiB;
            ram_budget = std::max<int64_t>(0, avail - host_other - reserve);
            ram_desc = string_format("auto: available %lld MiB - other host memory %lld MiB - reserve %lld MiB",
                    (long long) (avail/MiB), (long long) (host_other/MiB), (long long) (reserve/MiB));
        }
        int64_t ram_used = 0;
        for (size_t i = 0; i < order.size(); i++) {
            const auto * r = order[i].first;
            if (assign[i] == -1 && ram_used + r->bytes <= ram_budget) {
                ram_used += r->bytes;
                assign[i] = -2;
                moe.warm.push_back(r->layer);
                moe.warm.push_back(r->id);
            }
        }
    }

    // report
    struct tier { size_t n = 0; int64_t bytes = 0; int64_t count = 0; };
    std::vector<tier> hot_tiers(n_slots);
    tier warm_tier;
    tier cold_tier;
    for (size_t i = 0; i < order.size(); i++) {
        tier & t = assign[i] >= 0 ? hot_tiers[assign[i]] : assign[i] == -2 ? warm_tier : cold_tier;
        t.n++;
        t.bytes += order[i].first->bytes;
        t.count += order[i].first->count;
    }
    const double nsel = (double) std::max<int64_t>(moe.n_selections, 1);
    LOG_INF("%s: moe placement from ranking (profile %s, %lld selections, %zu experts)%s\n", __func__,
            moe.profile.c_str(), (long long) moe.n_selections, R.size(),
            hot_devs.empty() ? "" : " [EXPERIMENTAL multi-device, not tested on real multi-GPU]");
    for (int sl = 0; sl < n_slots; sl++) {
        LOG_INF("%s:   VRAM budget: %s\n", __func__, budget_desc[sl].c_str());
    }
    if (moe.ram_pin) {
        LOG_INF("%s:   RAM budget: %lld MiB (%s)\n", __func__, (long long) (ram_budget/MiB), ram_desc.c_str());
    }
    LOG_INF("%s:   %-20s %8s %12s %22s\n", __func__, "tier", "experts", "MiB", "share of selections");
    const char * fn = __func__;
    auto row = [&](const std::string & name, const tier & t) {
        LOG_INF("%s:   %-20s %8zu %12.1f %21.1f%%\n", fn, name.c_str(), t.n, t.bytes/(double) MiB, 100.0*t.count/nsel);
    };
    for (int sl = 0; sl < n_slots; sl++) {
        row(hot_devs.empty() ? std::string("VRAM (hot)") : "VRAM " + slot_name[sl], hot_tiers[sl]);
    }
    if (moe.ram_pin) {
        row("RAM (pinned)", warm_tier);
    }
    row(moe.ram_pin ? "disk (mmap)" : "RAM (cold)", cold_tier);

    mparams.moe_placement = true;
    mparams.moe_hot       = moe.hot.data();
    mparams.n_moe_hot     = moe.hot.size() / 2;
    mparams.moe_ram_pin   = moe.ram_pin;
    mparams.moe_warm      = moe.warm.data();
    mparams.n_moe_warm    = moe.warm.size() / 2;
    mparams.moe_devices   = moe.dev_ptrs.empty() ? nullptr : moe.dev_ptrs.data();
    mparams.moe_hot_dev   = moe.hot_dev.data();
}
