#include "llama-moe-residency.h"

#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

// second miss of a cold expert within this many boundary() calls (ubatches) admits it for promotion; matches the
// setting measured in WEIGHT_PROVIDER.md "Phase 2a decision data" (section 8 of analyze.py)
static constexpr int32_t ADMIT_WINDOW = 8;

llama_moe_residency::llama_moe_residency(const llama_model & model) {
    force_churn = getenv("LLAMA_MOE_DYNAMIC_FORCE_CHURN") != nullptr;

    ggml_backend_dev_t transfer_dev = nullptr;

    for (size_t il = 0; il < model.layers.size(); ++il) {
        const auto & pl = model.layers[il].moe_pl;
        if (!pl.dynamic) {
            continue;
        }
        if (pl.buckets.size() != 2) {
            throw std::runtime_error("moe residency: expected one hot and one cold bucket");
        }

        const llama_moe_bucket & hot  = pl.buckets[0];
        const llama_moe_bucket & cold = pl.buckets[1];
        if (!hot.gate || !hot.up || !hot.down || !hot.ids ||
            !cold.gate || !cold.up || !cold.down || !cold.ids || !cold.ids0) {
            throw std::runtime_error("moe residency: incomplete bucket tensors");
        }

        ggml_backend_buffer_type_t hot_buft = ggml_backend_buffer_get_type(hot.gate->buffer);
        ggml_backend_dev_t         hot_dev  = ggml_backend_buft_get_device(hot_buft);
        if (!transfer_dev) {
            transfer_dev = hot_dev;
        } else if (hot_dev != transfer_dev) {
            throw std::runtime_error("moe residency: hot buckets of different layers are on different devices, "
                    "moe_dynamic supports a single hot device only");
        }

        layers_.resize(std::max(layers_.size(), il + 1));
        layer_state & ls = layers_[il];

        ls.hot_gate  = hot.gate;
        ls.hot_up    = hot.up;
        ls.hot_down  = hot.down;
        ls.hot_ids   = hot.ids;
        ls.cold_gate = cold.gate;
        ls.cold_up   = cold.up;
        ls.cold_down = cold.down;
        ls.cold_ids  = cold.ids;
        ls.cold_ids0 = cold.ids0;
        ls.bucket    = pl.bucket;

        ls.n_slots  = (int32_t) hot.gate->ne[2];
        ls.n_expert = cold.gate->ne[2];

        const auto & initial_hot = model.moe_dynamic_initial_hot((int) il);
        if ((int32_t) initial_hot.size() != ls.n_slots) {
            throw std::runtime_error("moe residency: warm-start hot set size does not match the hot bucket");
        }

        ls.slot_expert            = initial_hot;
        ls.slot_last_use          = std::vector<int32_t>(ls.n_slots, 0);
        ls.slot_pending           = std::vector<bool>(ls.n_slots, false);
        ls.slot_used_since_promote = std::vector<bool>(ls.n_slots, false);

        ls.expert_slot     = std::vector<int32_t>(ls.n_expert, -1);
        ls.last_miss_tick  = std::vector<int32_t>(ls.n_expert, -1);
        ls.in_flight       = std::vector<bool>(ls.n_expert, false);
        ls.pending_routing.clear();

        for (int32_t s = 0; s < ls.n_slots; ++s) {
            ls.expert_slot[ls.slot_expert[s]] = s;
        }
    }

    if (layers_.empty() || !transfer_dev) {
        throw std::runtime_error("moe residency: no dynamic layer found");
    }

    transfer_backend = ggml_backend_dev_init(transfer_dev, nullptr);
    if (!transfer_backend) {
        throw std::runtime_error("moe residency: failed to init the transfer backend");
    }

    // one pinned (or, without a host buffer type, plain) staging region, sized for the largest expert of any layer
    size_t max_gate = 0, max_up = 0, max_down = 0;
    for (const auto & ls : layers_) {
        if (!ls.hot_gate) {
            continue;
        }
        max_gate = std::max(max_gate, (size_t) ls.hot_gate->nb[2]);
        max_up   = std::max(max_up,   (size_t) ls.hot_up->nb[2]);
        max_down = std::max(max_down, (size_t) ls.hot_down->nb[2]);
    }

    // must be genuinely host-accessible (the worker thread memcpy's the cold expert into it directly, then hands
    // it to set_tensor_async): the hot device's own buffer type is wrong here when that device is remote (RPC) or
    // discrete (CUDA) memory, unlike its *host* buffer type (pinned memory on a real GPU, plain CPU memory when
    // the device has no such concept, e.g. RPC)
    ggml_backend_buffer_type_t staging_buft = ggml_backend_dev_host_buffer_type(transfer_dev);
    if (!staging_buft) {
        staging_buft = ggml_backend_dev_buffer_type(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU));
    }
    staging_buf.reset(ggml_backend_buft_alloc_buffer(staging_buft, max_gate + max_up + max_down));
    if (!staging_buf) {
        throw std::runtime_error("moe residency: failed to allocate the staging buffer");
    }
    uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(staging_buf.get());
    staging_gate = base;
    staging_up   = base + max_gate;
    staging_down = base + max_gate + max_up;

    worker = std::thread(&llama_moe_residency::worker_main, this);

    LLAMA_LOG_INFO("%s: moe dynamic residency: %zu layers, device %s%s\n", __func__, layers_.size(),
            ggml_backend_dev_name(transfer_dev), force_churn ? " [FORCE CHURN]" : "");
}

llama_moe_residency::~llama_moe_residency() {
    {
        std::lock_guard<std::mutex> lock(mu);
        stop_worker = true;
    }
    cv.notify_all();
    if (worker.joinable()) {
        worker.join();
    }
    if (transfer_backend) {
        ggml_backend_free(transfer_backend);
    }
    log_counters();
}

void llama_moe_residency::worker_main() {
    for (;;) {
        job j;
        {
            std::unique_lock<std::mutex> lock(mu);
            cv.wait(lock, [&] { return stop_worker || !queue.empty(); });
            if (stop_worker && queue.empty()) {
                return;
            }
            j = queue.front();
            queue.pop_front();
        }

        layer_state & ls = layers_[j.il];
        const size_t nb_gate = ls.hot_gate->nb[2];
        const size_t nb_up   = ls.hot_up->nb[2];
        const size_t nb_down = ls.hot_down->nb[2];

        // read from the cold (mmap) tensor on this thread, so page faults of the disk tier land here, not on the
        // main thread; then async H2D on a stream of our own, separate from the compute stream
        memcpy(staging_gate, (const uint8_t *) ls.cold_gate->data + (size_t) j.expert*nb_gate, nb_gate);
        memcpy(staging_up,   (const uint8_t *) ls.cold_up->data   + (size_t) j.expert*nb_up,   nb_up);
        memcpy(staging_down, (const uint8_t *) ls.cold_down->data + (size_t) j.expert*nb_down, nb_down);

        ggml_backend_tensor_set_async(transfer_backend, ls.hot_gate, staging_gate, (size_t) j.slot*nb_gate, nb_gate);
        ggml_backend_tensor_set_async(transfer_backend, ls.hot_up,   staging_up,   (size_t) j.slot*nb_up,   nb_up);
        ggml_backend_tensor_set_async(transfer_backend, ls.hot_down, staging_down, (size_t) j.slot*nb_down, nb_down);
        ggml_backend_synchronize(transfer_backend);

        {
            std::lock_guard<std::mutex> lock(mu);
            done.push_back({ j.il, j.expert, j.slot, nb_gate + nb_up + nb_down });
        }
    }
}

// table update only (PHASE2_REVIEW.md section 9/16): flip the bucket/cold-id (and, on eviction, ids_hot) entries
// of a promoted or evicted expert
static void set_i32(ggml_tensor * t, int64_t idx, int32_t value) {
    ggml_backend_tensor_set(t, &value, (size_t) idx*sizeof(int32_t), sizeof(int32_t));
}

void llama_moe_residency::commit_promotion(const completion & c) {
    layer_state & ls = layers_[c.il];

    set_i32(ls.hot_ids,   c.expert, c.slot);
    set_i32(ls.bucket,    c.expert, 0); // bucket 0 = hot
    set_i32(ls.cold_ids,  c.expert, -1); // skip the cold FFN for this expert from now on
    set_i32(ls.cold_ids0, c.expert, 0);

    ls.expert_slot[c.expert]        = c.slot;
    ls.slot_pending[c.slot]         = false;
    ls.slot_used_since_promote[c.slot] = false;
    ls.in_flight[c.expert]          = false;

    n_promotions++;
    n_bytes_copied += c.bytes;
}

void llama_moe_residency::admit(int il, int32_t expert) {
    layer_state & ls = layers_[il];

    // victim: least recently used non-pinned, non-pending slot; slot 0 is always pinned
    int32_t victim_slot = -1;
    int32_t victim_use  = 0;
    for (int32_t s = 1; s < ls.n_slots; ++s) {
        if (ls.slot_pending[s]) {
            continue;
        }
        if (victim_slot < 0 || ls.slot_last_use[s] < victim_use) {
            victim_slot = s;
            victim_use  = ls.slot_last_use[s];
        }
    }
    if (victim_slot < 0) {
        // every non-pinned slot has a promotion in flight; try again on a later miss
        return;
    }

    const int32_t victim_expert = ls.slot_expert[victim_slot];
    if (ls.slot_used_since_promote[victim_slot]) {
        n_useful_promotions++;
    }

    set_i32(ls.bucket,    victim_expert, 1); // bucket 1 = cold
    set_i32(ls.cold_ids,  victim_expert, victim_expert);
    set_i32(ls.cold_ids0, victim_expert, victim_expert);
    // reset to the same dummy (slot 0, pinned) as every other cold expert: without this, a graph that selects
    // victim_expert again before its next promotion would still read victim_slot for the discarded one-slot-per-row
    // computation (PHASE2_REVIEW.md section 10/Q3), racing with the worker thread about to overwrite that slot
    set_i32(ls.hot_ids,   victim_expert, 0);
    ls.expert_slot[victim_expert] = -1;

    ls.slot_expert[victim_slot]  = expert;
    ls.slot_pending[victim_slot] = true;
    ls.slot_last_use[victim_slot] = tick;
    ls.in_flight[expert] = true;

    {
        std::lock_guard<std::mutex> lock(mu);
        queue.push_back({ il, expert, victim_slot });
    }
    cv.notify_one();
}

void llama_moe_residency::boundary() {
    ++tick;

    // 1. apply finished promotions
    std::vector<completion> finished;
    {
        std::lock_guard<std::mutex> lock(mu);
        finished.swap(done);
    }
    for (const auto & c : finished) {
        commit_promotion(c);
    }

    // 2. process the routing collected after the previous graph_compute
    for (size_t il = 0; il < layers_.size(); ++il) {
        layer_state & ls = layers_[il];
        if (ls.n_slots == 0) {
            continue;
        }
        for (int32_t e : ls.pending_routing) {
            const int32_t slot = ls.expert_slot[e];
            if (slot >= 0) {
                n_hits++;
                ls.slot_last_use[slot] = tick;
                ls.slot_used_since_promote[slot] = true;
                continue;
            }

            n_misses++;
            if (ls.in_flight[e]) {
                continue;
            }
            if (force_churn) {
                admit((int) il, e);
                continue;
            }
            const int32_t last = ls.last_miss_tick[e];
            ls.last_miss_tick[e] = tick;
            if (last >= 0 && tick - last <= ADMIT_WINDOW) {
                admit((int) il, e);
            }
        }
        ls.pending_routing.clear();
    }
}

void llama_moe_residency::collect_routing(const llm_graph_result * res) {
    for (size_t il = 0; il < layers_.size(); ++il) {
        layer_state & ls = layers_[il];
        if (ls.n_slots == 0) {
            continue;
        }
        if (il >= res->t_moe_ids.size() || !res->t_moe_ids[il]) {
            continue;
        }
        const ggml_tensor * t = res->t_moe_ids[il];
        const int64_t n = ggml_nelements(t);
        ls.pending_routing.resize(n);
        ggml_backend_tensor_get(t, ls.pending_routing.data(), 0, n*sizeof(int32_t));
    }
}

void llama_moe_residency::log_counters() const {
    LLAMA_LOG_INFO("%s: moe dynamic residency: hits %lld, misses %lld, promotions %lld, useful promotions %lld, "
            "bytes copied %lld\n", __func__,
            (long long) n_hits, (long long) n_misses, (long long) n_promotions,
            (long long) n_useful_promotions, (long long) n_bytes_copied);
}
