#include "llama-moe-residency.h"

#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

// second miss of a cold expert within this many ticks admits it for promotion; matches the setting measured in
// WEIGHT_PROVIDER.md "Phase 2a decision data" (section 8 of analyze.py). A tick is one ubatch by default, or one
// ubatch token with moe_dynamic_decode_window.
static constexpr int32_t ADMIT_WINDOW = 8;

llama_moe_residency::llama_moe_residency(const llama_model & model) {
    force_churn = getenv("LLAMA_MOE_DYNAMIC_FORCE_CHURN") != nullptr;

    if (const char * path = getenv("LLAMA_MOE_DYNAMIC_LOG")) {
        log_enabled = true;
        log_path    = path;
    }

    batch_threshold = model.moe_dynamic_batch_threshold();
    decode_window   = model.moe_dynamic_decode_window();
    bw_mbs          = model.moe_dynamic_bw_mbs();

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
        ls.miss_streak     = std::vector<int32_t>(ls.n_expert, 0);
        ls.in_flight       = std::vector<bool>(ls.n_expert, false);
        ls.last_promo_id   = std::vector<int64_t>(ls.n_expert, -1);
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

    LLAMA_LOG_INFO("%s: moe dynamic residency: %zu layers, device %s%s; batch_threshold=%d decode_window=%d bw_mbs=%.0f%s\n",
            __func__, layers_.size(), ggml_backend_dev_name(transfer_dev), force_churn ? " [FORCE CHURN]" : "",
            batch_threshold, (int) decode_window, bw_mbs, log_enabled ? " [EVENT LOG]" : "");
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
    if (log_enabled) {
        write_promo_log();
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

        if (j.promo_id >= 0) {
            std::lock_guard<std::mutex> lock(log_mu);
            log[j.promo_id].t_transfer_start_us = ggml_time_us();
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

        if (j.promo_id >= 0) {
            std::lock_guard<std::mutex> lock(log_mu);
            log[j.promo_id].t_transfer_end_us = ggml_time_us();
        }

        {
            std::lock_guard<std::mutex> lock(mu);
            done.push_back({ j.il, j.expert, j.slot, nb_gate + nb_up + nb_down, j.promo_id });
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

    if (c.promo_id >= 0) {
        std::lock_guard<std::mutex> lock(log_mu);
        log[c.promo_id].token_commit = total_tokens;
    }
}

int64_t llama_moe_residency::log_new_promo(int il, int32_t expert, bool phase_prefill, const char * reason,
        int32_t misses_in_window, int32_t victim_expert, int64_t victim_promo_id, size_t bytes) {
    std::lock_guard<std::mutex> lock(log_mu);
    const int64_t id = (int64_t) log.size();
    log.push_back({});
    promo_event & pe = log[id];
    pe.promo_id       = id;
    pe.layer           = il;
    pe.expert          = expert;
    pe.phase_prefill   = phase_prefill;
    pe.ubatch_size     = pending_n_tokens;
    pe.token_admit     = total_tokens;
    pe.reason          = reason;
    pe.misses_in_window = misses_in_window;
    pe.t_admit_us      = ggml_time_us();
    pe.victim_expert   = victim_expert;
    pe.victim_promo_id = victim_promo_id;
    pe.bytes           = bytes;
    return id;
}

void llama_moe_residency::log_evict(int64_t promo_id, int64_t token) {
    if (promo_id < 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(log_mu);
    promo_event & pe = log[promo_id];
    if (pe.token_evict < 0) {
        pe.token_evict   = token;
        pe.evict_reason  = "LRU";
    }
}

void llama_moe_residency::write_promo_log() const {
    FILE * f = fopen(log_path.c_str(), "w");
    if (!f) {
        LLAMA_LOG_WARN("%s: moe dynamic residency: failed to open LLAMA_MOE_DYNAMIC_LOG '%s'\n", __func__, log_path.c_str());
        return;
    }
    fprintf(f, "promo_id,layer,expert,phase,ubatch_size,token_admit,reason,misses_in_window,t_admit_us,"
            "t_transfer_start_us,t_transfer_end_us,token_commit,demands_while_pending,token_first_reuse,"
            "reuse_count,token_evict,evict_reason,victim_expert,victim_promo_id,re_misses_after_eviction,bytes\n");
    std::lock_guard<std::mutex> lock(log_mu);
    for (const auto & pe : log) {
        fprintf(f, "%lld,%d,%d,%s,%u,%lld,%s,%d,%lld,%lld,%lld,%lld,%d,%lld,%d,%lld,%s,%d,%lld,%d,%zu\n",
                (long long) pe.promo_id, pe.layer, pe.expert, pe.phase_prefill ? "prefill" : "decode",
                pe.ubatch_size, (long long) pe.token_admit, pe.reason, pe.misses_in_window,
                (long long) pe.t_admit_us, (long long) pe.t_transfer_start_us, (long long) pe.t_transfer_end_us,
                (long long) pe.token_commit, pe.demands_while_pending, (long long) pe.token_first_reuse,
                pe.reuse_count, (long long) pe.token_evict, pe.evict_reason, pe.victim_expert,
                (long long) pe.victim_promo_id, pe.re_misses_after_eviction, pe.bytes);
    }
    fclose(f);
    LLAMA_LOG_INFO("%s: moe dynamic residency: wrote %zu promotion events to '%s'\n", __func__, log.size(), log_path.c_str());
}

bool llama_moe_residency::admit(int il, int32_t expert, bool phase_prefill, const char * reason) {
    layer_state & ls = layers_[il];

    const size_t bytes = ls.hot_gate->nb[2] + ls.hot_up->nb[2] + ls.hot_down->nb[2];
    if (bw_mbs > 0.0f && (double) bytes > budget_bytes) {
        // over the byte budget for this boundary; try again on a later miss
        return false;
    }

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
        return false;
    }

    const int32_t victim_expert = ls.slot_expert[victim_slot];
    if (ls.slot_used_since_promote[victim_slot]) {
        n_useful_promotions++;
    }
    const int64_t victim_promo_id = ls.last_promo_id[victim_expert];
    if (log_enabled) {
        log_evict(victim_promo_id, total_tokens);
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

    int64_t promo_id = -1;
    if (log_enabled) {
        promo_id = log_new_promo(il, expert, phase_prefill, reason, ls.miss_streak[expert], victim_expert,
                victim_promo_id, bytes);
    }
    ls.last_promo_id[expert] = promo_id;

    if (bw_mbs > 0.0f) {
        budget_bytes -= (double) bytes;
    }

    {
        std::lock_guard<std::mutex> lock(mu);
        queue.push_back({ il, expert, victim_slot, promo_id });
    }
    cv.notify_one();
    return true;
}

void llama_moe_residency::boundary() {
    // 1. apply finished promotions
    std::vector<completion> finished;
    {
        std::lock_guard<std::mutex> lock(mu);
        finished.swap(done);
    }
    for (const auto & c : finished) {
        commit_promotion(c);
    }

    if (pending_n_tokens == 0) {
        return;
    }

    const bool phase_prefill = pending_n_tokens > 1;
    const bool admission_ok  = !(batch_threshold > 0 && (int32_t) pending_n_tokens > batch_threshold);

    total_tokens += pending_n_tokens;
    tick += decode_window ? (int32_t) pending_n_tokens : 1;

    const int64_t now = ggml_time_us();
    if (bw_mbs > 0.0f) {
        const double dt_s = last_boundary_time_us >= 0 ? (now - last_boundary_time_us) / 1e6 : 0.0;
        budget_bytes = (double) bw_mbs * 1e6 * dt_s;
    }
    last_boundary_time_us = now;

    // 2. process the routing collected after the previous graph_compute, in two passes so that admissions decided
    // from this ubatch's own misses never change whether an earlier entry of the same ubatch counts as a hit or a
    // miss (see the session notes: interleaving them undercounted hits badly on a large prefill ubatch, since an
    // expert evicted by entry #500 would wrongly show entries #1..499 of the same expert as misses, even though
    // the actual graph read it hot throughout)
    for (size_t il = 0; il < layers_.size(); ++il) {
        layer_state & ls = layers_[il];
        if (ls.n_slots == 0) {
            continue;
        }

        std::vector<int32_t> candidates;
        std::vector<bool> queued_candidate(ls.n_expert, false);

        for (int32_t e : ls.pending_routing) {
            const int32_t slot = ls.expert_slot[e];
            if (slot >= 0) {
                // hit
                if (phase_prefill) n_hits_prefill++; else n_hits_decode++;
                ls.slot_last_use[slot] = tick;
                ls.slot_used_since_promote[slot] = true;
                if (log_enabled && ls.last_promo_id[e] >= 0) {
                    std::lock_guard<std::mutex> lock(log_mu);
                    promo_event & pe = log[ls.last_promo_id[e]];
                    if (pe.token_commit >= 0) {
                        if (pe.token_first_reuse < 0) {
                            pe.token_first_reuse = total_tokens;
                        }
                        pe.reuse_count++;
                    }
                }
                continue;
            }

            // miss
            if (phase_prefill) n_misses_prefill++; else n_misses_decode++;
            if (log_enabled && ls.last_promo_id[e] >= 0) {
                std::lock_guard<std::mutex> lock(log_mu);
                promo_event & pe = log[ls.last_promo_id[e]];
                if (pe.token_commit < 0) {
                    pe.demands_while_pending++;
                } else if (pe.token_evict >= 0) {
                    pe.re_misses_after_eviction++;
                }
            }

            if (ls.in_flight[e] || !admission_ok || queued_candidate[e]) {
                continue;
            }

            bool eligible;
            if (force_churn) {
                eligible = true;
                ls.miss_streak[e]++;
            } else {
                const int32_t last = ls.last_miss_tick[e];
                if (last < 0 || tick - last > ADMIT_WINDOW) {
                    ls.miss_streak[e] = 1; // fresh streak: outside the window (or never missed before)
                } else {
                    ls.miss_streak[e]++;
                }
                eligible = ls.miss_streak[e] >= 2;
            }
            ls.last_miss_tick[e] = tick;

            if (eligible) {
                candidates.push_back(e);
                queued_candidate[e] = true;
            }
        }

        for (int32_t e : candidates) {
            admit((int) il, e, phase_prefill, force_churn ? "force-churn" : "second-miss");
        }

        ls.pending_routing.clear();
    }

    if (!phase_prefill) {
        n_decode_tokens += pending_n_tokens;
    }
    pending_n_tokens = 0;
}

void llama_moe_residency::collect_routing(const llm_graph_result * res, uint32_t n_tokens) {
    pending_n_tokens = n_tokens;
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
    const uint64_t n_hits    = n_hits_prefill + n_hits_decode;
    const uint64_t n_misses  = n_misses_prefill + n_misses_decode;
    const double useful_pct  = n_promotions ? 100.0 * (double) n_useful_promotions / (double) n_promotions : 0.0;
    const double bytes_per_decode_token = n_decode_tokens ? (double) n_bytes_copied / (double) n_decode_tokens : 0.0;

    LLAMA_LOG_INFO("%s: moe dynamic residency: hits %lld (prefill %lld, decode %lld), misses %lld (prefill %lld, "
            "decode %lld), promotions %lld, useful promotions %lld (%.1f%%), bytes copied %lld, decode tokens %lld, "
            "bytes/decode token %.0f\n", __func__,
            (long long) n_hits, (long long) n_hits_prefill, (long long) n_hits_decode,
            (long long) n_misses, (long long) n_misses_prefill, (long long) n_misses_decode,
            (long long) n_promotions, (long long) n_useful_promotions, useful_pct,
            (long long) n_bytes_copied, (long long) n_decode_tokens, bytes_per_decode_token);
}
