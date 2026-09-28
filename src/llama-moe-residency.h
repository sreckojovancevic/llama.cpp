#pragma once

#include "ggml-cpp.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct llama_model;
class llm_graph_result;

// Phase 2a: dynamic MoE expert residency on top of the static hot/cold placement (llama_moe_placement).
//
// The hot bucket of a dynamic layer is a fixed-size set of VRAM slots (llama_moe_placement::initial_hot at warm
// start). Slot 0 is pinned and never evicted (build_moe_ffn reads it as a dummy for every non-hot token). A miss
// (an expert selected by routing that is not currently hot) is admitted for promotion on its second occurrence
// within a short window; an admitted promotion evicts the least-recently-used non-pinned slot (a table update
// only, see PHASE2_REVIEW.md section 9) and copies the expert's weights into that slot on a worker thread, through
// a pinned staging buffer, on a backend instance of its own (a second ggml_backend_t for the same device, so the
// copy does not share a stream with compute). The new residency only becomes visible - the hot/cold/bucket tables
// are only written - at a ubatch boundary, and only once the copy has finished; see llama_context::process_ubatch.
//
// Admission policy switches (each off by default, so the base behavior is reproducible without them - see
// llama_model_params for details): moe_dynamic_batch_threshold (no admission from large ubatches),
// moe_dynamic_decode_window (count the window in ubatch tokens, not one tick per ubatch) and moe_dynamic_bw_mbs
// (a per-boundary byte budget). "Phase" (prefill vs decode) for the hit/miss counters and the promotion log is
// always ubatch tokens == 1, independent of these switches.
//
// See PHASE2_REVIEW.md for the full design and the answers to PHASE2_DESIGN.md's review questions.
struct llama_moe_residency {
    explicit llama_moe_residency(const llama_model & model);
    ~llama_moe_residency();

    llama_moe_residency(const llama_moe_residency &) = delete;
    llama_moe_residency & operator=(const llama_moe_residency &) = delete;

    // call at the start of llama_context::process_ubatch, after ggml_backend_sched_synchronize: applies finished
    // promotions, processes the routing of the previous ubatch (miss accounting, admission, eviction) and starts
    // new promotions on the worker thread
    void boundary();

    // call right after a successful graph_compute: reads back the routing of this ubatch (llm_graph_result::t_moe_ids)
    // for use at the next boundary() call; n_tokens is the ubatch's token count (phase = n_tokens == 1)
    void collect_routing(const llm_graph_result * res, uint32_t n_tokens);

    void log_counters() const;

private:
    struct layer_state {
        // device tensors (owned by the model, persistent for its lifetime)
        ggml_tensor * hot_gate = nullptr, * hot_up = nullptr, * hot_down = nullptr, * hot_ids = nullptr;
        ggml_tensor * cold_gate = nullptr, * cold_up = nullptr, * cold_down = nullptr;
        ggml_tensor * cold_ids = nullptr, * cold_ids0 = nullptr; // flagged / unflagged (offloadable) form
        ggml_tensor * bucket = nullptr;

        int32_t n_slots = 0;
        int64_t n_expert = 0;

        std::vector<int32_t> slot_expert;             // [n_slots]: global expert id resident (or promoting) in each slot
        std::vector<int32_t> slot_last_use;            // [n_slots]: LRU tick
        std::vector<bool>    slot_pending;             // [n_slots]: promotion queued/in flight into this slot
        std::vector<bool>    slot_used_since_promote;  // [n_slots]: hit at least once since its last promotion

        std::vector<int32_t> expert_slot;              // [n_expert]: slot of a resident expert, -1 = cold or pending
        std::vector<int32_t> last_miss_tick;           // [n_expert]: tick of the previous miss, -1 = none
        std::vector<int32_t> miss_streak;              // [n_expert]: consecutive misses since the window last reset
        std::vector<bool>    in_flight;                // [n_expert]: a promotion for this expert is queued/running
        std::vector<int64_t> last_promo_id;             // [n_expert]: promo_event index of the last promotion, -1 = none

        std::vector<int32_t> pending_routing;          // ids selected in the last collected ubatch, read at boundary()
    };

    struct job {
        int     il;
        int32_t expert;
        int32_t slot;
        int64_t promo_id; // -1 when the event log is off
    };

    struct completion {
        int     il;
        int32_t expert;
        int32_t slot;
        size_t  bytes;
        int64_t promo_id;
    };

    // one row of the LLAMA_MOE_DYNAMIC_LOG=<path.csv> event log (env var, opt-in); see tools/expert-trace/promo-report.py
    struct promo_event {
        int64_t     promo_id = -1;
        int         layer = -1;
        int32_t     expert = -1;
        bool        phase_prefill = false; // ubatch tokens > 1 at admission
        uint32_t    ubatch_size = 0;       // ubatch tokens at admission
        int64_t     token_admit = -1;      // total ubatch tokens processed so far, at admission
        const char * reason = "";          // "second-miss" | "force-churn"
        int32_t     misses_in_window = 0;

        int64_t t_admit_us = 0;
        int64_t t_transfer_start_us = 0;
        int64_t t_transfer_end_us = 0;
        int64_t token_commit = -1;

        int32_t demands_while_pending = 0; // misses on `expert` between admission and commit

        int64_t token_first_reuse = -1;    // first hit after commit
        int32_t reuse_count = 0;           // hits after commit, before eviction ("useful" iff > 0)

        int64_t     token_evict = -1;
        const char * evict_reason = "";    // "LRU"
        int32_t     victim_expert = -1;    // expert this promotion evicted
        int64_t     victim_promo_id = -1;  // that expert's own promo_id, -1 if it was a warm-start resident

        int32_t re_misses_after_eviction = 0; // misses on `expert` after its own eviction

        size_t bytes = 0;
    };

    void commit_promotion(const completion & c);
    // returns false (does nothing) if there is no evictable slot or the byte budget does not allow it
    bool admit(int il, int32_t expert, bool phase_prefill, const char * reason);
    void worker_main();

    int64_t log_new_promo(int il, int32_t expert, bool phase_prefill, const char * reason, int32_t misses_in_window,
            int32_t victim_expert, int64_t victim_promo_id, size_t bytes);
    void log_evict(int64_t promo_id, int64_t token);
    void write_promo_log() const;

    std::vector<layer_state> layers_;
    int32_t tick = 0;
    int64_t total_tokens = 0; // ubatch tokens processed so far, all phases; used for log timestamps only
    uint32_t pending_n_tokens = 0; // token count of the ubatch whose routing is in pending_routing, set by collect_routing

    // worker thread: one pinned staging buffer and one backend instance dedicated to promotion transfers, separate
    // from the compute backend/stream
    ggml_backend_t         transfer_backend = nullptr;
    ggml_backend_buffer_ptr staging_buf;
    uint8_t * staging_gate = nullptr;
    uint8_t * staging_up   = nullptr;
    uint8_t * staging_down = nullptr;

    std::thread             worker;
    std::mutex               mu;
    std::condition_variable  cv;
    bool                     stop_worker = false;
    std::deque<job>          queue;
    std::vector<completion>  done;

    bool force_churn = false; // LLAMA_MOE_DYNAMIC_FORCE_CHURN=1: admit every miss immediately (for testing)

    // admission policy switches, see llama_model_params for what they do; each 0/false = off (base behavior)
    int32_t batch_threshold = 0;
    bool    decode_window = false;
    float   bw_mbs = 0.0f;
    int64_t last_boundary_time_us = -1;
    double  budget_bytes = 0.0;

    // event log (LLAMA_MOE_DYNAMIC_LOG), off unless the env var is set
    bool                     log_enabled = false;
    std::string              log_path;
    mutable std::mutex       log_mu;
    std::vector<promo_event> log;

    // counters, printed by log_counters()
    uint64_t n_hits_prefill      = 0;
    uint64_t n_misses_prefill    = 0;
    uint64_t n_hits_decode       = 0;
    uint64_t n_misses_decode     = 0;
    uint64_t n_decode_tokens     = 0;
    uint64_t n_promotions        = 0;
    uint64_t n_useful_promotions = 0;
    uint64_t n_bytes_copied      = 0;
};
