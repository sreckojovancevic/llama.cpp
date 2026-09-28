#pragma once

#include "ggml-cpp.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
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
    // for use at the next boundary() call
    void collect_routing(const llm_graph_result * res);

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
        std::vector<bool>    in_flight;                // [n_expert]: a promotion for this expert is queued/running

        std::vector<int32_t> pending_routing;          // ids selected in the last collected ubatch, read at boundary()
    };

    struct job {
        int     il;
        int32_t expert;
        int32_t slot;
    };

    struct completion {
        int     il;
        int32_t expert;
        int32_t slot;
        size_t  bytes;
    };

    void commit_promotion(const completion & c);
    void admit(int il, int32_t expert);
    void worker_main();

    std::vector<layer_state> layers_;
    int32_t tick = 0;

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

    // counters, printed by log_counters()
    uint64_t n_hits             = 0;
    uint64_t n_misses           = 0;
    uint64_t n_promotions       = 0;
    uint64_t n_useful_promotions = 0;
    uint64_t n_bytes_copied     = 0;
};
