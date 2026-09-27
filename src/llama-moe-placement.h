#pragma once

#include "llama-model-loader.h"

#include "ggml-cpp.h"

#include <map>
#include <string>
#include <utility>
#include <vector>

struct llama_layer_moe_placement;

// static hot/cold MoE expert placement (llama_model_params::moe_placement)
// the experts of each hot bucket (one per hot device) are read into smaller tensors with the same row layout, plus I32
// tables that map global to local expert ids
// with mmap the cold bucket is the merged ffn_{gate,up,down}_exps tensor itself, left in the file mapping (global ids,
// the experts of the other buckets skipped); without mmap the cold experts are copied into a smaller tensor
// with moe_ram_pin the moe_warm experts of the cold bucket are locked in RAM
struct llama_moe_placement {
    llama_moe_placement(const llama_model_params & params, int n_layer, int64_t n_expert);

    // number of hot device slots: the length of moe_devices, or 1 without it
    int n_hot_slots() const { return n_slots; }

    // hot_bufts: buffer types of each hot device slot (for its bucket and its id table), cold_bufts: for the cold bucket
    // layer_bufts: buffer types of the layer device if it is a GPU, else nullptr; its hot bucket goes first and the cold
    // id table and the bucket table go there, so the layer needs fewer graph splits
    // cpu_buft: for the id tables when the layer has no hot experts
    // cold_merged: gate, up, down merged tensors already created by the caller (mmap), or nullptr to copy the cold experts
    void create_layer(llama_model_loader & ml, llama_layer_moe_placement & pl, int il,
            const std::string & name_gate, const std::string & name_up, const std::string & name_down,
            const std::vector<const buft_list_t *> & hot_bufts, const buft_list_t * layer_bufts,
            const buft_list_t & cold_bufts, ggml_backend_buffer_type_t cpu_buft, ggml_tensor * const * cold_merged);

    // allocate the tensors created above; with no_alloc use dummy buffers
    void alloc(bool no_alloc, std::vector<std::pair<ggml_context_ptr, std::vector<ggml_backend_buffer_ptr>>> & ctxs_bufs);

    // copy the expert slices from the model file; must run before llama_model_loader::load_all_data unmaps the file
    void load(llama_model_loader & ml);

    // lock the moe_warm experts in RAM; after all tensor data is loaded
    void lock(llama_mlocks & mlocks);

    bool ram_pin() const { return ram_pin_; }

private:
    ggml_context * ctx_for_buft(ggml_backend_buffer_type_t buft);

    struct slice {
        ggml_tensor *        dst;
        std::string          src;
        std::vector<int32_t> experts; // global id of each local expert
        bool                 unmap;   // the merged source tensor is not used after this
    };

    struct table {
        ggml_tensor *        dst;
        std::vector<int32_t> data;
    };

    // a run of experts of a cold tensor to lock
    struct lock_range {
        const ggml_tensor * t;
        int64_t             first;
        int64_t             n;
    };

    int64_t n_expert;
    bool    ram_pin_;

    int n_slots = 1;

    std::vector<std::vector<int>>  hot_slot; // [n_layer][n_expert]: hot device slot, -1 = cold
    std::vector<std::vector<bool>> is_warm;  // [n_layer][n_expert]

    std::vector<slice>      slices;
    std::vector<table>      tables;
    std::vector<lock_range> locks;

    // summary
    size_t n_hot   = 0;
    size_t n_cold  = 0;
    size_t n_warm  = 0;
    size_t b_hot   = 0;
    size_t b_cold  = 0;
    size_t b_warm  = 0;

    std::vector<size_t>                     slot_n;
    std::vector<size_t>                     slot_bytes;
    std::vector<ggml_backend_buffer_type_t> slot_buft;

    std::map<ggml_backend_buffer_type_t, ggml_context_ptr> ctxs;
    size_t max_tensors;
};
