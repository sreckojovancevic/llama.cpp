#pragma once

#include "llama-model-loader.h"

#include "ggml-cpp.h"

#include <map>
#include <string>
#include <utility>
#include <vector>

struct llama_layer_moe_placement;

// static hot/cold MoE expert placement (llama_model_params::moe_placement)
// the merged ffn_{gate,up,down}_exps tensors of a layer are not loaded; instead the hot and the cold experts are
// read into two smaller tensors with the same row layout, plus I32 tables that map global to local expert ids
struct llama_moe_placement {
    llama_moe_placement(const int32_t * hot, size_t n_hot, int n_layer, int64_t n_expert);

    // hot_bufts: buffer types for the hot bucket, cold_bufts: for the cold bucket, remap_buft: for the id tables
    void create_layer(llama_model_loader & ml, llama_layer_moe_placement & pl, int il,
            const std::string & name_gate, const std::string & name_up, const std::string & name_down,
            const buft_list_t & hot_bufts, const buft_list_t & cold_bufts, ggml_backend_buffer_type_t remap_buft);

    // allocate the tensors created above; with no_alloc use dummy buffers
    void alloc(bool no_alloc, std::vector<std::pair<ggml_context_ptr, std::vector<ggml_backend_buffer_ptr>>> & ctxs_bufs);

    // copy the expert slices from the model file; must run before llama_model_loader::load_all_data unmaps the file
    void load(llama_model_loader & ml);

private:
    ggml_context * ctx_for_buft(ggml_backend_buffer_type_t buft);

    struct slice {
        ggml_tensor *        dst;
        std::string          src;
        std::vector<int32_t> experts; // global id of each local expert
    };

    struct table {
        ggml_tensor *        dst;
        std::vector<int32_t> data;
    };

    int64_t n_expert;

    std::vector<std::vector<bool>> is_hot; // [n_layer][n_expert]

    std::vector<slice> slices;
    std::vector<table> tables;

    std::map<ggml_backend_buffer_type_t, ggml_context_ptr> ctxs;
    size_t max_tensors;
};
