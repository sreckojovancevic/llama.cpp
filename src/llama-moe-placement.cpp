#include "llama-moe-placement.h"

#include "llama-impl.h"
#include "llama-model.h"

#include <cstring>
#include <set>
#include <stdexcept>

// first buffer type in the list that supports MUL_MAT_ID with a weight like w
static ggml_backend_buffer_type_t select_buft(const buft_list_t & bufts, const ggml_tensor * w_meta, bool skip) {
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead()*8,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx { ggml_init(params) };

    ggml_tensor * w   = ggml_new_tensor_3d(ctx.get(), w_meta->type, w_meta->ne[0], w_meta->ne[1], w_meta->ne[2]);
    ggml_tensor * b   = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, w->ne[0], 1, 512);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 8, 512);
    ggml_tensor * op  = ggml_mul_mat_id(ctx.get(), w, b, ids);
    ggml_mul_mat_id_set_skip(op, skip);

    for (const auto & [dev, buft] : bufts) {
        w->buffer = ggml_backend_buft_alloc_buffer(buft, 0);
        const bool ok = ggml_backend_dev_supports_op(dev, op);
        ggml_backend_buffer_free(w->buffer);
        w->buffer = nullptr;
        if (ok) {
            return buft;
        }
    }
    return nullptr;
}

// first buffer type in the list that supports GET_ROWS on an I32 id table
static ggml_backend_buffer_type_t select_table_buft(const buft_list_t & bufts, int64_t n_expert) {
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead()*4,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx { ggml_init(params) };

    ggml_tensor * t   = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 1, n_expert);
    ggml_tensor * ids = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 8*512);
    ggml_tensor * op  = ggml_get_rows(ctx.get(), t, ids);

    for (const auto & [dev, buft] : bufts) {
        t->buffer = ggml_backend_buft_alloc_buffer(buft, 0);
        const bool ok = ggml_backend_dev_supports_op(dev, op);
        ggml_backend_buffer_free(t->buffer);
        t->buffer = nullptr;
        if (ok) {
            return buft;
        }
    }
    return nullptr;
}

llama_moe_placement::llama_moe_placement(const int32_t * hot, size_t n_hot, int n_layer, int64_t n_expert)
    : n_expert(n_expert), is_hot(n_layer, std::vector<bool>(n_expert, false)), max_tensors((size_t) n_layer*9 + 1) {
    for (size_t i = 0; i < n_hot; ++i) {
        const int32_t il = hot[2*i + 0];
        const int32_t ie = hot[2*i + 1];
        if (il < 0 || il >= n_layer || ie < 0 || ie >= n_expert) {
            throw std::runtime_error(format("moe placement: invalid (layer, expert) = (%d, %d), model has %d layers and %d experts",
                        il, ie, n_layer, (int) n_expert));
        }
        is_hot[il][ie] = true;
    }
}

ggml_context * llama_moe_placement::ctx_for_buft(ggml_backend_buffer_type_t buft) {
    auto it = ctxs.find(buft);
    if (it != ctxs.end()) {
        return it->second.get();
    }
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead()*max_tensors,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        throw std::runtime_error("moe placement: failed to create ggml context");
    }
    ctxs.emplace(buft, ctx);
    return ctx;
}

void llama_moe_placement::create_layer(llama_model_loader & ml, llama_layer_moe_placement & pl, int il,
        const std::string & name_gate, const std::string & name_up, const std::string & name_down,
        const buft_list_t & hot_bufts, const buft_list_t & cold_bufts, ggml_backend_buffer_type_t cpu_buft) {
    std::vector<int32_t> hot_ids;
    std::vector<int32_t> cold_ids;

    std::vector<int32_t> ids_hot (n_expert);
    std::vector<int32_t> ids_cold(n_expert);
    std::vector<int32_t> bucket  (n_expert);

    for (int32_t e = 0; e < n_expert; ++e) {
        if (is_hot[il][e]) {
            ids_hot [e] = (int32_t) hot_ids.size();
            ids_cold[e] = -1;
            bucket  [e] = 0;
            hot_ids.push_back(e);
        } else {
            ids_hot [e] = 0;
            ids_cold[e] = (int32_t) cold_ids.size();
            bucket  [e] = 1;
            cold_ids.push_back(e);
        }
    }

    // the cold bucket skips the hot experts only when both buckets are used
    const bool cold_skip = !hot_ids.empty();

    auto make = [&](const std::string & name, ggml_tensor ** hot, ggml_tensor ** cold) {
        const ggml_tensor * meta = ml.require_tensor_meta(name);
        if (meta->ne[2] != n_expert || meta->ne[3] != 1) {
            throw std::runtime_error(format("moe placement: tensor '%s' has unexpected shape", name.c_str()));
        }

        ml.n_created++;
        ml.size_data -= ggml_nbytes(meta);

        auto add = [&](const std::vector<int32_t> & experts, const char * suffix, const buft_list_t & bufts, bool skip) -> ggml_tensor * {
            if (experts.empty()) {
                return nullptr;
            }
            ggml_tensor t_meta = *meta;
            t_meta.ne[2] = (int64_t) experts.size();
            t_meta.nb[3] = t_meta.nb[2]*t_meta.ne[2];

            ggml_backend_buffer_type_t buft = select_buft(bufts, &t_meta, skip);
            if (!buft) {
                throw std::runtime_error(format("moe placement: no buffer type for tensor '%s'", name.c_str()));
            }
            // as in llama_model_loader::create_tensor: no host buffer with mmap
            ggml_backend_dev_t buft_dev = ggml_backend_buft_get_device(buft);
            if (ml.use_mmap && buft_dev && buft == ggml_backend_dev_host_buffer_type(buft_dev)) {
                buft = cpu_buft;
            }
            ggml_tensor * t = ggml_dup_tensor(ctx_for_buft(buft), &t_meta);
            ggml_format_name(t, "%s.%s", name.c_str(), suffix);
            slices.push_back({ t, name, experts });
            return t;
        };

        *hot  = add(hot_ids,  "hot",  hot_bufts,  false);
        *cold = add(cold_ids, "cold", cold_bufts, cold_skip);
    };

    make(name_gate, &pl.gate_hot, &pl.gate_cold);
    make(name_up,   &pl.up_hot,   &pl.up_cold);
    make(name_down, &pl.down_hot, &pl.down_cold);

    // id tables next to the hot bucket (where the router runs when the layer is on the GPU), so the id mapping
    // adds no graph split; without hot experts the cold bucket uses them on the CPU
    ggml_backend_buffer_type_t table_buft = hot_ids.empty() ? nullptr : select_table_buft(hot_bufts, n_expert);
    if (!table_buft) {
        table_buft = cpu_buft;
    }

    auto add_table = [&](std::vector<int32_t> & data, const char * name) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx_for_buft(table_buft), GGML_TYPE_I32, 1, n_expert);
        ggml_format_name(t, "blk.%d.ffn_moe_%s", il, name);
        tables.push_back({ t, std::move(data) });
        return t;
    };

    pl.ids_hot  = add_table(ids_hot,  "ids_hot");
    pl.ids_cold = add_table(ids_cold, "ids_cold");
    pl.bucket   = add_table(bucket,   "bucket");

    LLAMA_LOG_DEBUG("%s: layer %d: %zu hot experts, %zu cold experts\n", __func__, il, hot_ids.size(), cold_ids.size());
}

void llama_moe_placement::alloc(bool no_alloc, std::vector<std::pair<ggml_context_ptr, std::vector<ggml_backend_buffer_ptr>>> & ctxs_bufs) {
    for (auto & [buft, ctx] : ctxs) {
        ggml_backend_buffer_t buf;
        if (no_alloc) {
            buf = ggml_backend_buft_alloc_buffer(buft, 0);
            for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
                t->buffer = buf;
            }
        } else {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft);
        }
        if (buf == nullptr) {
            throw std::runtime_error(format("moe placement: unable to allocate %s buffer", ggml_backend_buft_name(buft)));
        }
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        LLAMA_LOG_INFO("%s: %12s moe placement buffer size = %8.2f MiB\n", __func__,
                ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);

        std::vector<ggml_backend_buffer_ptr> bufs;
        bufs.emplace_back(buf);
        ctxs_bufs.emplace_back(std::move(ctx), std::move(bufs));
    }
    ctxs.clear();
}

void llama_moe_placement::load(llama_model_loader & ml) {
    std::vector<uint8_t> data;
    std::vector<uint8_t> read_buf;

    for (const auto & s : slices) {
        const auto & w = ml.require_weight(s.src.c_str());
        const size_t nb2 = w.tensor->nb[2];

        // whole tensor at once: repacking buffer types only accept full writes
        data.resize(ggml_nbytes(s.dst));
        read_buf.resize(ml.use_mmap ? 0 : nb2);
        for (size_t j = 0; j < s.experts.size(); ++j) {
            const void * src = ml.load_data_range(w, (size_t) s.experts[j]*nb2, nb2, read_buf.data());
            memcpy(data.data() + j*nb2, src, nb2);
        }
        ggml_backend_tensor_set(s.dst, data.data(), 0, data.size());
    }

    // release the pages of the merged tensors, they are not used after this
    if (ml.use_mmap) {
        std::set<std::string> done;
        for (const auto & s : slices) {
            if (done.insert(s.src).second) {
                ml.unmap_weight(ml.require_weight(s.src.c_str()));
            }
        }
    }

    for (const auto & t : tables) {
        ggml_backend_tensor_set(t.dst, t.data.data(), 0, ggml_nbytes(t.dst));
    }

    slices.clear();
    tables.clear();
}
