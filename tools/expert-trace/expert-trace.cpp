#include "arg.h"
#include "common.h"
#include "gguf.h"
#include "log.h"
#include "sampling.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

// name set in build_moe_ffn: cb(selected_experts, "ffn_moe_topk", il) -> "ffn_moe_topk-<il>"
static const char * TOPK_PREFIX = "ffn_moe_topk-";

struct trace_state {
    std::ofstream out;
    std::vector<char> buf;

    int64_t ubatch     = -1;
    int     last_layer = INT32_MAX;
    int64_t tok_base   = 0; // global index of first token in current ubatch
    int64_t n_tok_cur  = 0; // tokens in current ubatch
    int64_t n_rows     = 0;
    int64_t n_expert_used = 0;
    std::map<int, int64_t> rows_per_layer;
};

static bool trace_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    const bool is_topk = strncmp(t->name, TOPK_PREFIX, strlen(TOPK_PREFIX)) == 0;
    if (ask) {
        return is_topk;
    }
    if (!is_topk) {
        return true;
    }

    auto * st = (trace_state *) user_data;

    GGML_ASSERT(t->type == GGML_TYPE_I32);
    const int layer = atoi(t->name + strlen(TOPK_PREFIX));
    const int64_t n_used = t->ne[0];
    const int64_t n_tok  = t->ne[1];

    // layers come in increasing order within one graph, so a non-increasing layer means a new ubatch
    if (layer <= st->last_layer) {
        st->ubatch++;
        st->tok_base += st->n_tok_cur;
        st->n_tok_cur = n_tok;
    }
    if (n_tok != st->n_tok_cur) {
        LOG_WRN("%s: %s has %lld tokens, ubatch has %lld - token index may be wrong\n", __func__, t->name, (long long) n_tok, (long long) st->n_tok_cur);
    }
    st->last_layer    = layer;
    st->n_expert_used = n_used;

    // the tensor is a view of argsort output and is not contiguous
    st->buf.resize(ggml_nbytes(t));
    ggml_backend_tensor_get(t, st->buf.data(), 0, st->buf.size());

    const char * phase = st->n_tok_cur == 1 ? "decode" : "prefill";
    for (int64_t it = 0; it < n_tok; ++it) {
        for (int64_t k = 0; k < n_used; ++k) {
            const int32_t ex = *(const int32_t *) (st->buf.data() + it*t->nb[1] + k*t->nb[0]);
            st->out << st->ubatch << ',' << phase << ',' << layer << ',' << (st->tok_base + it) << ',' << k << ',' << ex << '\n';
        }
    }
    st->n_rows += n_tok*n_used;
    st->rows_per_layer[layer] += n_tok*n_used;

    return true;
}

struct layer_bytes {
    int64_t n_expert   = 0;
    int64_t gate       = 0;
    int64_t up         = 0;
    int64_t down       = 0;
    int64_t gate_up    = 0;
    std::string type;
};

// read per-expert tensor sizes from the GGUF file(s), at the stored quant type
static bool read_expert_bytes(const std::string & path, std::map<int, layer_bytes> & layers) {
    std::vector<std::string> paths = { path };

    {
        gguf_init_params ip = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
        gguf_context * g = gguf_init_from_file(path.c_str(), ip);
        if (!g) {
            return false;
        }
        const int64_t kid = gguf_find_key(g, "split.count");
        const int n_split = kid >= 0 ? (int) gguf_get_val_u16(g, kid) : 1;
        gguf_free(g);

        if (n_split > 1) {
            char prefix[4096];
            if (!llama_split_prefix(prefix, sizeof(prefix), path.c_str(), 0, n_split)) {
                LOG_WRN("%s: cannot get split prefix of %s, reading first file only\n", __func__, path.c_str());
            } else {
                paths.clear();
                for (int i = 0; i < n_split; ++i) {
                    char p[4096];
                    llama_split_path(p, sizeof(p), prefix, i, n_split);
                    paths.push_back(p);
                }
            }
        }
    }

    for (const auto & p : paths) {
        ggml_context * meta = nullptr;
        gguf_init_params ip = { /*.no_alloc =*/ true, /*.ctx =*/ &meta };
        gguf_context * g = gguf_init_from_file(p.c_str(), ip);
        if (!g) {
            LOG_ERR("%s: failed to read %s\n", __func__, p.c_str());
            return false;
        }
        for (ggml_tensor * t = ggml_get_first_tensor(meta); t; t = ggml_get_next_tensor(meta, t)) {
            // match "blk.<il>.ffn_<kind>_exps.weight"
            const std::string name = t->name;
            const std::string suffix = "_exps.weight";
            const size_t p_ffn = name.find(".ffn_");
            if (name.rfind("blk.", 0) != 0 || p_ffn == std::string::npos || t->ne[2] <= 0 ||
                name.size() < suffix.size() || name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0) {
                continue;
            }
            const int il = atoi(name.c_str() + 4);
            const std::string kind = name.substr(p_ffn + 5, name.size() - suffix.size() - (p_ffn + 5));

            auto & lb = layers[il];
            const int64_t per = (int64_t) ggml_nbytes(t) / t->ne[2];
            if      (kind == "gate")    { lb.gate    = per; }
            else if (kind == "up")      { lb.up      = per; }
            else if (kind == "down")    { lb.down    = per; }
            else if (kind == "gate_up") { lb.gate_up = per; }
            else { continue; }
            lb.n_expert = t->ne[2];
            const std::string type = ggml_type_name(t->type);
            if (lb.type.empty()) {
                lb.type = type;
            } else if (lb.type.find(type) == std::string::npos) {
                lb.type += "+" + type;
            }
        }
        gguf_free(g);
        ggml_free(meta);
    }
    return true;
}

static int64_t meta_int(const llama_model * model, const std::string & key, int64_t def) {
    char buf[128];
    if (llama_model_meta_val_str(model, key.c_str(), buf, sizeof(buf)) < 0) {
        return def;
    }
    return atoll(buf);
}

static std::string json_escape(const std::string & s) {
    std::string r;
    for (char c : s) {
        if (c == '"' || c == '\\') {
            r += '\\';
        }
        r += c;
    }
    return r;
}

static bool write_sidecar(const std::string & path, const common_params & params, const llama_model * model, const trace_state & st, int64_t n_tokens) {
    std::map<int, layer_bytes> layers;
    if (!read_expert_bytes(params.model.path, layers)) {
        return false;
    }

    char arch[128] = {0};
    llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));

    std::string fname = params.model.path;
    const size_t pos = fname.find_last_of("/\\");
    if (pos != std::string::npos) {
        fname = fname.substr(pos + 1);
    }

    const int64_t n_expert = meta_int(model, std::string(arch) + ".expert_count", layers.empty() ? 0 : layers.begin()->second.n_expert);
    const int64_t n_used   = meta_int(model, std::string(arch) + ".expert_used_count", st.n_expert_used);

    std::ofstream f(path);
    if (!f) {
        return false;
    }
    f << "{\n";
    f << "  \"model\": \"" << json_escape(fname) << "\",\n";
    f << "  \"arch\": \"" << json_escape(arch) << "\",\n";
    f << "  \"n_layer\": " << llama_model_n_layer(model) << ",\n";
    f << "  \"n_expert\": " << n_expert << ",\n";
    f << "  \"n_expert_used\": " << n_used << ",\n";
    f << "  \"n_layer_moe\": " << layers.size() << ",\n";
    f << "  \"n_tokens\": " << n_tokens << ",\n";
    f << "  \"n_ubatch\": " << (st.ubatch + 1) << ",\n";
    f << "  \"n_rows\": " << st.n_rows << ",\n";
    f << "  \"layers\": [\n";
    size_t i = 0;
    for (const auto & [il, lb] : layers) {
        const int64_t total = lb.gate + lb.up + lb.down + lb.gate_up;
        f << "    {\"layer\": " << il << ", \"n_expert\": " << lb.n_expert << ", \"type\": \"" << lb.type << "\""
          << ", \"gate_bytes\": " << lb.gate << ", \"up_bytes\": " << lb.up << ", \"down_bytes\": " << lb.down
          << ", \"gate_up_bytes\": " << lb.gate_up << ", \"expert_bytes\": " << total << "}"
          << (++i < layers.size() ? "," : "") << "\n";
    }
    f << "  ]\n";
    f << "}\n";
    return true;
}

static std::string sidecar_path(const std::string & csv) {
    const std::string ext = ".csv";
    if (csv.size() > ext.size() && csv.compare(csv.size() - ext.size(), ext.size(), ext) == 0) {
        return csv.substr(0, csv.size() - ext.size()) + ".json";
    }
    return csv + ".json";
}

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n    %s -m model.gguf -p \"prompt\" -n 128 --trace-out trace.csv\n", argv[0]);
    LOG("\n");
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    // strip our own arg before passing the rest to the common parser
    std::string trace_out;
    std::vector<char *> args;
    for (int i = 0; i < argc; ++i) {
        if (strcmp(argv[i], "--trace-out") == 0 && i + 1 < argc) {
            trace_out = argv[++i];
            continue;
        }
        args.push_back(argv[i]);
    }

    common_params params;
    params.n_predict = 32;

    if (!common_params_parse((int) args.size(), args.data(), params, LLAMA_EXAMPLE_COMMON, print_usage)) {
        return 1;
    }
    if (trace_out.empty()) {
        fprintf(stderr, "error: --trace-out <file> is required\n");
        print_usage(argc, argv);
        return 1;
    }

    common_init();

    trace_state st;
    st.out.open(trace_out);
    if (!st.out) {
        LOG_ERR("%s: cannot open %s\n", __func__, trace_out.c_str());
        return 1;
    }
    st.out << "ubatch,phase,layer,token,rank,expert\n";

    llama_backend_init();
    llama_numa_init(params.numa);

    params.cb_eval = trace_cb_eval;
    params.cb_eval_user_data = &st;
    params.warmup = false;

    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> prompt = common_tokenize(ctx, params.prompt, llama_vocab_get_add_bos(vocab), true);
    if (prompt.empty()) {
        LOG_ERR("%s : no input tokens - use -p or -f\n", __func__);
        return 1;
    }
    if ((int) prompt.size() + std::max(params.n_predict, 0) > (int) llama_n_ctx(ctx)) {
        LOG_ERR("%s : prompt (%zu) + n_predict (%d) exceeds context size (%u)\n", __func__, prompt.size(), params.n_predict, llama_n_ctx(ctx));
        return 1;
    }

    common_sampler * smpl = common_sampler_init(model, params.sampling);

    // request output for all prompt tokens, so the last layer is not reduced to the output tokens only
    // decode in chunks of n_ubatch to keep the logits buffer small
    const int n_chunk = std::min(llama_n_batch(ctx), llama_n_ubatch(ctx));
    llama_batch batch = llama_batch_init(n_chunk, 0, 1);
    int64_t n_past = 0;

    for (size_t i = 0; i < prompt.size(); i += n_chunk) {
        const int n = std::min((int) (prompt.size() - i), n_chunk);
        common_batch_clear(batch);
        for (int j = 0; j < n; ++j) {
            common_batch_add(batch, prompt[i + j], n_past + j, { 0 }, true);
        }
        if (llama_decode(ctx, batch)) {
            LOG_ERR("%s : failed to eval prompt\n", __func__);
            return 1;
        }
        n_past += n;
    }
    llama_batch_free(batch);

    for (int i = 0; i < params.n_predict; ++i) {
        llama_token id = common_sampler_sample(smpl, ctx, -1);
        common_sampler_accept(smpl, id, true);
        if (llama_vocab_is_eog(vocab, id)) {
            break;
        }
        LOG("%s", common_token_to_piece(ctx, id).c_str());
        if (i + 1 == params.n_predict) {
            break;
        }
        if (llama_decode(ctx, llama_batch_get_one(&id, 1))) {
            LOG_ERR("%s : failed to eval\n", __func__);
            return 1;
        }
        n_past++;
    }
    LOG("\n");

    st.out.close();

    const std::string meta_path = sidecar_path(trace_out);
    if (!write_sidecar(meta_path, params, model, st, n_past)) {
        LOG_ERR("%s : failed to write %s\n", __func__, meta_path.c_str());
        return 1;
    }

    LOG_INF("\n%s: tokens = %lld, ubatches = %lld, moe layers traced = %zu, rows = %lld\n", __func__,
            (long long) n_past, (long long) (st.ubatch + 1), st.rows_per_layer.size(), (long long) st.n_rows);
    LOG_INF("%s: trace written to %s, metadata to %s\n", __func__, trace_out.c_str(), meta_path.c_str());

    common_sampler_free(smpl);
    llama_perf_context_print(ctx);

    llama_backend_free();

    return 0;
}
