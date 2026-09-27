#pragma once

#include "common.h"

#include <string>

// sizes like 512M, 4G, 1.5GiB, 1048576 (bytes); throws std::invalid_argument
int64_t common_parse_size(const std::string & s);

// read a --moe-placement file: a hot set (analyze.py --emit-placement) or a ranking (--emit-ranking)
void common_moe_placement_load_file(const std::string & path, common_params_moe_placement & moe);

// with a ranking: measure the free VRAM with a no_alloc load of the model and context (after --fit), fill it with
// the experts of highest count per byte, optionally pick the RAM tier, print the placement report and update mparams
void common_moe_placement_resolve(common_params & params, llama_model_params & mparams, const llama_context_params & cparams);
