#pragma once
#include <cstdint>

namespace sparkinfer {
// A warm arena alone does not predict graph reuse: graph parameters also belong
// to a model and a session. Require a second sighting of that same identity.
struct PrefillWarmKey {
    int tokens = -1;
    uint64_t sequence = 0;
    const void* model = nullptr;
    const void* recurrent = nullptr;
    const void* convolution = nullptr;
    const void* block_table = nullptr;

    bool matches(const PrefillWarmKey& other) const {
        return tokens >= 0 && tokens == other.tokens && sequence == other.sequence &&
               model == other.model && recurrent == other.recurrent &&
               convolution == other.convolution && block_table == other.block_table;
    }
};
} // namespace sparkinfer
