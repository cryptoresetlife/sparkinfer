#include "../src/models/prefill_warm_key.h"
#include <cstdio>

int main() {
    int identities[5]{};
    const sparkinfer::PrefillWarmKey a{256, 1, &identities[0], &identities[1],
                                      &identities[2], &identities[3]};
    auto check = [](bool ok, const char* name) {
        if (!ok) std::fprintf(stderr, "FAIL: %s\n", name);
        return ok;
    };
    bool ok = check(!sparkinfer::PrefillWarmKey{}.matches(a), "cold key");
    ok &= check(!sparkinfer::PrefillWarmKey{}.matches({}), "two cold keys");
    ok &= check(a.matches(a), "same identity permits capture");
    auto b = a;
    b.tokens = 512; ok &= check(!a.matches(b), "new length");
    b = a; ++b.sequence; ok &= check(!a.matches(b), "new session, recycled addresses");
    b = a; b.model = &identities[4]; ok &= check(!a.matches(b), "new model");
    b = a; b.recurrent = &identities[4]; ok &= check(!a.matches(b), "new recurrent state");
    b = a; b.convolution = &identities[4]; ok &= check(!a.matches(b), "new convolution state");
    b = a; b.block_table = &identities[4]; ok &= check(!a.matches(b), "new block table");
    b = a; b = {}; ok &= check(!b.matches(a), "arena eviction invalidates warmth");
    if (ok) std::puts("prefill warm-key identity checks passed");
    return ok ? 0 : 1;
}
