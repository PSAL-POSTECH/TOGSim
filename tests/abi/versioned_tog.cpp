// A one-compute TOG that states its ABI version itself: TOGSIM_HOST makes the header only declare the symbol.
// Built plain it defines togsim_abi_version as the header's; -DPREVIOUS_VERSION states one less, -DOLD_SYMBOL
// defines the symbol of ABI 15 and earlier (togsim_producer_abi_version = 15), -DNO_SYMBOL defines neither.
#define TOGSIM_HOST
#include "togsim_runtime.h"

#if defined(PREVIOUS_VERSION)
extern "C" const int32_t togsim_abi_version = TOGSIM_ABI_VERSION - 1;
#elif defined(OLD_SYMBOL)
extern "C" const int32_t togsim_producer_abi_version = 15;
#elif !defined(NO_SYMBOL)
extern "C" const int32_t togsim_abi_version = TOGSIM_ABI_VERSION;
#endif

static void tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)iv; (void)n_iv;
  const int64_t out[1] = {1};
  togsim_compute(ctx, 0, 0, 0, nullptr, nullptr, 0, out, 1);
}

extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n_shape_args) {
  (void)shape_args; (void)n_shape_args;
  int64_t iv[1] = {0};
  togsim_dispatch(ctx, tile, iv, 1, 0);
}

extern "C" const int32_t togsim_spad_buffer_count = 2;
extern "C" const int64_t togsim_spad_buffer_bytes[2] = {0, 64};
