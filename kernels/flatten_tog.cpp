// Flatten moves no data and computes nothing: the kernel dispatches no work-item and names no spad buffer.
// It takes no shape arguments.
#include "togsim_runtime.h"

#include "kernel_shape.h"

extern "C" const int32_t togsim_spad_buffer_count = 0;

// Nothing to dispatch.
extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n) {
  (void)ctx;
  kernel_shape("flatten", "", 0, nullptr, shape_args, n, nullptr);
}
