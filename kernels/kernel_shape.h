// What the kernels here share: reading their shape arguments. Without a tog_shape.txt a kernel runs its example
// shape; another count than its own, or a dimension below 1, is refused, as the wrong shape would still replay.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define KERNEL_ELEM_BITS 32
#define KERNEL_ELEM_BYTES 4
#define KERNEL_MAX(a, b) ((a) > (b) ? (a) : (b))
#define KERNEL_CEIL(a, b) (((a) + (b) - 1) / (b))

// Fill `shape` with the `count` dimensions `names` lists: `shape_args` when the TOG was given them, else `example`.
static inline void kernel_shape(const char* kernel, const char* names, int32_t count, const int64_t* example,
                                const int64_t* shape_args, int32_t n_shape_args, int64_t* shape) {
  if (n_shape_args != 0 && n_shape_args != count) {
    fprintf(stderr, "kernel %s: it takes %d shape arguments (%s), tog_shape.txt gives %d\n",
            kernel, (int)count, names, (int)n_shape_args);
    exit(1);
  }
  for (int32_t i = 0; i < count; ++i) {
    shape[i] = n_shape_args ? shape_args[i] : example[i];
    if (shape[i] < 1) {
      fprintf(stderr, "kernel %s: shape argument %d of (%s) is %lld, it must be at least 1\n",
              kernel, (int)i, names, (long long)shape[i]);
      exit(1);
    }
  }
}
