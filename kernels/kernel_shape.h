// What the kernels here share: reading their shape arguments, and the extent and the tile id of a short last tile.
// Without a tog_shape.txt a kernel runs its example shape; another count than its own, or a dimension below 1, is
// refused, as the wrong shape would still replay.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define KERNEL_ELEM_BITS 32
#define KERNEL_ELEM_BYTES 4
#define KERNEL_MAX(a, b) ((a) > (b) ? (a) : (b))
#define KERNEL_MIN(a, b) ((a) < (b) ? (a) : (b))
#define KERNEL_CEIL(a, b) (((a) + (b) - 1) / (b))

// The extent of the tile at `at` on an axis `dim` long that is walked `tile` at a time: what is left, for the last.
static inline int64_t kernel_extent(int64_t dim, int64_t at, int64_t tile) { return KERNEL_MIN(tile, dim - at); }

// The bit a tile `extent` long sets in its variant for the kernel's axis number `axis`: set when it is short of `tile`.
static inline int64_t kernel_short(int64_t extent, int64_t tile, int32_t axis) {
  return extent < tile ? (int64_t)1 << axis : 0;
}

// The tile id of compute node `node`, of a kernel with `nodes` of them, on a tile of `variant`: node + nodes * variant.
// Variant 0 is the whole tile and keeps the node's own id; a variant is the short axes the node's tile spans.
static inline uint64_t kernel_tile_id(int64_t variant, int64_t nodes, uint64_t node) {
  return node + (uint64_t)(variant * nodes);
}

// Fill `shape` with the `count` values `names` lists: `shape_args` when the TOG was given them, else `example`.
// The first `extents` are dimensions, at least 1; the rest are offsets such as a padding, at least 0.
static inline void kernel_shape_from(const char* kernel, const char* names, int32_t count, int32_t extents,
                                     const int64_t* example, const int64_t* shape_args, int32_t n_shape_args,
                                     int64_t* shape) {
  if (n_shape_args != 0 && n_shape_args != count) {
    fprintf(stderr, "kernel %s: it takes %d shape arguments (%s), tog_shape.txt gives %d\n",
            kernel, (int)count, names, (int)n_shape_args);
    exit(1);
  }
  for (int32_t i = 0; i < count; ++i) {
    shape[i] = n_shape_args ? shape_args[i] : example[i];
    const int64_t least = i < extents ? 1 : 0;
    if (shape[i] < least) {
      fprintf(stderr, "kernel %s: shape argument %d of (%s) is %lld, it must be at least %d\n",
              kernel, (int)i, names, (long long)shape[i], (int)least);
      exit(1);
    }
  }
}

// Fill `shape` with the `count` dimensions `names` lists: `shape_args` when the TOG was given them, else `example`.
static inline void kernel_shape(const char* kernel, const char* names, int32_t count, const int64_t* example,
                                const int64_t* shape_args, int32_t n_shape_args, int64_t* shape) {
  kernel_shape_from(kernel, names, count, count, example, shape_args, n_shape_args, shape);
}
