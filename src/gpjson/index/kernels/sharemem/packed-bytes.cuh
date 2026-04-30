#pragma once

#include <cstddef>

// #define FORCED_GMEM_PACK_TYPE uint4
// #define FORCED_GMEM_PACK_TYPE uint2
// #define FORCED_GMEM_PACK_TYPE uint
// #define FORCED_GMEM_PACK_TYPE unsigned short
// #define FORCED_SMEM_PACK_TYPE uint2

namespace gpjson::index::kernels::sharemem::packed_bytes {

template <int PACK_BYTES> __host__ __device__ constexpr void check_size() {
  static_assert(PACK_BYTES == 2 || PACK_BYTES == 4 || PACK_BYTES == 8 ||
                    PACK_BYTES == 16,
                "Pack type must be 2, 4, 8, or 16 bytes.");
}

template <typename PackedT> __host__ __device__ constexpr int pack_size() {
  constexpr int PACK_BYTES = static_cast<int>(sizeof(PackedT));
  check_size<PACK_BYTES>();
  return PACK_BYTES;
}

template <int PACK_BYTES>
__host__ __device__ constexpr unsigned int pack_bit_mask() {
  check_size<PACK_BYTES>();
  return (1u << PACK_BYTES) - 1u;
}

template <typename PackedT>
__device__ __forceinline__ unsigned char *bytes(PackedT &packed) {
  return reinterpret_cast<unsigned char *>(&packed);
}

template <typename PackedT>
__device__ __forceinline__ const unsigned char *bytes(const PackedT &packed) {
  return reinterpret_cast<const unsigned char *>(&packed);
}

template <typename GmemPackT>
__device__ __forceinline__ GmemPackT load_gmem_pack_or_tail(
    const char *file, size_t fileSize, size_t gmem_byte_base) {
  constexpr int GMEM_PACK_BYTES = pack_size<GmemPackT>();

  GmemPackT packed{};

  if (gmem_byte_base + GMEM_PACK_BYTES <= fileSize) {
    const GmemPackT *file_packed_gmem =
        reinterpret_cast<const GmemPackT *>(file);
    packed = file_packed_gmem[gmem_byte_base / GMEM_PACK_BYTES];
  } else if (gmem_byte_base < fileSize) {
    unsigned char *dst = bytes(packed);

#pragma unroll
    for (int i = 0; i < GMEM_PACK_BYTES; ++i) {
      const size_t global_idx = gmem_byte_base + i;
      dst[i] = (global_idx < fileSize)
                   ? static_cast<unsigned char>(file[global_idx])
                   : static_cast<unsigned char>(0);
    }
  }

  return packed;
}

template <typename GmemPackT, typename SmemPackT>
__device__ __forceinline__ SmemPackT make_smem_pack_from_gmem_range(
    const char *file, size_t fileSize, size_t smem_global_byte_base) {
  constexpr int GMEM_PACK_BYTES = pack_size<GmemPackT>();
  constexpr int SMEM_PACK_BYTES = pack_size<SmemPackT>();

  SmemPackT smem_packed{};
  unsigned char *dst = bytes(smem_packed);

  const size_t smem_global_byte_end = smem_global_byte_base + SMEM_PACK_BYTES;
  const size_t first_gmem_byte_base =
      (smem_global_byte_base / GMEM_PACK_BYTES) * GMEM_PACK_BYTES;

  constexpr int MAX_GMEM_PACKS =
      (SMEM_PACK_BYTES + GMEM_PACK_BYTES - 1) / GMEM_PACK_BYTES + 1;

#pragma unroll
  for (int k = 0; k < MAX_GMEM_PACKS; ++k) {
    const size_t gmem_byte_base =
        first_gmem_byte_base + static_cast<size_t>(k) * GMEM_PACK_BYTES;

    if (gmem_byte_base >= smem_global_byte_end) {
      break;
    }

    const GmemPackT gmem_packed =
        load_gmem_pack_or_tail<GmemPackT>(file, fileSize, gmem_byte_base);
    const unsigned char *src = bytes(gmem_packed);

#pragma unroll
    for (int b = 0; b < GMEM_PACK_BYTES; ++b) {
      const size_t global_byte = gmem_byte_base + b;

      if (global_byte >= smem_global_byte_base &&
          global_byte < smem_global_byte_end) {
        const int smem_local_byte =
            static_cast<int>(global_byte - smem_global_byte_base);
        dst[smem_local_byte] = src[b];
      }
    }
  }

  return smem_packed;
}

template <bool TRANSPOSED, typename SmemPackT, int BYTES_PER_THREAD,
          int THREADS_PER_BLOCK>
__device__ __forceinline__ int
smem_index_from_byte_offset(size_t byte_offset_in_block) {
  constexpr int SMEM_PACK_BYTES = pack_size<SmemPackT>();
  constexpr int SMEM_GROUPS_PER_THREAD = BYTES_PER_THREAD / SMEM_PACK_BYTES;

  const int owner_tid =
      static_cast<int>(byte_offset_in_block / BYTES_PER_THREAD);
  const int byte_in_thread =
      static_cast<int>(byte_offset_in_block % BYTES_PER_THREAD);
  const int group = byte_in_thread / SMEM_PACK_BYTES;

  if constexpr (TRANSPOSED) {
    return group * THREADS_PER_BLOCK + owner_tid;
  } else {
    return owner_tid * SMEM_GROUPS_PER_THREAD + group;
  }
}

template <bool TRANSPOSED, typename GmemPackT, typename SmemPackT,
          int BYTES_PER_THREAD, int THREADS_PER_BLOCK>
__device__ __forceinline__ void
stage_file_to_smem(const char *file, size_t fileSize, size_t block_start,
                   SmemPackT *smem_packed_bytes) {
  constexpr int CHUNK_SIZE = THREADS_PER_BLOCK * BYTES_PER_THREAD;
  constexpr int GMEM_PACK_BYTES = pack_size<GmemPackT>();
  constexpr int SMEM_PACK_BYTES = pack_size<SmemPackT>();

  static_assert(BYTES_PER_THREAD % GMEM_PACK_BYTES == 0);
  static_assert(BYTES_PER_THREAD % SMEM_PACK_BYTES == 0);
  static_assert(CHUNK_SIZE % GMEM_PACK_BYTES == 0);
  static_assert(CHUNK_SIZE % SMEM_PACK_BYTES == 0);

  const int tid = threadIdx.x;

  if constexpr (GMEM_PACK_BYTES >= SMEM_PACK_BYTES) {
    static_assert(GMEM_PACK_BYTES % SMEM_PACK_BYTES == 0);

    constexpr int GMEM_PACKED_ELEMS_PER_BLOCK = CHUNK_SIZE / GMEM_PACK_BYTES;
    constexpr int SMEM_PACKS_PER_GMEM_PACK = GMEM_PACK_BYTES / SMEM_PACK_BYTES;

    for (int p = tid; p < GMEM_PACKED_ELEMS_PER_BLOCK; p += blockDim.x) {
      const size_t gmem_byte_offset_in_block =
          static_cast<size_t>(p) * GMEM_PACK_BYTES;
      const size_t gmem_global_byte_base =
          block_start + gmem_byte_offset_in_block;

      const GmemPackT gmem_packed = load_gmem_pack_or_tail<GmemPackT>(
          file, fileSize, gmem_global_byte_base);
      const unsigned char *src = bytes(gmem_packed);

#pragma unroll
      for (int sub = 0; sub < SMEM_PACKS_PER_GMEM_PACK; ++sub) {
        const size_t smem_byte_offset_in_block =
            gmem_byte_offset_in_block +
            static_cast<size_t>(sub) * SMEM_PACK_BYTES;

        SmemPackT smem_packed{};
        unsigned char *dst = bytes(smem_packed);

#pragma unroll
        for (int b = 0; b < SMEM_PACK_BYTES; ++b) {
          dst[b] = src[sub * SMEM_PACK_BYTES + b];
        }

        const int smem_idx =
            smem_index_from_byte_offset<TRANSPOSED, SmemPackT, BYTES_PER_THREAD,
                                        THREADS_PER_BLOCK>(
                smem_byte_offset_in_block);
        smem_packed_bytes[smem_idx] = smem_packed;
      }
    }
  } else {
    static_assert(SMEM_PACK_BYTES % GMEM_PACK_BYTES == 0);

    constexpr int SMEM_PACKED_ELEMS_PER_BLOCK = CHUNK_SIZE / SMEM_PACK_BYTES;

    for (int p = tid; p < SMEM_PACKED_ELEMS_PER_BLOCK; p += blockDim.x) {
      const size_t smem_byte_offset_in_block =
          static_cast<size_t>(p) * SMEM_PACK_BYTES;
      const size_t smem_global_byte_base =
          block_start + smem_byte_offset_in_block;

      const SmemPackT smem_packed =
          make_smem_pack_from_gmem_range<GmemPackT, SmemPackT>(
              file, fileSize, smem_global_byte_base);

      const int smem_idx =
          smem_index_from_byte_offset<TRANSPOSED, SmemPackT, BYTES_PER_THREAD,
                                      THREADS_PER_BLOCK>(
              smem_byte_offset_in_block);
      smem_packed_bytes[smem_idx] = smem_packed;
    }
  }
}

} // namespace gpjson::index::kernels::sharemem::packed_bytes
