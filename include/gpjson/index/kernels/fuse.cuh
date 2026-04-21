namespace gpjson::index::kernels::fuse {
__global__ void escape_carry_index(const char *file, int fileSize,
                                   char *escapeCarryIndex);

__global__ void
quote_carry_index_using_escape_carry_index(const char *file, int fileSize,
                                           char *escapeCarryIndex,
                                           char *quoteCarryIndex);

__global__ void string_index_using_escape_carry_index_quote_carry_index(
    const char *file, int fileSize, char *escapeCarryIndex,
    char *quoteCarryIndex, long *stringIndex, int stringIndexSize);

} // namespace gpjson::index::kernels::fuse
