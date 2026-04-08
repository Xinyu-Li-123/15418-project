#pragma once

namespace gpjson::index {
enum class IndexBuilderType { UNCOMBINED = 0, COMBINED, NO_ESCAPE_QUOTE };

// Build all the necessary indices. Abstract
class IndexBuilder {};

// Execute separate kernels
class UncombinedIndexBuilder : IndexBuilder {};

// Execute combined kernels
class CombinedIndexBuilder : IndexBuilder {};
} // namespace gpjson::index
