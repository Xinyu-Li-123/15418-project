#pragma once

#include "gpjson/profiler/profiler.hpp"

namespace gpjson {

struct RunContext {
  profiler::Profiler &profiler;
};

} // namespace gpjson
