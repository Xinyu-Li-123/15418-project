#include "gpjson/profiler/profiler.hpp"

#include "gpjson/log/log.hpp"

#include <cstdarg>
#include <cstdio>
#include <utility>
#include <vector>

namespace gpjson::profiler {

namespace {

constexpr double kNanosecondsPerMillisecond = 1000000.0;

} // namespace

Profiler::SegmentId::SegmentId(size_t index) : index_(index), valid_(true) {}

Profiler::ScopedSegment::ScopedSegment(Profiler *profiler,
                                       SegmentId segment_id) noexcept
    : profiler_(profiler), segment_id_(segment_id), active_(true) {}

Profiler::ScopedSegment::ScopedSegment(ScopedSegment &&other) noexcept
    : profiler_(other.profiler_), segment_id_(other.segment_id_),
      active_(other.active_) {
  other.profiler_ = nullptr;
  other.segment_id_ = SegmentId{};
  other.active_ = false;
}

Profiler::ScopedSegment &
Profiler::ScopedSegment::operator=(ScopedSegment &&other) noexcept {
  if (this == &other) {
    return *this;
  }

  if (active_) {
    close();
  }

  profiler_ = other.profiler_;
  segment_id_ = other.segment_id_;
  active_ = other.active_;

  other.profiler_ = nullptr;
  other.segment_id_ = SegmentId{};
  other.active_ = false;
  return *this;
}

Profiler::ScopedSegment::~ScopedSegment() { close(); }

void Profiler::ScopedSegment::close() {
  if (!active_) {
    return;
  }

  profiler_->end(segment_id_);
  active_ = false;
}

bool Profiler::ScopedSegment::active() const { return active_; }

Profiler::SegmentId Profiler::begin(std::string_view name) {
  SegmentRecord record;
  record.name = std::string(name);
  record.start_time = Clock::now();
  segments_.push_back(std::move(record));
  return SegmentId(segments_.size() - 1);
}

Profiler::SegmentId Profiler::beginf(const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  const SegmentId segment_id = begin_formatted(fmt, args);
  va_end(args);
  return segment_id;
}

Profiler::SegmentId Profiler::begin_formatted(const char *fmt, va_list args) {
  return begin(vformat(fmt, args));
}

void Profiler::end(SegmentId segment_id) {
  Assert(segment_id.valid_, "Profiler end called with invalid segment id.");
  if (!segment_id.valid_) {
    return;
  }

  Assert(segment_id.index_ < segments_.size(),
         "Profiler end segment id %zu out of bounds.", segment_id.index_);
  if (segment_id.index_ >= segments_.size()) {
    return;
  }

  SegmentRecord &record = segments_[segment_id.index_];
  Assert(!record.completed, "Profiler segment '%s' already completed.",
         record.name.c_str());
  if (record.completed) {
    return;
  }

  record.elapsed =
      std::chrono::duration_cast<Duration>(Clock::now() - record.start_time);
  record.completed = true;
}

Profiler::ScopedSegment Profiler::scope(std::string_view name) {
  return ScopedSegment(this, begin(name));
}

Profiler::ScopedSegment Profiler::scopef(const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  const SegmentId segment_id = begin_formatted(fmt, args);
  va_end(args);
  return ScopedSegment(this, segment_id);
}

void Profiler::print(FILE *out) const {
  std::fprintf(out, "Profiler results:\n");
  for (const SegmentRecord &record : segments_) {
    if (!record.completed) {
      Assert(false, "Profiler segment '%s' was not completed before printing.",
             record.name.c_str());
      std::fprintf(out, "  %s: incomplete\n", record.name.c_str());
      continue;
    }

    const double elapsed_ms = static_cast<double>(record.elapsed.count()) /
                              kNanosecondsPerMillisecond;
    std::fprintf(out, "  %s: %.3f ms\n", record.name.c_str(), elapsed_ms);
  }
  std::fflush(out);
}

std::string Profiler::vformat(const char *fmt, va_list args) const {
  va_list copied_args;
  va_copy(copied_args, args);
  const int size = std::vsnprintf(nullptr, 0, fmt, copied_args);
  va_end(copied_args);

  Assert(size >= 0, "Failed to format profiler segment name.");
  if (size < 0) {
    return {};
  }

  std::vector<char> buffer(static_cast<size_t>(size) + 1, '\0');
  va_copy(copied_args, args);
  std::vsnprintf(buffer.data(), buffer.size(), fmt, copied_args);
  va_end(copied_args);

  return std::string(buffer.data(), static_cast<size_t>(size));
}

} // namespace gpjson::profiler
