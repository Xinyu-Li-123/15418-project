#pragma once

#include <chrono>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

namespace gpjson::profiler {

class Profiler {
public:
  class SegmentId {
  public:
    SegmentId() = default;

  private:
    explicit SegmentId(size_t index);

    size_t index_{0};
    bool valid_{false};

    friend class Profiler;
  };

  class ScopedSegment {
  public:
    ScopedSegment() = default;
    ScopedSegment(Profiler *profiler, SegmentId segment_id) noexcept;

    ScopedSegment(const ScopedSegment &) = delete;
    ScopedSegment &operator=(const ScopedSegment &) = delete;

    ScopedSegment(ScopedSegment &&other) noexcept;
    ScopedSegment &operator=(ScopedSegment &&other) noexcept;

    ~ScopedSegment();

    void close();
    bool active() const;

  private:
    Profiler *profiler_{nullptr};
    SegmentId segment_id_{};
    bool active_{false};
  };

  SegmentId begin(std::string_view name);
  SegmentId beginf(const char *fmt, ...);

  void end(SegmentId segment_id);

  ScopedSegment scope(std::string_view name);
  ScopedSegment scopef(const char *fmt, ...);

  void print(FILE *out = stdout) const;

private:
  using Clock = std::chrono::steady_clock;
  using Duration = std::chrono::nanoseconds;
  using TimePoint = Clock::time_point;

  struct SegmentRecord {
    std::string name;
    TimePoint start_time{};
    Duration elapsed{Duration::zero()};
    bool completed{false};
  };

  SegmentId begin_formatted(const char *fmt, va_list args);
  std::string vformat(const char *fmt, va_list args) const;

  std::vector<SegmentRecord> segments_;
};
} // namespace gpjson::profiler
