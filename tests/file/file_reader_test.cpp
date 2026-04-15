#include "gpjson/file/error.hpp"
#include "gpjson/file/file_reader.hpp"

#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

namespace {

using gpjson::file::FileReader;
using gpjson::file::PartitionView;

std::string bytes_to_string(const void *bytes, size_t size) {
  if (bytes == nullptr || size == 0) {
    return {};
  }
  return {reinterpret_cast<const char *>(bytes), size};
}

std::filesystem::path write_temp_file(const std::string &filename,
                                      const std::string &contents) {
  const auto path = std::filesystem::temp_directory_path() / filename;
  std::ofstream output(path, std::ios::binary);
  EXPECT_TRUE(output.is_open());
  output << contents;
  output.close();
  return path;
}

void expect_partition(const PartitionView &partition,
                      size_t expected_id,
                      size_t expected_start,
                      size_t expected_end,
                      const std::string &expected_contents) {
  EXPECT_EQ(partition.partition_id(), expected_id);
  EXPECT_EQ(partition.global_start_offset(), expected_start);
  EXPECT_EQ(partition.global_end_offset(), expected_end);
  EXPECT_EQ(partition.size_bytes(), expected_contents.size());
  EXPECT_EQ(bytes_to_string(partition.bytes(), partition.size_bytes()),
            expected_contents);
}

class FileReaderTest : public ::testing::Test {
protected:
  void TearDown() override {
    if (!temp_path_.empty()) {
      std::error_code error;
      std::filesystem::remove(temp_path_, error);
    }
  }

  std::filesystem::path create_file(const std::string &filename,
                                    const std::string &contents) {
    temp_path_ = write_temp_file(filename, contents);
    return temp_path_;
  }

  const std::filesystem::path &temp_path() const { return temp_path_; }

private:
  std::filesystem::path temp_path_;
};

TEST_F(FileReaderTest, CreatesSinglePartitionForWholeFile) {
  const std::string contents = "{\"id\":1}\n{\"id\":2}\n";
  const auto path = create_file("gpjson_file_reader_single_partition.json",
                                contents);

  FileReader reader(path.string());
  reader.create_partitions(0);

  const auto &metadata = reader.metadata();
  ASSERT_EQ(metadata.file_path, path.string());
  ASSERT_EQ(metadata.file_size_bytes, contents.size());
  ASSERT_EQ(metadata.num_partitions, 1U);

  const auto &partitions = reader.get_partition_views();
  ASSERT_EQ(partitions.size(), 1U);
  expect_partition(partitions.front(), 0, 0, contents.size(), contents);
}

TEST_F(FileReaderTest, SplitsPartitionsAtNewlineBoundaries) {
  const std::string contents = "alpha\nbeta\ngamma";
  create_file("gpjson_file_reader_partitioned.txt", contents);

  FileReader reader(temp_path().string());
  reader.create_partitions(7);

  const auto &metadata = reader.metadata();
  ASSERT_EQ(metadata.file_size_bytes, contents.size());
  ASSERT_EQ(metadata.num_partitions, 3U);

  const auto &partitions = reader.get_partition_views();
  ASSERT_EQ(partitions.size(), 3U);
  expect_partition(partitions[0], 0, 0, 5, "alpha");
  expect_partition(partitions[1], 1, 6, 10, "beta");
  expect_partition(partitions[2], 2, 11, contents.size(), "gamma");
}

TEST_F(FileReaderTest, ThrowsWhenNoNewlineExistsBeforeBoundary) {
  create_file("gpjson_file_reader_partition_error.txt", "abcdefghij");

  FileReader reader(temp_path().string());
  EXPECT_THROW(reader.create_partitions(4), gpjson::error::file::PartitionError);
}

} // namespace
