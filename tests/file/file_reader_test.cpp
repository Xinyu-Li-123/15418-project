#include "file/file_reader_test_utils.hpp"

#include "gpjson/file/error.hpp"
#include "gpjson/file/file.hpp"

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

#include <gtest/gtest.h>

namespace {

using gpjson::file::FilePartition;

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

void expect_partition(const FilePartition &partition, size_t expected_id,
                      size_t expected_start, size_t expected_end,
                      const std::string &expected_contents) {
  EXPECT_EQ(partition.partition_id(), expected_id);
  EXPECT_EQ(partition.global_start_offset(), expected_start);
  EXPECT_EQ(partition.global_end_offset(), expected_end);
  EXPECT_EQ(partition.size_bytes(), expected_contents.size());
  EXPECT_EQ(bytes_to_string(partition.host_bytes(), partition.size_bytes()),
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

class FileReaderBehaviorTest
    : public FileReaderTest,
      public ::testing::WithParamInterface<gpjson::file::FileReaderType> {
protected:
  std::unique_ptr<gpjson::file::FileReader>
  make_reader(const std::filesystem::path &path) const {
    return gpjson::test::file::make_file_reader(GetParam(), path.string());
  }
};

void expect_single_partition_for_whole_file(gpjson::file::FileReader &reader,
                                            const std::filesystem::path &path,
                                            const std::string &contents) {
  const auto &metadata = reader.metadata();
  ASSERT_EQ(metadata.file_path, path.string());
  ASSERT_EQ(metadata.file_size_bytes, contents.size());
  ASSERT_EQ(metadata.num_partitions, 1U);

  const auto &partitions = reader.get_partitions();
  ASSERT_EQ(partitions.size(), 1U);
  expect_partition(partitions.front(), 0, 0, contents.size(), contents);
}

void expect_partitions_split_at_newlines(gpjson::file::FileReader &reader,
                                         const std::string &contents) {
  const auto &metadata = reader.metadata();
  ASSERT_EQ(metadata.file_size_bytes, contents.size());
  ASSERT_EQ(metadata.num_partitions, 3U);

  const auto &partitions = reader.get_partitions();
  ASSERT_EQ(partitions.size(), 3U);
  expect_partition(partitions[0], 0, 0, 5, "alpha");
  expect_partition(partitions[1], 1, 6, 10, "beta");
  expect_partition(partitions[2], 2, 11, contents.size(), "gamma");
}

void expect_empty_file_partition(gpjson::file::FileReader &reader,
                                 const std::filesystem::path &path) {
  const auto &metadata = reader.metadata();
  ASSERT_EQ(metadata.file_path, path.string());
  ASSERT_EQ(metadata.file_size_bytes, 0U);
  ASSERT_EQ(metadata.num_partitions, 1U);

  const auto &partitions = reader.get_partitions();
  ASSERT_EQ(partitions.size(), 1U);
  expect_partition(partitions.front(), 0, 0, 0, "");
}

TEST_P(FileReaderBehaviorTest, CreatesSinglePartitionForWholeFile) {
  const std::string contents = "{\"id\":1}\n{\"id\":2}\n";
  const auto path =
      create_file("gpjson_file_reader_single_partition.json", contents);
  auto reader = make_reader(path);

  reader->create_partitions(0);
  expect_single_partition_for_whole_file(*reader, path, contents);
}

TEST_P(FileReaderBehaviorTest, SplitsPartitionsAtNewlineBoundaries) {
  const std::string contents = "alpha\nbeta\ngamma";
  const auto path = create_file("gpjson_file_reader_partitioned.txt", contents);
  auto reader = make_reader(path);

  reader->create_partitions(7);
  expect_partitions_split_at_newlines(*reader, contents);
}

TEST_P(FileReaderBehaviorTest, ThrowsWhenNoNewlineExistsBeforeBoundary) {
  const auto path =
      create_file("gpjson_file_reader_partition_error.txt", "abcdefghij");
  auto reader = make_reader(path);

  EXPECT_THROW(reader->create_partitions(4),
               gpjson::error::file::PartitionError);
}

TEST_P(FileReaderBehaviorTest, CreatesSingleEmptyPartitionForEmptyFile) {
  const auto path = create_file("gpjson_file_reader_empty.txt", "");
  auto reader = make_reader(path);

  reader->create_partitions(64);
  expect_empty_file_partition(*reader, path);
}

INSTANTIATE_TEST_SUITE_P(
    AllFileReaders, FileReaderBehaviorTest,
    ::testing::ValuesIn(gpjson::test::file::kFileReaderTypes),
    gpjson::test::file::file_reader_type_test_name);

} // namespace
