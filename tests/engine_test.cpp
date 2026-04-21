#include "gpjson/engine.hpp"

#include "gpjson/error/common.hpp"
#include "utils/utils.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

namespace {

std::filesystem::path write_temp_file(const std::string &filename,
                                      const std::string &contents) {
  const auto path = std::filesystem::temp_directory_path() / filename;
  std::ofstream output(path, std::ios::binary);
  EXPECT_TRUE(output.is_open());
  output << contents;
  output.close();
  return path;
}

class EngineIntegrationTest : public ::testing::Test {
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

private:
  std::filesystem::path temp_path_;
};

TEST_F(EngineIntegrationTest, QueriesPartitionedLdJsonFileEndToEnd) {
  GPJSON_SKIP_IF_CUDA_UNAVAILABLE();

  const auto path =
      create_file("gpjson_engine_partitioned.ldjson",
                  "{\"name\":\"Ada\",\"scores\":[1,2],\"lang\":\"en\"}\n"
                  "{\"name\":\"Bob\",\"scores\":[3,4],\"lang\":\"fr\"}\n");

  const gpjson::EngineOptions options{
      gpjson::file::FileReaderOptions{
          gpjson::file::FileReaderType::COPIED,
      },
      gpjson::index::IndexBuilderOptions{
          gpjson::index::IndexBuilderType::UNCOMBINED,
          43,
          1,
          64,
          1,
          64,
      },
  };

  gpjson::Engine engine;
  const std::vector<std::string> queries{
      "$.name",
      "$.scores[1]",
  };

  const auto result = engine.query(path.string(), queries, options);

  ASSERT_EQ(result.queries().size(), 2U);

  ASSERT_EQ(result.queries()[0].query_text(), "$.name");
  ASSERT_EQ(result.queries()[0].lines().size(), 2U);
  ASSERT_EQ(result.queries()[0].lines()[0].values().size(), 1U);
  ASSERT_EQ(result.queries()[0].lines()[1].values().size(), 1U);
  EXPECT_EQ(result.queries()[0].lines()[0].values()[0].json_text(), "\"Ada\"");
  EXPECT_EQ(result.queries()[0].lines()[1].values()[0].json_text(), "\"Bob\"");

  ASSERT_EQ(result.queries()[1].query_text(), "$.scores[1]");
  ASSERT_EQ(result.queries()[1].lines().size(), 2U);
  ASSERT_EQ(result.queries()[1].lines()[0].values().size(), 1U);
  ASSERT_EQ(result.queries()[1].lines()[1].values().size(), 1U);
  EXPECT_EQ(result.queries()[1].lines()[0].values()[0].json_text(), "2");
  EXPECT_EQ(result.queries()[1].lines()[1].values()[0].json_text(), "4");
}

TEST_F(EngineIntegrationTest, RejectsUnimplementedMmapFileReader) {
  const auto path = create_file("gpjson_engine_unimplemented_mmap.ldjson",
                                "{\"name\":\"Ada\"}\n");

  const gpjson::EngineOptions options{
      gpjson::file::FileReaderOptions{
          gpjson::file::FileReaderType::MMAP,
      },
      gpjson::index::IndexBuilderOptions{},
  };
  const std::vector<std::string> queries{"$.name"};

  gpjson::Engine engine;

  EXPECT_THROW(engine.query(path.string(), queries, options),
               gpjson::error::common::ImplementationError);
}

} // namespace
