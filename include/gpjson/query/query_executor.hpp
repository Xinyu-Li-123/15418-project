#pragma once

namespace gpjson::query {
class QueryExecutor {
public:
  QueryResult execute(std::string query, file::FileReader file_reader,
                      index::IndexBuilder index_builder);

  std::vector<QueryResult> executeBatch(std::vector<std::string query>,
                                        file::PartitionView partition_view,
                                        index::IndexBuilder index_builder);
};
} // namespace gpjson::query
