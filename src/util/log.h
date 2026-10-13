// Minimal thread-safe logger (console + file) and CSV metrics writer.
#pragma once

#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace util {

class Logger {
 public:
  static Logger& get();
  void open(const std::string& path);  // append mode
  void info(const std::string& msg);
  void warn(const std::string& msg);

 private:
  void write(const std::string& level, const std::string& msg);
  std::mutex mu_;
  std::ofstream file_;
};

inline void log_info(const std::string& m) { Logger::get().info(m); }
inline void log_warn(const std::string& m) { Logger::get().warn(m); }

// Appends rows to a CSV file, writing the header when the file is new/empty.
class CsvWriter {
 public:
  CsvWriter(const std::string& path, const std::vector<std::string>& columns);
  void write(const std::map<std::string, double>& row);

 private:
  std::string path_;
  std::vector<std::string> columns_;
};

std::string timestamp();
// Atomically replaces `dst` with `tmp` (rename). Retries for a moment because on Windows a file
// that another process is reading cannot be replaced; falls back to copying. Returns success.
bool replace_file(const std::string& tmp, const std::string& dst);
std::string format(const char* fmt, ...);

}  // namespace util
