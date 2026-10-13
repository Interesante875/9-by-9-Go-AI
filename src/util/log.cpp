#include "util/log.h"

#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <thread>

namespace util {

Logger& Logger::get() {
  static Logger l;
  return l;
}

void Logger::open(const std::string& path) {
  std::lock_guard<std::mutex> lock(mu_);
  if (file_.is_open()) file_.close();
  file_.open(path, std::ios::app);
}

void Logger::info(const std::string& msg) { write("INFO", msg); }
void Logger::warn(const std::string& msg) { write("WARN", msg); }

void Logger::write(const std::string& level, const std::string& msg) {
  std::lock_guard<std::mutex> lock(mu_);
  std::string line = "[" + timestamp() + "] " + level + " " + msg;
  std::cout << line << std::endl;
  if (file_.is_open()) file_ << line << std::endl;
}

bool replace_file(const std::string& tmp, const std::string& dst) {
  std::error_code ec;
  for (int attempt = 0; attempt < 20; ++attempt) {
    std::filesystem::rename(tmp, dst, ec);
    if (!ec) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  std::filesystem::copy_file(tmp, dst, std::filesystem::copy_options::overwrite_existing, ec);
  std::error_code ec2;
  std::filesystem::remove(tmp, ec2);
  return !ec;
}

std::string timestamp() {
  auto now = std::chrono::system_clock::now();
  std::time_t t = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
#ifdef _WIN32
  localtime_s(&tm, &t);
#else
  localtime_r(&t, &tm);
#endif
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
  return buf;
}

std::string format(const char* fmt, ...) {
  char buf[2048];
  va_list args;
  va_start(args, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, args);
  va_end(args);
  return buf;
}

CsvWriter::CsvWriter(const std::string& path, const std::vector<std::string>& columns)
    : path_(path), columns_(columns) {
  std::error_code ec;
  bool fresh = !std::filesystem::exists(path_, ec) || std::filesystem::file_size(path_, ec) == 0;
  if (fresh) {
    std::ofstream out(path_);
    for (size_t i = 0; i < columns_.size(); ++i) out << (i ? "," : "") << columns_[i];
    out << "\n";
  }
}

void CsvWriter::write(const std::map<std::string, double>& row) {
  std::ofstream out(path_, std::ios::app);
  for (size_t i = 0; i < columns_.size(); ++i) {
    if (i) out << ",";
    auto it = row.find(columns_[i]);
    if (it != row.end()) {
      char buf[64];
      std::snprintf(buf, sizeof(buf), "%.6g", it->second);
      out << buf;
    }
  }
  out << "\n";
}

}  // namespace util
