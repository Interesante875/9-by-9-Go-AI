// Tiny self-contained unit-test framework (no external dependency).
#pragma once

#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace testing {

struct TestCase {
  std::string name;
  std::function<void()> fn;
};

inline std::vector<TestCase>& registry() {
  static std::vector<TestCase> r;
  return r;
}

struct Registrar {
  Registrar(const std::string& name, std::function<void()> fn) { registry().push_back({name, std::move(fn)}); }
};

struct Failure {
  std::string msg;
};

}  // namespace testing

// LibTorch (glog-style) defines CHECK macros too; ours take precedence in test files.
#undef CHECK
#undef CHECK_EQ
#undef CHECK_NEAR

#define TEST(name)                                                   \
  static void test_##name();                                         \
  static testing::Registrar registrar_##name(#name, test_##name);   \
  static void test_##name()

#define CHECK(cond)                                                                                  \
  do {                                                                                               \
    if (!(cond)) {                                                                                   \
      std::ostringstream os_;                                                                        \
      os_ << __FILE__ << ":" << __LINE__ << ": CHECK(" #cond ") failed";                            \
      throw testing::Failure{os_.str()};                                                             \
    }                                                                                                \
  } while (0)

#define CHECK_EQ(a, b)                                                                               \
  do {                                                                                               \
    auto va_ = (a);                                                                                  \
    auto vb_ = (b);                                                                                  \
    if (!(va_ == vb_)) {                                                                             \
      std::ostringstream os_;                                                                        \
      os_ << __FILE__ << ":" << __LINE__ << ": CHECK_EQ(" #a ", " #b ") failed: " << va_ << " vs " << vb_; \
      throw testing::Failure{os_.str()};                                                             \
    }                                                                                                \
  } while (0)

#define CHECK_NEAR(a, b, tol)                                                                        \
  do {                                                                                               \
    double va_ = double(a), vb_ = double(b);                                                         \
    if (!(std::abs(va_ - vb_) <= (tol))) {                                                           \
      std::ostringstream os_;                                                                        \
      os_ << __FILE__ << ":" << __LINE__ << ": CHECK_NEAR(" #a ", " #b ") failed: " << va_ << " vs " << vb_; \
      throw testing::Failure{os_.str()};                                                             \
    }                                                                                                \
  } while (0)
