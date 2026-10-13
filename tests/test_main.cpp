#include <chrono>
#include <cstring>

#include "test_framework.h"

int main(int argc, char** argv) {
  const char* filter = argc > 1 ? argv[1] : nullptr;
  int passed = 0, failed = 0;
  for (auto& t : testing::registry()) {
    if (filter && t.name.find(filter) == std::string::npos) continue;
    auto t0 = std::chrono::steady_clock::now();
    try {
      t.fn();
      double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      std::cout << "[ PASS ] " << t.name << " (" << int(ms) << " ms)\n";
      ++passed;
    } catch (const testing::Failure& f) {
      std::cout << "[ FAIL ] " << t.name << "\n         " << f.msg << "\n";
      ++failed;
    } catch (const std::exception& e) {
      std::cout << "[ FAIL ] " << t.name << "\n         exception: " << e.what() << "\n";
      ++failed;
    }
  }
  std::cout << "\n" << passed << " passed, " << failed << " failed\n";
  return failed ? 1 : 0;
}
