#pragma once

#include <memory>
#include <string>

#include "play/engine.h"

namespace server {

struct ServerOptions {
  std::string host = "127.0.0.1";
  int port = 8080;
  std::string web_dir;   // directory containing index.html (auto-detected if empty)
  std::string run_dir;   // training run for the dashboard / "reload latest model"
  int default_visits = 400;
  bool open_browser = false;
};

extern std::string g_argv0_dir;  // directory of the executable (for finding web/)

int run_server(std::shared_ptr<play::PlayEngine> engine, const ServerOptions& opts);

}  // namespace server
