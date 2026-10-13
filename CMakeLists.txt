cmake_minimum_required(VERSION 3.18)
project(go9 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)  # recent LibTorch releases require C++20
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(GO9_BUILD_TESTS "Build unit tests" ON)

# LibTorch can be found in two ways:
#  1) -DTORCH_ROOT=<libtorch dir>  (contains include/ and lib/). Links the libraries directly, so a
#     CUDA build of LibTorch works WITHOUT installing the CUDA toolkit (recommended on Windows).
#     A pip-installed torch works too: TORCH_ROOT = <site-packages>/torch
#  2) -DCMAKE_PREFIX_PATH=<libtorch dir>  - classic find_package(Torch). For CUDA builds of LibTorch
#     this requires the matching CUDA toolkit to be installed.
set(TORCH_ROOT "" CACHE PATH "LibTorch root directory (direct linking, no CUDA toolkit needed)")
find_package(Threads REQUIRED)
if(TORCH_ROOT)
  message(STATUS "go9: linking LibTorch directly from ${TORCH_ROOT}")
  add_library(go9_torch INTERFACE)
  target_include_directories(go9_torch SYSTEM INTERFACE
    ${TORCH_ROOT}/include ${TORCH_ROOT}/include/torch/csrc/api/include)
  foreach(lib torch torch_cpu c10)
    find_library(GO9_LIB_${lib} ${lib} PATHS ${TORCH_ROOT}/lib NO_DEFAULT_PATH REQUIRED)
    target_link_libraries(go9_torch INTERFACE ${GO9_LIB_${lib}})
  endforeach()
  foreach(lib torch_cuda c10_cuda)
    find_library(GO9_LIB_${lib} ${lib} PATHS ${TORCH_ROOT}/lib NO_DEFAULT_PATH)
    if(GO9_LIB_${lib})
      message(STATUS "go9: CUDA library found: ${GO9_LIB_${lib}}")
      if(UNIX)
        # Keep libtorch_cuda as a dependency even though no symbol is referenced directly,
        # otherwise the CUDA backend is never registered.
        target_link_libraries(go9_torch INTERFACE "-Wl,--no-as-needed" ${GO9_LIB_${lib}} "-Wl,--as-needed")
      else()
        target_link_libraries(go9_torch INTERFACE ${GO9_LIB_${lib}})
      endif()
    endif()
  endforeach()
  if(UNIX)
    target_link_options(go9_torch INTERFACE "-Wl,-rpath,${TORCH_ROOT}/lib")
  endif()
  set(TORCH_LIBRARIES go9_torch)
  set(TORCH_INSTALL_PREFIX ${TORCH_ROOT})
  set(TORCH_CXX_FLAGS_LIST "")
else()
  find_package(Torch REQUIRED)
  separate_arguments(TORCH_CXX_FLAGS_LIST UNIX_COMMAND "${TORCH_CXX_FLAGS}")
endif()

if(MSVC)
  add_compile_options(/utf-8 /bigobj /permissive- /W3)
  add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
else()
  add_compile_options(-Wall -Wextra -Wno-unused-parameter)
endif()

# ---------------------------------------------------------------- core (no torch)
add_library(go9_core STATIC
  src/go/board.cpp
  src/go/env.cpp
  src/nn/features.cpp
)
target_include_directories(go9_core PUBLIC src)

# ---------------------------------------------------------------- engine (torch)
add_library(go9_engine STATIC
  src/nn/model.cpp
  src/nn/evaluator.cpp
  src/mcts/mcts.cpp
  src/train/replay.cpp
  src/train/config.cpp
  src/train/selfplay.cpp
  src/train/trainer.cpp
  src/train/arena.cpp
  src/util/log.cpp
  src/play/engine.cpp
)
target_include_directories(go9_engine PUBLIC src)
target_link_libraries(go9_engine PUBLIC go9_core ${TORCH_LIBRARIES} Threads::Threads)
target_compile_options(go9_engine PUBLIC ${TORCH_CXX_FLAGS_LIST})

# ---------------------------------------------------------------- executable
add_executable(go9
  src/main.cpp
  src/server/server.cpp
  src/gtp/gtp.cpp
)
target_include_directories(go9 PRIVATE src third_party/httplib)
target_link_libraries(go9 PRIVATE go9_engine)
if(WIN32)
  target_link_libraries(go9 PRIVATE ws2_32)
endif()

# Copy the web GUI next to the executable.
add_custom_command(TARGET go9 POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/web $<TARGET_FILE_DIR:go9>/web)

# On Windows the LibTorch DLLs must sit next to the executable.
if(MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  if(TORCH_DLLS)
    add_custom_command(TARGET go9 POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS} $<TARGET_FILE_DIR:go9>)
  endif()
endif()

# ---------------------------------------------------------------- tests
if(GO9_BUILD_TESTS)
  enable_testing()
  add_executable(go9_tests
    tests/test_main.cpp
    tests/test_board.cpp
    tests/test_env.cpp
    tests/test_replay.cpp
    tests/test_mcts.cpp
    tests/test_nn.cpp
  )
  target_include_directories(go9_tests PRIVATE src tests)
  target_link_libraries(go9_tests PRIVATE go9_engine)
  add_test(NAME go9_tests COMMAND go9_tests)

  add_executable(go9_crossval tests/crossval_main.cpp)
  target_link_libraries(go9_crossval PRIVATE go9_core)
  if(EXISTS ${CMAKE_SOURCE_DIR}/tests/data/openspiel_games.txt)
    add_test(NAME crossval_openspiel COMMAND go9_crossval ${CMAKE_SOURCE_DIR}/tests/data/openspiel_games.txt)
  endif()
  if(MSVC AND TORCH_DLLS)
    add_custom_command(TARGET go9_tests POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS} $<TARGET_FILE_DIR:go9_tests>)
  endif()
endif()
