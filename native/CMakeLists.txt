cmake_minimum_required(VERSION 3.20)
project(valle_tts_native LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INSTALL_LIBDIR lib CACHE STRING "Static library install path" FORCE)
add_compile_definitions(GGML_MAX_NAME=128 QWEN_STATIC)
if(MSVC)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX)
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>)
endif()
find_package(Threads REQUIRED)
add_subdirectory(../vendor/ggml ggml)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/version.h "#define QWEN_VERSION \"51512f1 (valle-tts pinned)\"\n")
add_library(valle_qwen STATIC
  bridge.cpp
  ../vendor/qwentts/src/qwen.cpp
  ../vendor/qwentts/src/pipeline-tts.cpp
  ../vendor/qwentts/src/pipeline-codec.cpp)
target_include_directories(valle_qwen PRIVATE ../vendor/qwentts/src ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(valle_qwen PRIVATE ggml ggml-cpu ggml-base Threads::Threads)
install(TARGETS valle_qwen ARCHIVE DESTINATION lib)
