cmake_minimum_required (VERSION 3.1)
project (hard_shadow)

find_package(OpenMP REQUIRED)

include_directories(${PROJECT_SOURCE_DIR})

add_library(tinyobj ${PROJECT_SOURCE_DIR}/tinyobjloader/tiny_obj_loader.cc)

set (CMAKE_CXX_STANDARD 11)

add_executable(hard_shadow main.cpp common.cpp mesh.cpp model_loader.cpp ppc.cpp)
target_link_libraries (hard_shadow PRIVATE tinyobj "${OpenMP_CXX_FLAGS}")
target_compile_options(hard_shadow PRIVATE "${OpenMP_CXX_FLAGS}")
