cmake_minimum_required(VERSION 3.4...3.18)
project(indexremove)

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.10.4
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(indexremove src/main.cpp)
target_include_directories(indexremove PUBLIC include)
target_compile_features(indexremove PUBLIC cxx_std_20)

install(TARGETS indexremove DESTINATION pythonmodules)
