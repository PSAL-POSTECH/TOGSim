cmake_minimum_required(VERSION 3.15.0)
set(project_name "AiFrameworkSim")
project(${project_name})

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include("${CMAKE_SOURCE_DIR}/build/conanbuildinfo.cmake")
conan_basic_setup()

# C++ setttings
set(CMAKE_CXX_STANDARD 17)
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/build/bin")
set(LIBRARY_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/build/lib")
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
message("BINARY DIR ${CMAKE_BINARY_DIR}")
# Build source
add_subdirectory("${PROJECT_SOURCE_DIR}/src")

# Add libaray ramulator
include_directories("${PROJECT_SOURCE_DIR}/include")
set(RAMULATOR_PYTHON_BINDINGS OFF CACHE BOOL "TOGSim links the library only")
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/ramulator2")

# Add libaray booksim
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/booksim")

target_include_directories(Simulator PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(Simulator PUBLIC ${PROJECT_SOURCE_DIR}/include/scheduler)
target_include_directories(Simulator PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(Simulator booksim2 ramulator yaml-cpp ${CONAN_LIBS} stdc++fs)

# Build the unit test the test runner executes (tests/run.py)
add_subdirectory("${PROJECT_SOURCE_DIR}/tests/unit")
