# tensor_tiling_test: the unit test source and Instruction.cc, compiled and linked as the simulator's sources are.
set(CMAKE_CXX_STANDARD 23)
add_executable(tensor_tiling_test tensor_tiling.cc "${PROJECT_SOURCE_DIR}/src/Instruction.cc")
target_compile_definitions(tensor_tiling_test PRIVATE TOGSIM_HOST)
target_include_directories(tensor_tiling_test PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(tensor_tiling_test ${CONAN_LIBS})
