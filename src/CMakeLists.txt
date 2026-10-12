CMAKE_MINIMUM_REQUIRED(VERSION 3.15)

# project setting
set(LIB_NAME "Simulator")

# set source and headers
set(CMAKE_CXX_STANDARD 23)
file(GLOB_RECURSE SRC_FILES
  "${CMAKE_SOURCE_DIR}/src/*.h"
  "${CMAKE_SOURCE_DIR}/src/*.cc"
)

# build
add_executable(${LIB_NAME} ${SRC_FILES})

# Export the executable's dynamic symbols (-rdynamic) so a dlopen'd TOG (.so)
# resolves the togsim_* runtime callbacks back into this binary.
set_target_properties(${LIB_NAME} PROPERTIES ENABLE_EXPORTS ON)

# TOGSim's TUs only declare togsim_abi_version; a TOG's TU defines it
# (togsim_runtime.h), so dlsym on the TOG's handle finds the TOG's own.
target_compile_definitions(${LIB_NAME} PRIVATE TOGSIM_HOST)
