if (DEFINED ENV{NEUWARE_HOME})
    set(BANG_FOUND TRUE)
    set(BANG_DIR $ENV{NEUWARE_HOME})
    set(BANG_INCLUDE_DIR ${BANG_DIR}/include)
    find_library(BANG_CNRT_LIB cnrt ${BANG_DIR}/lib64)
    find_library(DL_LIB dl)
endif ()

file(GLOB BANG_COMPILER_SRCS
        src/contrib/bang/codegen_bang.cc
        src/contrib/bang/modify_parallel_model.cc
        )
file(GLOB BANG_RUNTIME_SRCS
        src/contrib/bang/bang_module.cc
        src/contrib/bang/bang_device_api.cc
        src/contrib/bang/bang_common.cc
        )
list(APPEND COMPILER_SRCS ${BANG_COMPILER_SRCS})
message(STATUS "Build with BANG codegen")
if (BANG_FOUND)
    include_directories(${BANG_INCLUDE_DIR})
    list(APPEND RUNTIME_SRCS ${BANG_RUNTIME_SRCS})
    list(APPEND TVM_RUNTIME_LINKER_LIBS ${BANG_CNRT_LIB})
    list(APPEND TVM_RUNTIME_LINKER_LIBS ${DL_LIB})
    message(STATUS "Build with CNRT runtime")
endif ()