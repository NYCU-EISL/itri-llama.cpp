set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)
set(CMAKE_SYSTEM_VERSION 1)

if (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^(riscv)")
    message(STATUS "HOST SYSTEM ${CMAKE_HOST_SYSTEM_PROCESSOR}")
else()
    set(GNU_MACHINE riscv64-unknown-linux-gnu CACHE STRING "GNU compiler triple")
    if (DEFINED ENV{RISCV_ROOT_PATH})
        file(TO_CMAKE_PATH $ENV{RISCV_ROOT_PATH} RISCV_ROOT_PATH)
    else()
        message(FATAL_ERROR "RISCV_ROOT_PATH env must be defined")
    endif()

    set(RISCV_ROOT_PATH ${RISCV_ROOT_PATH} CACHE STRING "root path to riscv toolchain")
    set(CMAKE_C_COMPILER ${RISCV_ROOT_PATH}/bin/riscv64-unknown-linux-gcc)
    set(CMAKE_CXX_COMPILER ${RISCV_ROOT_PATH}/bin/riscv64-unknown-linux-g++)
    set(CMAKE_STRIP ${RISCV_ROOT_PATH}/bin/riscv64-unknown-linux-strip)
    set(CMAKE_FIND_ROOT_PATH "${RISCV_ROOT_PATH}/riscv64-unknown-linux")
    set(CMAKE_SYSROOT "${RISCV_ROOT_PATH}/sysroot")
endif()

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_CXX_FLAGS " ${CXX_FLAGS} -O3")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -latomic")


#[[

export RISCV_ROOT_PATH=/home/hyy/Documents/ITRI/BSP/toolchains/nds64le-linux-glibc-v5d/
cmake -B build-qilai -DGGML_CPU_QILAI=ON -DLLAMA_CURL=OFF -DGGML_RVV=OFF -DGGML_RV_ZFH=ON -DGGML_RV_ZICBOP=ON -DCMAKE_TOOLCHAIN_FILE=/home/hyy/Documents/llama.cpp/cmake/riscv64-andes-qilai-linux-gnu-gcc.cmake
cmake --build build-qilai -j$(nproc)

]]