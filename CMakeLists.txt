cmake_minimum_required(VERSION 3.15)

project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX
)

set(ALADIN_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aladin/src")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT (UNIX OR WIN32))
  message(FATAL_ERROR "Unsupported target platform")
endif()

# ---------------------------------------------------------------------------
# Python / pybind11
# ---------------------------------------------------------------------------

find_package(
  Python
  REQUIRED
  COMPONENTS Interpreter Development.Module
)

find_package(pybind11 CONFIG REQUIRED)


# ---------------------------------------------------------------------------
# OpenMP
# ---------------------------------------------------------------------------
#
# Linux:
#   Usually detected automatically through GCC/libgomp.
#
# macOS:
#   Apple Clang does not provide OpenMP. Homebrew libomp is used only
#   during compilation. At runtime, ALADIN uses PyTorch's bundled
#   torch/lib/libomp.dylib.
# ---------------------------------------------------------------------------

if(APPLE)

  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
      "import pathlib, torch; p = pathlib.Path(torch.__file__).parent; print(p)"
    RESULT_VARIABLE TORCH_RESULT
    OUTPUT_VARIABLE TORCH_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  if(NOT TORCH_RESULT EQUAL 0)
    message(FATAL_ERROR "Could not locate the installed PyTorch package.")
  endif()

  find_file(
    TORCH_OMP_HEADER
    NAMES omp.h
    PATHS
      "${TORCH_DIR}/include"
    NO_DEFAULT_PATH
  )

  find_library(
    TORCH_OMP_LIBRARY
    NAMES omp
    PATHS
      "${TORCH_DIR}/lib"
    NO_DEFAULT_PATH
  )

  if(NOT TORCH_OMP_HEADER)
    message(FATAL_ERROR "Could not locate omp.h in the PyTorch installation.")
  endif()

  if(NOT TORCH_OMP_LIBRARY)
    message(FATAL_ERROR "Could not locate libomp.dylib in the PyTorch installation.")
  endif()

  get_filename_component(TORCH_OMP_INCLUDE_DIR "${TORCH_OMP_HEADER}" DIRECTORY)

  message(STATUS "Using PyTorch OpenMP header: ${TORCH_OMP_HEADER}")
  message(STATUS "Using PyTorch OpenMP runtime: ${TORCH_OMP_LIBRARY}")

  set(
    OpenMP_CXX_FLAGS
    "-Xpreprocessor -fopenmp"
    CACHE STRING ""
    FORCE
  )

  set(
    OpenMP_CXX_LIB_NAMES
    "omp"
    CACHE STRING ""
    FORCE
  )

  set(
    OpenMP_omp_LIBRARY
    "${TORCH_OMP_LIBRARY}"
    CACHE FILEPATH ""
    FORCE
  )

endif()

find_package(OpenMP REQUIRED COMPONENTS CXX)


# ---------------------------------------------------------------------------
# IIR library
# ---------------------------------------------------------------------------

set(
  LIBSRC
  ${ALADIN_SRC_DIR}/iir/Biquad.cpp
  ${ALADIN_SRC_DIR}/iir/Butterworth.cpp
  ${ALADIN_SRC_DIR}/iir/Cascade.cpp
  ${ALADIN_SRC_DIR}/iir/ChebyshevI.cpp
  ${ALADIN_SRC_DIR}/iir/ChebyshevII.cpp
  ${ALADIN_SRC_DIR}/iir/Custom.cpp
  ${ALADIN_SRC_DIR}/iir/PoleFilter.cpp
  ${ALADIN_SRC_DIR}/iir/RBJ.cpp
)

set(
  LIBINCLUDE
  ${ALADIN_SRC_DIR}/iir/Biquad.h
  ${ALADIN_SRC_DIR}/iir/Butterworth.h
  ${ALADIN_SRC_DIR}/iir/Cascade.h
  ${ALADIN_SRC_DIR}/iir/ChebyshevI.h
  ${ALADIN_SRC_DIR}/iir/ChebyshevII.h
  ${ALADIN_SRC_DIR}/iir/Common.h
  ${ALADIN_SRC_DIR}/iir/Custom.h
  ${ALADIN_SRC_DIR}/iir/Layout.h
  ${ALADIN_SRC_DIR}/iir/MathSupplement.h
  ${ALADIN_SRC_DIR}/iir/PoleFilter.h
  ${ALADIN_SRC_DIR}/iir/RBJ.h
  ${ALADIN_SRC_DIR}/iir/State.h
  ${ALADIN_SRC_DIR}/iir/Types.h
)

add_library(iir STATIC ${LIBSRC})
add_library(iir::iir ALIAS iir)

target_compile_features(iir PUBLIC cxx_std_11)

target_include_directories(
  iir
  PUBLIC
    $<BUILD_INTERFACE:${ALADIN_SRC_DIR}>
  PRIVATE
    $<BUILD_INTERFACE:${ALADIN_SRC_DIR}/iir>
)

set_target_properties(
  iir
  PROPERTIES
    POSITION_INDEPENDENT_CODE TRUE
    VERSION "${PROJECT_VERSION}"
    PUBLIC_HEADER ${ALADIN_SRC_DIR}/iir.h
    PRIVATE_HEADER "${LIBINCLUDE}"
)


# ---------------------------------------------------------------------------
# ALADIN Python extension
# ---------------------------------------------------------------------------

set(
  headers
  ${ALADIN_SRC_DIR}/asra/asra.h
  ${ALADIN_SRC_DIR}/asra/filter.h
  ${ALADIN_SRC_DIR}/reflect.h
  ${ALADIN_SRC_DIR}/cluster.h
  ${ALADIN_SRC_DIR}/common.h
  ${ALADIN_SRC_DIR}/pwaveprocessor.h
  ${ALADIN_SRC_DIR}/helpers.h
)

set(
  sources
  ${ALADIN_SRC_DIR}/asra/asra.cpp
  ${ALADIN_SRC_DIR}/asra/filter.cpp
  ${ALADIN_SRC_DIR}/bindings.cpp
  ${ALADIN_SRC_DIR}/reflect.cpp
  ${ALADIN_SRC_DIR}/common.cpp
  ${ALADIN_SRC_DIR}/pwaveprocessor.cpp
  ${ALADIN_SRC_DIR}/cluster.cpp
  ${ALADIN_SRC_DIR}/helpers.cpp
)

python_add_library(
  _main
  MODULE
  ${headers}
  ${sources}
  WITH_SOABI
)

target_link_libraries(
  _main
  PRIVATE
    pybind11::headers
    iir::iir
    OpenMP::OpenMP_CXX
)

# Homebrew's libomp headers are needed for compilation on macOS.
if(APPLE)

  target_include_directories(
    _main
    SYSTEM PRIVATE
    "${TORCH_OMP_INCLUDE_DIR}"
  )

  set_target_properties(
    _main
    PROPERTIES
      INSTALL_RPATH "@loader_path/../torch/lib"
  )

  add_custom_command(
    TARGET _main
    POST_BUILD
    COMMAND install_name_tool
      -change
      "/opt/llvm-openmp/lib/libomp.dylib"
      "@rpath/libomp.dylib"
      "$<TARGET_FILE:_main>"
    COMMENT "Rewriting libomp dependency to @rpath/libomp.dylib"
    VERBATIM
  )


endif()


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

install(
  TARGETS _main
  DESTINATION aladin
)
