cmake_minimum_required(VERSION 3.15)

project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX
)

if(NOT (UNIX OR WIN32))
  message(FATAL_ERROR "Unsupported target platform")
endif()

# Python and pybind11
find_package(
  Python
  REQUIRED
  COMPONENTS Interpreter Development.Module
)

find_package(pybind11 CONFIG REQUIRED)

# ---------------------------------------------------------------------------
# OpenMP
#
# Linux:
#   Normally detected automatically when GCC and libgomp are installed.
#
# macOS:
#   Apple Clang does not ship with OpenMP. Homebrew installs libomp as
#   keg-only, so CMake needs explicit compiler flags and library paths.
# ---------------------------------------------------------------------------

if(APPLE)
  # First allow the user or CI environment to provide a libomp prefix.
  set(LIBOMP_PREFIX "" CACHE PATH "Path to the libomp installation")

  # If no prefix was supplied, try to retrieve it from Homebrew.
  if(NOT LIBOMP_PREFIX)
    find_program(HOMEBREW_EXECUTABLE brew)

    if(HOMEBREW_EXECUTABLE)
      execute_process(
        COMMAND "${HOMEBREW_EXECUTABLE}" --prefix libomp
        RESULT_VARIABLE LIBOMP_RESULT
        OUTPUT_VARIABLE LIBOMP_BREW_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
      )

      if(LIBOMP_RESULT EQUAL 0 AND EXISTS "${LIBOMP_BREW_PREFIX}")
        set(LIBOMP_PREFIX "${LIBOMP_BREW_PREFIX}")
      endif()
    endif()
  endif()

  # Common Homebrew fallback paths.
  if(NOT LIBOMP_PREFIX)
    if(EXISTS "/opt/homebrew/opt/libomp")
      set(LIBOMP_PREFIX "/opt/homebrew/opt/libomp")
    elseif(EXISTS "/usr/local/opt/libomp")
      set(LIBOMP_PREFIX "/usr/local/opt/libomp")
    endif()
  endif()

  if(LIBOMP_PREFIX)
    message(STATUS "Using libomp from: ${LIBOMP_PREFIX}")

    set(
      OpenMP_CXX_FLAGS
      "-Xpreprocessor -fopenmp"
      CACHE STRING "OpenMP C++ compiler flags"
      FORCE
    )

    set(
      OpenMP_CXX_LIB_NAMES
      "omp"
      CACHE STRING "OpenMP C++ library names"
      FORCE
    )

    set(
      OpenMP_omp_LIBRARY
      "${LIBOMP_PREFIX}/lib/libomp.dylib"
      CACHE FILEPATH "Path to the OpenMP runtime"
      FORCE
    )

    # omp.h is located here for Homebrew's keg-only libomp installation.
    include_directories(SYSTEM "${LIBOMP_PREFIX}/include")
  else()
    message(
      FATAL_ERROR
      "OpenMP was not found on macOS.\n"
      "Install it with:\n"
      "  brew install libomp\n"
      "Alternatively, pass its location explicitly:\n"
      "  cmake -DLIBOMP_PREFIX=/path/to/libomp ..."
    )
  endif()
endif()

find_package(OpenMP REQUIRED COMPONENTS CXX)

link_directories("${PROJECT_SOURCE_DIR}")

# if(NOT CMAKE_BUILD_TYPE)
#   set(
#     CMAKE_BUILD_TYPE
#     Debug
#     CACHE STRING "Choose the type of build."
#     FORCE
#   )
# endif()

set(
  LIBSRC
  src/iir/Biquad.cpp
  src/iir/Butterworth.cpp
  src/iir/Cascade.cpp
  src/iir/ChebyshevI.cpp
  src/iir/ChebyshevII.cpp
  src/iir/Custom.cpp
  src/iir/PoleFilter.cpp
  src/iir/RBJ.cpp
)

set(
  LIBINCLUDE
  src/iir/Biquad.h
  src/iir/Butterworth.h
  src/iir/Cascade.h
  src/iir/ChebyshevI.h
  src/iir/ChebyshevII.h
  src/iir/Common.h
  src/iir/Custom.h
  src/iir/Layout.h
  src/iir/MathSupplement.h
  src/iir/PoleFilter.h
  src/iir/RBJ.h
  src/iir/State.h
  src/iir/Types.h
)

add_library(iir STATIC ${LIBSRC})
add_library(iir::iir ALIAS iir)

target_compile_features(iir PUBLIC cxx_std_11)

target_include_directories(
  iir
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/iir>
)

set_target_properties(
  iir
  PROPERTIES
    POSITION_INDEPENDENT_CODE TRUE
    VERSION "${PROJECT_VERSION}"
    PUBLIC_HEADER src/iir.h
    PRIVATE_HEADER "${LIBINCLUDE}"
)

set(
  headers
  src/asra/asra.h
  src/asra/filter.h
  src/reflect.h
  src/cluster.h
  src/common.h
  src/pwaveprocessor.h
  src/helpers.h
)

set(
  sources
  src/asra/asra.cpp
  src/asra/filter.cpp
  src/bindings.cpp
  src/reflect.cpp
  src/common.cpp
  src/pwaveprocessor.cpp
  src/cluster.cpp
  src/helpers.cpp
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

# The install directory is the output wheel directory.
install(TARGETS _main DESTINATION aladin)