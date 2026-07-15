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

    # -------------------------------------------------------------------
    # Which libomp.dylib do we actually LINK against?
    #
    # This extension is always loaded into a Python process alongside
    # PyTorch, which bundles its own copy of libomp.dylib. If we link
    # against a second, independent copy (e.g. Homebrew's), both runtimes
    # end up managing worker threads in the same process, and the two
    # implementations corrupt each other's internal thread state under
    # concurrent load (crashes inside __kmp_suspend_64 with a garbage/NULL
    # thread table pointer). Linking against the exact same libomp.dylib
    # that PyTorch loads avoids introducing that second runtime.
    #
    # We still need Homebrew's headers (omp.h) at compile time since
    # PyTorch's wheel does not ship them; the two builds are ABI-compatible
    # (both are upstream LLVM's OpenMP runtime), so mixing header source
    # and library source here is safe.
    # -------------------------------------------------------------------
    set(TORCH_OMP_LIBRARY "")
    set(TORCH_OMP_LIB_DIR "")

    if(Python_EXECUTABLE)
      execute_process(
        COMMAND "${Python_EXECUTABLE}" -c
          "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"
        RESULT_VARIABLE TORCH_LIBDIR_RESULT
        OUTPUT_VARIABLE TORCH_LIBDIR_OUTPUT
        ERROR_VARIABLE TORCH_LIBDIR_ERROR
        OUTPUT_STRIP_TRAILING_WHITESPACE
      )

      if(TORCH_LIBDIR_RESULT EQUAL 0 AND EXISTS "${TORCH_LIBDIR_OUTPUT}/libomp.dylib")
        set(TORCH_OMP_LIB_DIR "${TORCH_LIBDIR_OUTPUT}")
        set(TORCH_OMP_LIBRARY "${TORCH_LIBDIR_OUTPUT}/libomp.dylib")
      endif()
    endif()

    set(TORCH_OMP_INSTALL_NAME "")

    if(TORCH_OMP_LIBRARY)
      message(STATUS "Linking against PyTorch's bundled libomp: ${TORCH_OMP_LIBRARY}")

      set(
        OpenMP_omp_LIBRARY
        "${TORCH_OMP_LIBRARY}"
        CACHE FILEPATH "Path to the OpenMP runtime"
        FORCE
      )

      # Some PyTorch wheels ship a libomp.dylib whose own recorded install
      # name (LC_ID_DYLIB) is a hardcoded, build-machine-only path (e.g.
      # /opt/llvm-openmp/lib/libomp.dylib) instead of an @rpath-relative one.
      # dyld resolves a dependency using THAT self-declared ID, not the path
      # we used to locate the file while linking -- so if it's broken, _main
      # will fail to load at runtime with "Library not loaded: <bogus path>"
      # even though we linked against the correct file. Detect that here so
      # we can rewrite it after building (see the POST_BUILD step below).
      execute_process(
        COMMAND otool -D "${TORCH_OMP_LIBRARY}"
        RESULT_VARIABLE TORCH_OMP_OTOOL_RESULT
        OUTPUT_VARIABLE TORCH_OMP_OTOOL_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
      )

      if(TORCH_OMP_OTOOL_RESULT EQUAL 0)
        # otool -D prints the file path on line 1 and the install name on line 2.
        string(REPLACE "\n" ";" TORCH_OMP_OTOOL_LINES "${TORCH_OMP_OTOOL_OUTPUT}")
        list(LENGTH TORCH_OMP_OTOOL_LINES TORCH_OMP_OTOOL_NUM_LINES)
        if(TORCH_OMP_OTOOL_NUM_LINES GREATER_EQUAL 2)
          list(GET TORCH_OMP_OTOOL_LINES 1 TORCH_OMP_INSTALL_NAME)
        endif()
      endif()

      if(TORCH_OMP_INSTALL_NAME AND NOT TORCH_OMP_INSTALL_NAME STREQUAL "@rpath/libomp.dylib")
        message(
          STATUS
          "PyTorch's libomp.dylib has a non-relocatable install name "
          "(${TORCH_OMP_INSTALL_NAME}); will rewrite _main's reference to "
          "@rpath/libomp.dylib after building."
        )
      endif()
    else()
      message(
        WARNING
        "Could not locate a PyTorch-bundled libomp.dylib (is torch installed "
        "for ${Python_EXECUTABLE}?). Falling back to Homebrew's libomp; this "
        "extension will then load a *second*, independent OpenMP runtime "
        "alongside PyTorch's, which is known to crash under concurrent load "
        "on macOS (see __kmp_suspend_64 segfaults). Install torch before "
        "building, or set OMP_NUM_THREADS=1 / nnUNet_def_n_proc=1 as a "
        "workaround.\n"
        "Python_EXECUTABLE: ${Python_EXECUTABLE}\n"
        "exit code: ${TORCH_LIBDIR_RESULT}\n"
        "stdout: ${TORCH_LIBDIR_OUTPUT}\n"
        "stderr: ${TORCH_LIBDIR_ERROR}"
      )

      set(
        OpenMP_omp_LIBRARY
        "${LIBOMP_PREFIX}/lib/libomp.dylib"
        CACHE FILEPATH "Path to the OpenMP runtime"
        FORCE
      )
    endif()

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

if(APPLE AND TORCH_OMP_LIB_DIR)
  # Make sure dyld resolves libomp.dylib's @rpath-relative install name
  # (as used by PyTorch's wheel) to torch's copy, not some other libomp
  # found earlier on the default search path.
  set_target_properties(
    _main
    PROPERTIES
      BUILD_RPATH "${TORCH_OMP_LIB_DIR}"
      INSTALL_RPATH "${TORCH_OMP_LIB_DIR}"
  )

  if(TORCH_OMP_INSTALL_NAME AND NOT TORCH_OMP_INSTALL_NAME STREQUAL "@rpath/libomp.dylib")
    # torch's libomp.dylib has a broken (non-relocatable) install name; _main
    # currently references that literal broken path. Rewrite it to
    # @rpath/libomp.dylib so it resolves via the rpath set above instead.
    add_custom_command(
      TARGET _main POST_BUILD
      COMMAND install_name_tool -change
              "${TORCH_OMP_INSTALL_NAME}" "@rpath/libomp.dylib" "$<TARGET_FILE:_main>"
      COMMENT "Rewriting broken libomp.dylib install name in _main"
      VERBATIM
    )
  endif()
endif()

# The install directory is the output wheel directory.
install(TARGETS _main DESTINATION aladin)