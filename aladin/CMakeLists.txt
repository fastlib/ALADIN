cmake_minimum_required(VERSION 3.15)
project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX)

if (NOT (UNIX OR WIN32))
  message(FATAL_ERROR "unsupported target platform")
endif()

# TODO: Use `Development.Embed` component when requiring cmake >= 3.18
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

link_directories(${PROJECT_SOURCE_DIR})

# if(NOT CMAKE_BUILD_TYPE)
#   set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
# endif()

set(LIBSRC
  src/iir/Biquad.cpp
  src/iir/Butterworth.cpp
  src/iir/Cascade.cpp
  src/iir/ChebyshevI.cpp
  src/iir/ChebyshevII.cpp
  src/iir/Custom.cpp
  src/iir/PoleFilter.cpp
  src/iir/RBJ.cpp)

set(LIBINCLUDE
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
  src/iir/Types.h)

add_library(iir STATIC ${LIBSRC})
add_library(iir::iir ALIAS iir)
target_compile_features(iir PUBLIC cxx_std_11)
target_include_directories(iir
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>      # for Iir.h
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/iir>  # everything else
)
set_target_properties(iir PROPERTIES
  POSITION_INDEPENDENT_CODE TRUE
  VERSION ${PROJECT_VERSION}
  PUBLIC_HEADER src/iir.h
  PRIVATE_HEADER "${LIBINCLUDE}"
)

set(headers
    src/asra/asra.h
    src/asra/filter.h
    src/reflect.h
    src/cluster.h
    src/common.h
    src/pwaveprocessor.h
    src/helpers.h
)
set(sources
    src/asra/asra.cpp
    src/asra/filter.cpp
    src/bindings.cpp
    src/reflect.cpp
    src/common.cpp
    src/pwaveprocessor.cpp
    src/cluster.cpp
    src/helpers.cpp
)

python_add_library(_main MODULE ${headers} ${sources} WITH_SOABI)
target_link_libraries(_main PRIVATE pybind11::headers)
target_link_libraries(_main PRIVATE iir::iir)
target_link_libraries(_main PRIVATE OpenMP::OpenMP_CXX)

# The install directory is the output (wheel) directory
install(TARGETS _main DESTINATION aladin)
