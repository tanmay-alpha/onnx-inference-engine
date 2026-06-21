# =============================================================================
# Crucible — C++ unit tests (GoogleTest)
# =============================================================================
# Issue #1: scaffold the testing target. Real tests arrive alongside each
# module (test_tensor.cpp in Issue #2, test_onnx_parser.cpp in #4, etc.).
# =============================================================================

set(CRUCIBLE_GTEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/googletest)

if(NOT EXISTS "${CRUCIBLE_GTEST_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "googletest submodule not initialised.\n"
        "Run from repo root:  git submodule update --init --recursive"
    )
endif()

set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK   OFF CACHE BOOL "" FORCE)
add_subdirectory(${CRUCIBLE_GTEST_DIR} gtest_build EXCLUDE_FROM_ALL)

# Discover all test_*.cpp files. Issue #1 contributes a single smoke test
# (see test_smoke.cpp). Future issues drop in test_tensor.cpp etc. and they
# will be picked up automatically on the next CMake configure.
file(GLOB CRUCIBLE_TEST_SOURCES CONFIGURE_DEPENDS "test_*.cpp")

add_executable(crucible_tests ${CRUCIBLE_TEST_SOURCES})

target_link_libraries(crucible_tests
    PRIVATE
        crucible::core
        GTest::gtest_main
)

target_include_directories(crucible_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(CRUCIBLE_WARNINGS_AS_ERRORS)
    target_compile_options(crucible_tests PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Werror>
        $<$<CXX_COMPILER_ID:MSVC>:/WX>
    )
endif()

include(GoogleTest)
gtest_discover_tests(crucible_tests
    PROPERTIES
        LABELS "unit"
)
