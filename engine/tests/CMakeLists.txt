# =============================================================================
# Crucible — C++ unit tests (GoogleTest)
# =============================================================================
# One executable (crucible_tests) built from every tests/test_*.cpp; each
# GoogleTest case is registered with CTest via gtest_discover_tests.
# =============================================================================

set(CRUCIBLE_GTEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/googletest)

if(NOT EXISTS "${CRUCIBLE_GTEST_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "googletest submodule not initialised.\n"
        "Run from repo root:  git submodule update --init --depth 1 engine/third_party/googletest"
    )
endif()

set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK   OFF CACHE BOOL "" FORCE)
# MSVC: googletest defaults to the static CRT (/MT) while our targets use
# /MD; mixing them fails with LNK2038. Use the shared CRT everywhere.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
add_subdirectory(${CRUCIBLE_GTEST_DIR} gtest_build EXCLUDE_FROM_ALL)

# Every test_*.cpp in this directory is picked up on the next configure.
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
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/..
    PROPERTIES
        LABELS "unit"
)
