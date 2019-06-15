cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 14)

project(zma VERSION 1.0.0.0 LANGUAGES CXX)

option(DEVELOPMENT "Generate a development build" OFF)

set(PROJECT_NAME zma)

set(DARWIN_LIB_DIR ${CMAKE_SOURCE_DIR}/Engine/_Drivers/darwin)

set(SOURCES 
${CMAKE_SOURCE_DIR}/includes/zma.hpp
${CMAKE_SOURCE_DIR}/tests/zma_unit_tests.cpp
${CMAKE_SOURCE_DIR}/thirdparty/catch.hpp
)

set(INCLUDES
${CMAKE_SOURCE_DIR}/includes
${CMAKE_SOURCE_DIR}/thirdparty
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build)

if(DEVELOPMENT)
	add_compile_definitions(DEV_BUILD)
endif(DEVELOPMENT)

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PUBLIC ${INCLUDES})
