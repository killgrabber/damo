cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(damo_exe)
set(CMAKE_BUILD_TYPE Release)

file(GLOB SOURCES src/*.cpp include/damo/*.h)

find_package(PCL 1.12.1 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(${PROJECT_NAME} ${SOURCES})

include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(${PROJECT_NAME}
  PUBLIC ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
)

target_include_directories(${PROJECT_NAME}
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)
