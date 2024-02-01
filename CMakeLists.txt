cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

# create project
project(DAMO)

# download CPM.cmake
file(
  DOWNLOAD
  https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.38.3/CPM.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake
  EXPECTED_HASH SHA256=cc155ce02e7945e7b8967ddfaff0b050e958a723ef7aad3766d368940cb15494
)
set(CPM_SOURCE_CACHE "$ENV{HOME}/.cpm-source-cache")
set(CPM_USE_NAMED_CACHE_DIRECTORIES ON)

include(${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake)

CPMAddPackage(
  NAME glvnd
  GIT_REPOSITORY https://github.com/NVIDIA/libglvnd.git
  VERSION "1.2.0"
)

CPMAddPackage(
  NAME flann
  GIT_REPOSITORY https://github.com/flann-lib/flann.git
  GIT_TAG "1.9.2"
)

#CPMAddPackage(
#  NAME eigen3
#  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
#  GIT_TAG "3.4.0"
#  OPTIONS
#    "EIGEN_TEST_X87 OFF"
#)

CPMAddPackage(
  NAME Boost
  VERSION 1.81.0
  GITHUB_REPOSITORY "boostorg/boost"
  GIT_TAG "boost-1.81.0"
)

CPMAddPackage(
  NAME vtk
  GIT_REPOSITORY https://gitlab.kitware.com/vtk/vtk.git
  VERSION "9.3.0"
)

#CPMAddPackage(
#  NAME PointCloudLibrary
#  VERSION 1.14.0
#  GIT_REPOSITORY "https://github.com/PointCloudLibrary/pcl.git"
#  GIT_TAG "pcl-1.14.0"
#)

add_subdirectory(src/damo)
