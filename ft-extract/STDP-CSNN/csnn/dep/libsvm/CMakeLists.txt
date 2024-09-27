include_directories(${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB_RECURSE SOURCES svm.cpp)

add_library(libsvm SHARED ${SOURCES})
