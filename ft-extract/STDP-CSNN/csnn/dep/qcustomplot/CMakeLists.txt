include_directories(${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB_RECURSE SOURCES qcustomplot.cpp)

add_library(qcustomplot SHARED ${SOURCES})
target_link_libraries(qcustomplot Qt4::QtCore Qt4::QtGui)
