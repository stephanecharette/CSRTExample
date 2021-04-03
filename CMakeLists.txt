# CSRT object tracking example (C) 2021 Stephane Charette <stephanecharette@gmail.com>
# MIT license applies.  See "license.txt" for details.

CMAKE_MINIMUM_REQUIRED (VERSION 3.0)

PROJECT (CSRTExample C CXX)

SET (CMAKE_BUILD_TYPE Release)
SET (CMAKE_CXX_STANDARD 17)
SET (CMAKE_CXX_STANDARD_REQUIRED ON)

FIND_PACKAGE (Threads	REQUIRED)
FIND_PACKAGE (OpenCV	REQUIRED)	# sudo apt-get install libopencv-dev

INCLUDE_DIRECTORIES (${OpenCV_INCLUDE_DIRS})

ADD_DEFINITIONS ("-Wall -Wextra -Werror -Wno-unused-parameter")

ADD_EXECUTABLE (CSRTExample main.cpp)
TARGET_LINK_LIBRARIES (CSRTExample Threads::Threads ${OpenCV_LIBS})
INSTALL (TARGETS CSRTExample DESTINATION bin)

