project(DBSCAN)

CMAKE_MINIMUM_REQUIRED( VERSION 2.8 )

FIND_PACKAGE(OpenMP)
SET(CMAKE_C_FLAGS "-std=gnu99 -pedantic -fPIC -pedantic -mtune=native ${OpenMP_C_FLAGS}")
SET(CMAKE_CXX_FLAGS "-std=c++11 -Wall -fPIC -pedantic -mtune=native ${OpenMP_CXX_FLAGS}")

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON) 
set(Boost_USE_STATIC_RUNTIME OFF)

FIND_PACKAGE(PythonInterp)
FIND_PACKAGE(PythonLibs)
#find_package(Boost REQUIRED COMPONENTS program_options python)
find_package(Boost COMPONENTS program_options OPTIONAL_COMPONENTS python)
if (NOT Boost_PROGRAM_OPTIONS_FOUND)
  MESSAGE(FATAL_ERROR "Boost's program_options library not found.")
endif (NOT Boost_PROGRAM_OPTIONS_FOUND)

include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
message(STATUS "Using boost include dirs: ${Boost_INCLUDE_DIRS}")
message(STATUS "Using boost library dirs: ${Boost_LIBRARY_DIRS}")
message(STATUS "Using boost libraries: ${Boost_PROGRAM_OPTIONS_LIBRARIES} ${Boost_PYTHON_LIBRARIES}")
message(STATUS "Using boost thread library: ${Boost_THREAD_LIBRARY}")
link_directories(${Boost_LIBRARY_DIRS})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
      set(CMAKE_CXX_FLAGS_DEBUG "-g3" CACHE STRING "Debug CXX compiler flags" FORCE)
      set(CMAKE_C_FLAGS_DEBUG "-g3" CACHE STRING "Debug C compiler flags" FORCE)
endif(CMAKE_BUILD_TYPE STREQUAL "Debug")

SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin )
SET( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib )
SET( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib )

#add_subdirectory(glog)
SET( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake )
FIND_PACKAGE(Glog REQUIRED)
#add_subdirectory(googletest)
FIND_PACKAGE(GTest REQUIRED)
FIND_PACKAGE(EIGEN3 REQUIRED)

include_directories(
	${CMAKE_CURRENT_SOURCE_DIR} 
        ${EIGEN3_INCLUDE_DIRS}
        ${GTEST_INCLUDE_DIRS}
        ${GLOG_INCLUDE_DIRS}
	)

add_definitions(-DBOOST_UBLAS_NDEBUG)

add_library(dbscan dbscan.cpp)
target_link_libraries(dbscan ${GTEST_BOTH_LIBRARIES} ${Boost_PROGRAM_OPTIONS_LIBRARIES} ${Boost_PYTHON_LIBRARIES} ${GLOG_LIBRARIES})
SET(DBSCAN_LIB_NAME "$<TARGET_FILE_NAME:dbscan>")

#add_executable(test_dbscan test_dbscan.cpp)
#target_link_libraries(test_dbscan dbscan)

if (PYTHONLIBS_FOUND AND Boost_PYTHON_FOUND)
	include_directories( 
		${PYTHON_INCLUDE_PATH} 
	)
	         
	ADD_LIBRARY(pydbscan SHARED PyDBSCAN.cpp)
	target_link_libraries(pydbscan dbscan boost_python)

	SET_TARGET_PROPERTIES(pydbscan
		PROPERTIES
		    MAJOR_VERSION 1
		    MINOR_VERSION 0
		        LIBRARY_OUTPUT_NAME pydbscan
		        PREFIX ""
	)

endif(PYTHONLIBS_FOUND AND Boost_PYTHON_FOUND)

# add_executable(vptree main_vptree.cpp)



# add_executable(dset_test dataset_test.cpp)
# add_executable(dbscan_vp test_dbscan_vp.cpp)

enable_testing()

add_subdirectory(tests)
add_subdirectory(tools)

get_filename_component(PROJECT_INSTALL_DIR ${CMAKE_INSTALL_PREFIX} ABSOLUTE BASE_DIR ${PROJECT_BINARY_DIR})

CONFIGURE_FILE(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/DBSCANConfig.cmake.in
    ${PROJECT_BINARY_DIR}/cmake/DBSCANConfig.cmake_at_configured
    @ONLY
)

# Deal with items configured with generator expressions:
file(GENERATE
     OUTPUT
     ${PROJECT_BINARY_DIR}/cmake/DBSCANConfig.cmake
     INPUT
     ${PROJECT_BINARY_DIR}/cmake/DBSCANConfig.cmake_at_configured
     )

CONFIGURE_FILE(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/DBSCANConfigVersion.cmake.in
    ${PROJECT_BINARY_DIR}/cmake/DBSCANConfigVersion.cmake
    @ONLY
)

INSTALL(
    FILES 
    ${PROJECT_BINARY_DIR}/cmake/DBSCANConfig.cmake
    ${PROJECT_BINARY_DIR}/cmake/DBSCANConfigVersion.cmake
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindGlog.cmake
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindGMock.cmake
    DESTINATION share/DBSCAN/cmake
)

set_target_properties(dbscan PROPERTIES PUBLIC_HEADER "dbscan.h")
install(TARGETS dbscan
	EXPORT DBSCANTargets
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	PUBLIC_HEADER DESTINATION include
	COMPONENT Devel)
