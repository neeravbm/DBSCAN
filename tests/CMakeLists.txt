INCLUDE_DIRECTORIES( SYSTEM ${PROJECT_SOURCE_DIR}/glog/src/)
INCLUDE_DIRECTORIES( SYSTEM ${PROJECT_SOURCE_DIR}/googletest/googletest/include )
INCLUDE_DIRECTORIES( SYSTEM ${PROJECT_SOURCE_DIR}/googletest/googlemock/include )

SET(MAIN_TEST_CPP ${PROJECT_SOURCE_DIR}/tests/main_test.cpp)

FIND_PACKAGE(GMock REQUIRED)

# Common function for test environment set up
FUNCTION( setupTests module sources libraries )
    ADD_DEFINITIONS( -DSOURCE_DIR=\"${CMAKE_SOURCE_DIR}/\" )
    ADD_DEFINITIONS( -DBIN_DIR=\"${CMAKE_BINARY_DIR}/\" )
    ADD_DEFINITIONS( -DCURRENT_TEST_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\" )
    ADD_DEFINITIONS( -DCURRENT_BIN_DIR=\"${CMAKE_CURRENT_BINARY_DIR}/\" )

    INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/.. ${GLOG_INCLUDE_DIRS} ${GTEST_INCLUDE_DIRS} ${GMOCK_INCLUDE_DIRS})

    SET( exeName test${module} )
    ADD_EXECUTABLE( ${exeName} ${sources} )

    SET_TARGET_PROPERTIES (${exeName} PROPERTIES
                            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

    #LINK_DIRECTORIES( ${PROJECT_BINARY_DIR}/${module} )

    TARGET_LINK_LIBRARIES( ${exeName} ${libraries} ${GLOG_LIBRARIES} ${GTEST_BOTH_LIBRARIES} ${GMOCK_BOTH_LIBRARIES} pthread )

ADD_TEST( ${exeName} ${CMAKE_BINARY_DIR}/tests/${exeName} --gtest_output=xml:${CMAKE_BINARY_DIR}/junit/${testResultName})
set_tests_properties(${exeName} PROPERTIES ENVIRONMENT "GLIBCXX_FORCE_NEW=1")
ENDFUNCTION( setupTests )

SET( SOURCES
    ${MAIN_TEST_CPP}
    test_vptree.cpp
    test_dbscan_vp.cpp
    test_dataset.cpp
)

SET( LIBRARIES )

setupTests( "dbscan" "${SOURCES}" "${LIBRARIES}" )

SET( SOURCES
    ${MAIN_TEST_CPP}
    test_benchmark.cpp
)

SET( LIBRARIES )

setupTests( "benchmark" "${SOURCES}" "${LIBRARIES}" )
