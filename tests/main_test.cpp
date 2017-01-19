#include <cstdlib>
#include <ctime>
#include <random>

#include "gtest/gtest.h"

int main( int argc, char** argv )
{
    std::random_device rd;
    std::mt19937 mt( rd() );

    ::testing::InitGoogleTest( &argc, argv );

    return RUN_ALL_TESTS();
}
