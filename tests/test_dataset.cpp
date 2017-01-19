#include "gtest/gtest.h"

#include <dataset.h>

using namespace clustering;

namespace {
static const std::string CURRENT_TDIR( CURRENT_TEST_DIR );
}

TEST( Dataset, Iris )
{
    Dataset::Ptr dset = Dataset::create();
    ASSERT_TRUE( dset->load_csv( CURRENT_TDIR + "/csv/iris.data.txt" ) );
}