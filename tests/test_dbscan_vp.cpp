#include "gtest/gtest.h"

#include <cmath>
#include <iostream>
#include <numeric>
#include <cassert>

#ifdef _OPENMP
  #include <omp.h>
#endif

#include <dataset.h>
#include <Eigen/Dense>

#include <glog/logging.h>

#include "dbscan_vp.h"

namespace {
static const std::string CURRENT_TDIR( CURRENT_TEST_DIR );
}

using namespace clustering;

TEST( DBSCAN_VP, TwoClusters )
{
    Dataset::Ptr dset = Dataset::create();
    dset->load_csv( CURRENT_TDIR + "/csv/vptree01.csv" );

    DBSCAN_VP::Ptr dbs = boost::make_shared< DBSCAN_VP >( dset );
    dbs->fit();
    dbs->predict( 0.01, 5 );

    const DBSCAN_VP::Labels& l = dbs->get_labels();

    for ( size_t i = 0; i < l.size(); ++i ) {
        LOG( INFO ) << "Element = " << i << " cluster = " << l[i];
        if ( i < 5 ) {
            EXPECT_EQ( l[i], 0 );
        } else {
            EXPECT_EQ( l[i], 1 );
        }
    }
}

TEST( DBSCAN_VP, OneCluster )
{
    Dataset::Ptr dset = Dataset::create();
    dset->load_csv( CURRENT_TDIR + "/csv/vptree02.csv" );

    DBSCAN_VP::Ptr dbs = boost::make_shared< DBSCAN_VP >( dset );
    dbs->fit();
    dbs->predict( 0.01, 5 );

    const DBSCAN_VP::Labels& l = dbs->get_labels();

    for ( size_t i = 0; i < l.size(); ++i ) {
        LOG( INFO ) << "Element = " << i << " cluster = " << l[i];
        if ( i < 6 ) {
            EXPECT_EQ( l[i], 0 );
        } else {
            EXPECT_EQ( l[i], -1 );
        }
    }
}

TEST( DBSCAN_VP, NoClusters )
{
    Dataset::Ptr dset = Dataset::create();
    dset->load_csv( CURRENT_TDIR + "/csv/vptree03.csv" );

    DBSCAN_VP::Ptr dbs = boost::make_shared< DBSCAN_VP >( dset );
    dbs->fit();
    dbs->predict( 0.01, 2 );

    const DBSCAN_VP::Labels& l = dbs->get_labels();

    for ( size_t i = 0; i < l.size(); ++i ) {
        LOG( INFO ) << "Element = " << i << " cluster = " << l[i];
        EXPECT_EQ( l[i], -1 );
    }
}

TEST( DBSCAN_VP, Iris )
{
    Dataset::Ptr dset = Dataset::create();
    dset->load_csv( CURRENT_TDIR + "/csv/iris.data.txt" );

    DBSCAN_VP::Ptr dbs = boost::make_shared< DBSCAN_VP >( dset );

    dbs->fit();
    dbs->predict( 0.4, 5 );

    const DBSCAN_VP::Labels& l = dbs->get_labels();

    for ( size_t i = 0; i < l.size(); ++i ) {
        LOG( INFO ) << "Element = " << i << " cluster = " << l[i];
    }
}

TEST( DBSCAN_VP, IrisAnalyze )
{
    Dataset::Ptr dset = Dataset::create();
    dset->load_csv( CURRENT_TDIR + "/csv/iris.data.txt" );

    DBSCAN_VP::Ptr dbs = boost::make_shared< DBSCAN_VP >( dset );

    dbs->fit();
    const auto r = dbs->predict_eps( 3u );

    for ( size_t i = 0; i < r.size(); ++i ) {
        std::cout << ( i + 1 ) << "," << r[i] << std::endl;
    }
}
