#include "gtest/gtest.h"

#include <cmath>
#include <iostream>
#include <numeric>
#include <cassert>

#ifdef _OPENMP
  #include <omp.h>
#endif

#include <Eigen/Dense>

#include "vptree.h"
#include "dataset.h"

using namespace clustering;

namespace {
static const std::string CURRENT_TDIR( CURRENT_TEST_DIR );
static const float MIN_T = 0.001;

inline double dist( const Eigen::VectorXf& p1, const Eigen::VectorXf& p2 )
{
    return ( p1 - p2 ).norm();
}

typedef VPTREE< Eigen::VectorXf, dist > TTree;
}

TEST( VPTree, DistSearch )
{
    Dataset::Ptr dset = Dataset::create();
    ASSERT_TRUE( dset->load_csv( CURRENT_TDIR + "/csv/vptree01.csv" ) );

    TTree tree;
    tree.create( dset );

    const Dataset::DataContainer& d = dset->data();

    for ( size_t i = 0; i < d.size(); ++i ) {

        TTree::TNeighborsList nlist;

        LOG( INFO ) << "Searching for " << i << " [" << d[i] << "]";

        tree.search_by_dist( d[i], MIN_T, nlist );

        EXPECT_EQ( nlist.size(), 5u );

        for ( size_t j = 0; j < nlist.size(); ++j ) {
            LOG( INFO ) << "Found neighbor id = " << nlist[j].first << " dist= " << nlist[j].second << " [" << d[nlist[j].first] << "]";
            EXPECT_TRUE( dist( d[i], d[nlist[j].first] ) < MIN_T );
        }
    }
}

TEST( VPTree, KNSearch )
{
    Dataset::Ptr dset = Dataset::create();
    ASSERT_TRUE( dset->load_csv( CURRENT_TDIR + "/csv/vptree01.csv" ) );

    TTree tree;
    tree.create( dset );

    const Dataset::DataContainer& d = dset->data();

    for ( size_t i = 0; i < d.size(); ++i ) {

        TTree::TNeighborsList nlist;

        LOG( INFO ) << "Searching for " << i << " [" << d[i] << "]";

        tree.search_by_k( d[i], 5u, nlist );

        EXPECT_EQ( nlist.size(), 5u );

        for ( size_t j = 0; j < nlist.size(); ++j ) {
            LOG( INFO ) << "Found neighbor id = " << nlist[j].first << " dist= " << nlist[j].second << " [" << d[nlist[j].first] << "]";
            EXPECT_TRUE( dist( d[i], d[nlist[j].first] ) < MIN_T );
        }
    }
}

TEST( VPTree, KNSearchNoSimilar )
{
    Dataset::Ptr dset = Dataset::create();
    ASSERT_TRUE( dset->load_csv( CURRENT_TDIR + "/csv/vptree01.csv" ) );

    TTree tree;
    tree.create( dset );

    const Dataset::DataContainer& d = dset->data();

    for ( size_t i = 0; i < d.size(); ++i ) {

        TTree::TNeighborsList nlist;

        LOG( INFO ) << "Searching for " << i << " [" << d[i] << "]";

        tree.search_by_k( d[i], 5u, nlist, true );

        EXPECT_EQ( nlist.size(), 5u );

        for ( size_t j = 0; j < nlist.size(); ++j ) {
            LOG( INFO ) << "Found neighbor id = "
                        << nlist[j].first << " dist= " << nlist[j].second << " [" << d[nlist[j].first] << "]";
            EXPECT_TRUE( dist( d[i], d[nlist[j].first] ) > 0 );
        }
    }
}
