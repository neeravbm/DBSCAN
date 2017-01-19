#include <iostream>
#include <fstream>

#include <boost/shared_ptr.hpp>
#include <boost/program_options.hpp>

#include "glog/logging.h"

#include "dbscan_vp.h"

namespace po = boost::program_options;
using namespace clustering;

int main( int argc, char const* argv[] )
{
    std::string file_in;
    size_t k = 3;

    po::options_description option_desc( "kNN distance calculator" );

    option_desc.add_options()( "help", "Display help" )(
        "in,i", po::value< std::string >( &file_in )->required(), "Input file CSV format" )(
        "knn,k", po::value< size_t >( &k )->default_value( 3u ), "k-th neighbour" );

    po::variables_map options;

    po::store( po::command_line_parser( argc, argv ).options( option_desc ).run(), options );

    if ( options.empty() or options.count( "help" ) ) {
        std::cout << option_desc << std::endl;
        return 0;
    }

    po::notify( options );

    Dataset::Ptr dset = Dataset::create();
    LOG( INFO ) << "Loading dataset from " << file_in << " k = " << k;
    dset->load_csv( file_in );

    DBSCAN_VP::Ptr dbs = boost::make_shared< DBSCAN_VP >( dset );

    dbs->fit();

    LOG( INFO ) << "Fit time " << dbs->get_fit_time() << " seconds";

    const auto r = dbs->predict_eps( k );

    for ( size_t i = 0; i < r.size(); ++i ) {
        std::cout << i << "," << r[i] << std::endl;
    }

    return 0;
}