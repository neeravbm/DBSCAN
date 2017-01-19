#include "gtest/gtest.h"
#include <iostream>
#include <fstream>

#ifdef _OPENMP
  #include <omp.h>
#endif

#include "dbscan_vp.h"
#include "dataset.h"

using namespace clustering;

namespace {
static const std::string CURRENT_TDIR( CURRENT_TEST_DIR );

static const size_t NUM_MEASURES = 20;
static const size_t NUM_FEATURES = 50;
static const size_t STEP_LEN = 10000;
static const size_t NUM_STEPS = 10;

//////////////////////////////////////////////////////////////////////////////
//
// process_mem_usage(double &, double &) - takes two doubles by reference,
// attempts to read the system-dependent data for a process' virtual memory
// size and resident set size, and return the results in KB.
//
// On failure, returns 0.0, 0.0

void process_mem_usage( double& vm_usage, double& resident_set )
{
    using std::ios_base;
    using std::ifstream;
    using std::string;

    vm_usage = 0.0;
    resident_set = 0.0;

    // 'file' stat seems to give the most reliable results
    //
    ifstream stat_stream( "/proc/self/stat", ios_base::in );

    // dummy vars for leading entries in stat that we don't care about
    //
    string pid, comm, state, ppid, pgrp, session, tty_nr;
    string tpgid, flags, minflt, cminflt, majflt, cmajflt;
    string utime, stime, cutime, cstime, priority, nice;
    string O, itrealvalue, starttime;

    // the two fields we want
    //
    unsigned long vsize;
    long rss;

    stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr
        >> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt
        >> utime >> stime >> cutime >> cstime >> priority >> nice
        >> O >> itrealvalue >> starttime >> vsize >> rss; // don't care about the rest

    stat_stream.close();

    long page_size_kb = sysconf( _SC_PAGE_SIZE ) / 1024; // in case x86-64 is configured to use 2MB pages
    vm_usage = vsize / 1024.0;
    resident_set = rss * page_size_kb;
}

static const size_t BIG_SIZE_START = 2000000;
static const size_t BIG_SIZE_STEP = 1000000;
static const size_t BIG_SIZE_NUM = 0;

static const size_t NUM_METRICS = 4;
}

TEST( DISABLED_DBSCAN_VP, Iris )
{
    std::vector< size_t > MEASURE_SIZES_VECTOR;
    for ( size_t i = STEP_LEN; i < STEP_LEN * NUM_STEPS; i += STEP_LEN ) {
        MEASURE_SIZES_VECTOR.push_back( i );
    }

    for ( size_t i = BIG_SIZE_START; i < ( BIG_SIZE_START + ( BIG_SIZE_STEP * BIG_SIZE_NUM ) ); i += BIG_SIZE_STEP ) {
        MEASURE_SIZES_VECTOR.push_back( i );
    }

    Eigen::MatrixXd means( MEASURE_SIZES_VECTOR.size(), NUM_METRICS );

    for ( size_t j = 0; j < MEASURE_SIZES_VECTOR.size(); ++j ) {
        const size_t vsz = MEASURE_SIZES_VECTOR[j];

        LOG( INFO ) << "Data matrix size " << NUM_FEATURES << "x" << vsz;
        Dataset::Ptr dset = Dataset::create();
        dset->gen_cluster_data( NUM_FEATURES, vsz );

        Eigen::MatrixXd measures( NUM_MEASURES, NUM_METRICS );

        for ( size_t i = 0; i < NUM_MEASURES; ++i ) {
            DBSCAN_VP::Ptr dbs = boost::make_shared< DBSCAN_VP >( dset );

            dbs->fit();
            dbs->predict( 0.01, 5 );

            measures( i, 0 ) = dbs->get_fit_time();
            measures( i, 1 ) = dbs->get_predict_time();

            double vm_usage = 0.0;
            double res_usage = 0.0;
            process_mem_usage( vm_usage, res_usage );

            measures( i, 2 ) = vm_usage;
            measures( i, 3 ) = res_usage;

            VLOG( 3 ) << "Measure " << ( i + 1 ) << " fit time = " << dbs->get_fit_time() << " seconds "
                      << "predict time = " << dbs->get_predict_time()
                      << " VM = " << vm_usage << "KB RES = " << res_usage << "KB";
        }

        Eigen::VectorXd mm = measures.colwise().mean();

        means( j, 0 ) = mm( 0 );
        means( j, 1 ) = mm( 1 );
        means( j, 2 ) = mm( 2 );
        means( j, 3 ) = mm( 3 );

        LOG( INFO ) << "Means = [" << mm( 0 ) << ", " << mm( 1 ) << ", " << mm( 2 ) << "," << mm( 3 ) << "]";
    }

    static const std::string stat_file_name = "stat.csv";

    std::ofstream stat_file;
    stat_file.open( stat_file_name.c_str(), std::ios::out | std::ios::trunc );

    stat_file << "data size,seconds,vm size,res size" << std::endl;

    for ( size_t j = 0; j < MEASURE_SIZES_VECTOR.size(); ++j ) {
        stat_file << MEASURE_SIZES_VECTOR[j] << ","
                  << means( j, 0 ) << ", " << means( j, 1 ) << "," << means( j, 2 ) << "," << means( j, 3 )
                  << std::endl;
    }

    stat_file.close();

    LOG( INFO ) << "Statistics written to " << stat_file_name;
}
