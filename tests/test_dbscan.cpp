#include <omp.h>
#include "dbscan.h"

using namespace clustering;

int main()
{
    std::cout << "Generating data.." << std::endl;

    double start = omp_get_wtime();

    DBSCAN::ClusterData cl_d = DBSCAN::gen_cluster_data( 225, 100000 );

    double end = omp_get_wtime();

    std::cout << "data gen took: " << end - start << " seconds" << std::endl;

    DBSCAN dbs( 0.1, 5, 1 );

    start = omp_get_wtime();

    dbs.fit( cl_d );

    end = omp_get_wtime();

    std::cout << "clustering took: " << end - start << " seconds" << std::endl;

    return 0;
}
