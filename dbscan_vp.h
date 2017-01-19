#ifndef DBSCAN_VP_H
#define DBSCAN_VP_H

#include "vptree.h"
#include <Eigen/Dense>

#ifndef _OPENMP
  #include <time.h>
#endif

namespace clustering {
class DBSCAN_VP : private boost::noncopyable {
private:
    static inline double dist( const Eigen::VectorXf& p1, const Eigen::VectorXf& p2 )
    {
        return ( p1 - p2 ).norm();
    }

    typedef VPTREE< Eigen::VectorXf, dist > TVpTree;

    const Dataset::Ptr m_dset;

public:
    typedef std::vector< int32_t > Labels;
    typedef boost::shared_ptr< DBSCAN_VP > Ptr;

    DBSCAN_VP( const Dataset::Ptr dset )
        : m_dset( dset )
        , m_fit_time( .0 )
        , m_predict_time( .0 )
    {
    }

    ~DBSCAN_VP()
    {
    }

    void fit()
    {
        const Dataset::DataContainer& d = m_dset->data();

#ifdef _OPENMP
        const double start = omp_get_wtime();
#else
        const double start = time(NULL);
#endif

        m_vp_tree = boost::make_shared< TVpTree >();
        m_vp_tree->create( m_dset );

        const size_t dlen = d.size();

        prepare_labels( dlen );

#ifdef _OPENMP
        m_fit_time = omp_get_wtime() - start;
#else
        m_fit_time = time(NULL) - start;
#endif
    }

    const std::vector< double > predict_eps( size_t k )
    {
        const Dataset::DataContainer& d = m_dset->data();

        std::vector< double > r;
        r.reserve( d.size() );

        TVpTree::TNeighborsList nlist;

        for ( size_t i = 0; i < d.size(); ++i ) {
            m_vp_tree->search_by_k( d[i], k, nlist, true );

            if ( nlist.size() >= k ) {
                r.push_back( nlist[0].second );
            }
        }

        std::sort( r.begin(), r.end() );

        return std::move( r );
    }

    uint32_t predict( double eps, size_t min_elems )
    {

        std::vector< uint32_t > candidates;
        std::vector< uint32_t > new_candidates;

        uint32_t cluster_id = 0;

        TVpTree::TNeighborsList index_neigh;
        TVpTree::TNeighborsList n_neigh;
        TVpTree::TNeighborsList c_neigh;

#ifdef _OPENMP
        const double start = omp_get_wtime();
#else
        const double start = time(NULL);
#endif

        const Dataset::DataContainer& d = m_dset->data();
        const size_t dlen = d.size();

        for ( uint32_t pid = 0; pid < dlen; ++pid ) {
            if ( pid % 10000 == 0 )
                VLOG( 1 ) << "progress: pid = " << pid << " " << ( float( pid ) / float( dlen ) ) * 100 << "%";

            if ( m_labels[pid] >= 0 )
                continue;

            find_neighbors( d, eps, pid, index_neigh );

            // std::cout << "Analyzing pid " << pid << " Neigh size " << index_neigh.size() << std::endl;

            if ( index_neigh.size() < min_elems )
                continue;

            m_labels[pid] = cluster_id;

            //VLOG( 1 ) << "pid = " << pid << " neig = " << index_neigh.size();

            candidates.clear();
            candidates.push_back( pid );

            while ( candidates.size() > 0 ) {
                // std::cout << "\tcandidates = " << candidates.size() << std::endl;

                new_candidates.clear();

                for ( const auto& c_pid : candidates ) {

                    // if ( m_labels[c_pid] >= 0 )
                    //     continue;

                    find_neighbors( d, eps, c_pid, c_neigh );

                    //VLOG( 1 ) << "c_pid = " << c_pid << " neig = " << c_neigh.size();

                    for ( const auto& nn : c_neigh ) {

                        if ( m_labels[nn.first] >= 0 )
                            continue;

                        m_labels[nn.first] = cluster_id;

                        find_neighbors( d, eps, nn.first, n_neigh );

                        // VLOG( 1 ) << "nn.first = " << nn.first << " neig = " << n_neigh.size();

                        if ( n_neigh.size() >= min_elems ) {
                            new_candidates.push_back( nn.first );
                        }
                    }
                }

                // std::cout << "\tnew candidates = " << new_candidates.size() << std::endl;

                candidates = new_candidates;
            }
            ++cluster_id;
        }

#ifdef _OPENMP
        m_predict_time = omp_get_wtime() - start;
#else
        m_predict_time = time(NULL) - start;
#endif

        return cluster_id;
    }

    void reset()
    {
        m_vp_tree.reset();
        m_labels.clear();
    }

    const Labels& get_labels() const
    {
        return m_labels;
    }

    const double get_fit_time() const
    {
        return m_fit_time;
    }

    const double get_predict_time() const
    {
        return m_predict_time;
    }

private:
    void find_neighbors( const Dataset::DataContainer& d,
                         double eps,
                         uint32_t pid,
                         TVpTree::TNeighborsList& neighbors )
    {
        neighbors.clear();
        m_vp_tree->search_by_dist( d[pid], eps, neighbors );
    }

    Labels m_labels;

    void prepare_labels( size_t s )
    {
        m_labels.resize( s );

        for ( auto& l : m_labels ) {
            l = -1;
        }
    }

    TVpTree::Ptr m_vp_tree;
    double m_fit_time;
    double m_predict_time;
};

// std::ostream& operator<<( std::ostream& o, DBSCAN& d );
}

#endif // DBSCAN_VP_H
