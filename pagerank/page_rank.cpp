#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"


// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double* solution, double damping, double convergence)
{
  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs

  int numNodes = num_nodes(g);
  double equal_prob = 1.0 / numNodes;
  for (int i = 0; i < numNodes; ++i) {
    solution[i] = equal_prob;
  }

  /*
    CS149 students: Implement the page rank algorithm here.  You
    are expected to parallelize the algorithm using openMP.  Your
    solution may need to allocate (and free) temporary arrays.

    Basic page rank pseudocode is provided below to get you started:

    // initialization: see example code above
    score_old[vi] = 1/numNodes;

    while (!converged) {

    // compute score_new[vi] for all nodes vi:
    score_new[vi] = sum over all nodes vj reachable from incoming edges
    { score_old[vj] / number of edges leaving vj  }
    score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

    score_new[vi] += sum over all nodes v in graph with no outgoing edges
    { damping * score_old[v] / numNodes }

    // compute how much per-node scores have changed
    // quit once algorithm has converged

    global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
    converged = (global_diff < convergence)
    }
  */
  double* score_old = solution;
  double* score_new = new double[numNodes];
  double* score_diff = new double[numNodes];
  const double damping_const = (1.0 - damping) / numNodes;

  bool converged = false;
  while (!converged) {
    double sum_score_no_outedges = 0.0;
    #pragma omp parallel for
    for (int i = 0; i < numNodes; ++i) {
      double score_i = 0.0;
      const Vertex* in_start_i = incoming_begin(g, i);
      const Vertex* in_end_i = incoming_end(g, i);
      for (const Vertex* v = in_start_i; v != in_end_i; ++v) {
        score_i += score_old[*v] / outgoing_size(g, *v);
      }
      score_new[i] = (damping * score_i) + damping_const;

      if (outgoing_size(g, i) == 0) {
        #pragma omp atomic
        sum_score_no_outedges += score_old[i];
      }
    }

    sum_score_no_outedges = damping * sum_score_no_outedges / numNodes;

    #pragma omp parallel for
    for (int i = 0; i < numNodes; ++i) {
      score_new[i] += sum_score_no_outedges;
      score_diff[i] = abs(score_new[i] - score_old[i]);  // Local diff.
      score_old[i] = score_new[i];  // Copy the new score.
    }

    // Use reduction as we need to compute a local score for each node
    // and then accumulate.
    double global_diff = 0.0;
    #pragma omp parallel for reduction(+:global_diff)
    for (int i = 0; i < numNodes; ++i) {
      global_diff += score_diff[i];
    }

    converged = (global_diff < convergence);
  }

  delete[] score_new;
  delete[] score_diff;
}
