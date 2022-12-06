#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

void vertex_set_clear(vertex_set* list) {
  list->count = 0;
}

void vertex_set_init(vertex_set* list, int count) {
  list->max_vertices = count;
  list->vertices = (int*)malloc(sizeof(int) * list->max_vertices);
  vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(Graph g,
                   vertex_set* frontier,
                   vertex_set* new_frontier,
                   int* distances,
                   vertex_set* local_sets) {
  #pragma omp parallel
  {
    // Each thread builds a local set.
    const int tid = omp_get_thread_num();
    vertex_set* local = local_sets == nullptr ? nullptr : &(local_sets[tid]);
    if (local != nullptr) { vertex_set_clear(local); }

    #pragma omp for schedule(static)
    for (int i = 0; i < frontier->count; ++i) {
      int node = frontier->vertices[i];

      int start_edge = g->outgoing_starts[node];
      int end_edge = (node == g->num_nodes - 1)
        ? g->num_edges
        : g->outgoing_starts[node + 1];

      for (int neighbor = start_edge; neighbor < end_edge; ++neighbor) {
        int outgoing = g->outgoing_edges[neighbor];

        if (distances[outgoing] == NOT_VISITED_MARKER &&
            __sync_bool_compare_and_swap(&distances[outgoing],
                                         NOT_VISITED_MARKER,
                                         distances[node] + 1)) {
          if (local != nullptr) {
            local->vertices[local->count++] = outgoing;
          } else {
            int index = -1;
            #pragma omp atomic capture
            index = new_frontier->count++;
            new_frontier->vertices[index] = outgoing;
          }
        }
      }
    }
    if (local != nullptr && local->count > 0) {
      // Synchronously reserve space for copy.
      const int offset = __sync_fetch_and_add(&new_frontier->count, local->count);
      memcpy(new_frontier->vertices + offset, local->vertices, local->count * sizeof(int));
    }
  }
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution* sol) {
  vertex_set list1;
  vertex_set list2;
  vertex_set_init(&list1, graph->num_nodes);
  vertex_set_init(&list2, graph->num_nodes);

  vertex_set* frontier = &list1;
  vertex_set* new_frontier = &list2;

  vertex_set* local_sets = nullptr;
  const bool use_local_set = (num_edges(graph) / num_nodes(graph)) >= 5;
  #pragma omp single
  {
    if (use_local_set) {
      local_sets = new vertex_set[omp_get_max_threads()];
      #pragma omp parallel for
      for (int i = 0; i < omp_get_max_threads(); ++i) {
        vertex_set_init(&(local_sets[i]), graph->num_nodes);
      }
    }
  }

  // initialize all nodes to NOT_VISITED
  for (int i=0; i<graph->num_nodes; ++i)
    sol->distances[i] = NOT_VISITED_MARKER;

  // setup frontier with the root node
  frontier->vertices[frontier->count++] = ROOT_NODE_ID;
  sol->distances[ROOT_NODE_ID] = 0;

  for (int d = 0; frontier->count != 0; ++d) {

#ifdef VERBOSE
    double start_time = CycleTimer::currentSeconds();
#endif

    vertex_set_clear(new_frontier);

    top_down_step(graph, frontier, new_frontier, sol->distances, local_sets);

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

    // swap pointers
    vertex_set* tmp = frontier;
    frontier = new_frontier;
    new_frontier = tmp;
  }

  if (local_sets != nullptr) { delete[] local_sets; }
}

// Take one step of "bottom-up" BFS.
void bottom_up_step(Graph g,
                    vertex_set* frontier,
                    vertex_set* new_frontier,
                    int* distances,
                    int depth,
                    vertex_set* local_sets) {
  #pragma omp parallel
  {
    // Each thread builds a local set.
    const int tid = omp_get_thread_num();
    vertex_set* local = local_sets == nullptr ? nullptr : &(local_sets[tid]);
    if (local != nullptr) { vertex_set_clear(local); }

    #pragma omp for schedule(dynamic, 1000)
    for (int node = 0; node < g->num_nodes; ++node) {
      if (distances[node] != NOT_VISITED_MARKER) { continue; }

      int start_edge = g->incoming_starts[node];
      int end_edge = (node == g->num_nodes - 1)
        ? g->num_edges
        : g->incoming_starts[node + 1];

      for (int neighbor = start_edge; neighbor < end_edge; ++neighbor) {
        int incoming = g->incoming_edges[neighbor];

        if (distances[incoming] == depth) {
          distances[node] = depth + 1;
          local->vertices[local->count++] = node;
          break;
        }
      }
    }

    if (local != nullptr && local->count > 0) {
      // Synchronously reserve space for copy.
      const int offset = __sync_fetch_and_add(&new_frontier->count, local->count);
      memcpy(new_frontier->vertices + offset, local->vertices, local->count * sizeof(int));
    }
  }
}

void bfs_bottom_up(Graph graph, solution* sol)
{
  // CS149 students:
  //
  // You will need to implement the "bottom up" BFS here as
  // described in the handout.
  //
  // As a result of your code's execution, sol.distances should be
  // correctly populated for all nodes in the graph.
  //
  // As was done in the top-down case, you may wish to organize your
  // code by creating subroutine bottom_up_step() that is called in
  // each step of the BFS process.
  vertex_set list1;
  vertex_set list2;
  vertex_set_init(&list1, graph->num_nodes);
  vertex_set_init(&list2, graph->num_nodes);

  vertex_set* frontier = &list1;
  vertex_set* new_frontier = &list2;

  vertex_set* local_sets = nullptr;
  #pragma omp single
  {
    local_sets = new vertex_set[omp_get_max_threads()];
    #pragma omp parallel for
    for (int i = 0; i < omp_get_max_threads(); ++i) {
      vertex_set_init(&(local_sets[i]), graph->num_nodes);
    }
  }

  // initialize all nodes to NOT_VISITED
  for (int i=0; i<graph->num_nodes; ++i)
    sol->distances[i] = NOT_VISITED_MARKER;

  // setup frontier with the root node
  frontier->vertices[frontier->count++] = ROOT_NODE_ID;
  sol->distances[ROOT_NODE_ID] = 0;

  for (int d = 0; frontier->count != 0; ++d) {
#ifdef VERBOSE
    double start_time = CycleTimer::currentSeconds();
#endif

    vertex_set_clear(new_frontier);

    bottom_up_step(graph, frontier, new_frontier, sol->distances, d,
                   local_sets);

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

    // swap pointers
    vertex_set* tmp = frontier;
    frontier = new_frontier;
    new_frontier = tmp;
  }

  if (local_sets != nullptr) { delete[] local_sets; }
}

void bfs_hybrid(Graph graph, solution* sol)
{
  // CS149 students:
  //
  // You will need to implement the "hybrid" BFS here as
  // described in the handout.
  vertex_set list1;
  vertex_set list2;
  vertex_set_init(&list1, graph->num_nodes);
  vertex_set_init(&list2, graph->num_nodes);

  vertex_set* frontier = &list1;
  vertex_set* new_frontier = &list2;

  vertex_set* local_sets = nullptr;
  const bool use_local_set = (num_edges(graph) / num_nodes(graph)) >= 5;
  #pragma omp single
  {
    local_sets = new vertex_set[omp_get_max_threads()];
    #pragma omp parallel for
    for (int i = 0; i < omp_get_max_threads(); ++i) {
      vertex_set_init(&(local_sets[i]), graph->num_nodes);
    }
  }
  vertex_set* top_down_locals = use_local_set ? local_sets : nullptr;

  // initialize all nodes to NOT_VISITED
  for (int i=0; i<graph->num_nodes; ++i)
    sol->distances[i] = NOT_VISITED_MARKER;

  // setup frontier with the root node
  frontier->vertices[frontier->count++] = ROOT_NODE_ID;
  sol->distances[ROOT_NODE_ID] = 0;

  for (int d = 0; frontier->count != 0; ++d) {
#ifdef VERBOSE
    double start_time = CycleTimer::currentSeconds();
#endif

    vertex_set_clear(new_frontier);

    if (frontier->count > 100000) {
      bottom_up_step(graph, frontier, new_frontier, sol->distances, d, local_sets);
    } else {
      top_down_step(graph, frontier, new_frontier, sol->distances, top_down_locals);
    }

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

    // swap pointers
    vertex_set* tmp = frontier;
    frontier = new_frontier;
    new_frontier = tmp;
  }

  if (local_sets != nullptr) { delete[] local_sets; }
}
