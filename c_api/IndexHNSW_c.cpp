/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include "IndexHNSW_c.h"
#include <faiss/IndexHNSW.h>
#include "Clustering_c.h"
#include "Index_c.h"
#include "macros_impl.h"

using faiss::IndexHNSW;
using faiss::IndexHNSWFlat;
using faiss::MetricType;

DEFINE_DESTRUCTOR(IndexHNSW)
DEFINE_INDEX_DOWNCAST(IndexHNSW)

void faiss_IndexHNSW_set_ef_construction(
        FaissIndexHNSW* p_index,
        int ef_construction) {
    auto& hnsw = reinterpret_cast<IndexHNSW*>(p_index)->hnsw;
    hnsw.efConstruction = ef_construction;
}

void faiss_IndexHNSW_set_ef_search(FaissIndexHNSW* p_index, int ef_search) {
    auto& hnsw = reinterpret_cast<IndexHNSW*>(p_index)->hnsw;
    hnsw.efSearch = ef_search;
}

void faiss_IndexHNSW_entry_point(
        const FaissIndexHNSW* p_index,
        idx_t* entry_point,
        int* max_level) {
    const auto& hnsw = reinterpret_cast<const IndexHNSW*>(p_index)->hnsw;
    *entry_point = hnsw.entry_point;
    *max_level = hnsw.max_level;
}

void faiss_IndexHNSW_levels(const FaissIndexHNSW* p_index, const int** levels) {
    const auto& hnsw = reinterpret_cast<const IndexHNSW*>(p_index)->hnsw;
    *levels = hnsw.levels.data();
}

void faiss_IndexHNSW_neighbors(
        const FaissIndexHNSW* p_index,
        idx_t no,
        int level_no,
        const FaissHNSWNeighborIdx** neighbors,
        size_t* neighbor_count) {
    size_t begin, end;
    const auto& hnsw = reinterpret_cast<const IndexHNSW*>(p_index)->hnsw;
    hnsw.neighbor_range(no, level_no, &begin, &end);
    *neighbors = &hnsw.neighbors.data()[begin];
    while (end > begin) {
        if (end - 1 >= begin && hnsw.neighbors[end - 1] >= 0) {
            break;
        }
        end--;
    }
    *neighbor_count = end - begin;
}

DEFINE_DESTRUCTOR(IndexHNSWFlat)
DEFINE_INDEX_DOWNCAST(IndexHNSWFlat)

int faiss_IndexHNSWFlat_new(FaissIndexHNSWFlat** p_index) {
    try {
        *p_index = reinterpret_cast<FaissIndexHNSWFlat*>(new IndexHNSWFlat());
    }
    CATCH_AND_HANDLE
}

int faiss_IndexHNSWFlat_new_with(FaissIndexHNSWFlat** p_index, int d, int m) {
    try {
        *p_index =
                reinterpret_cast<FaissIndexHNSWFlat*>(new IndexHNSWFlat(d, m));
    }
    CATCH_AND_HANDLE
}

int faiss_IndexHNSWFlat_new_with_metric(
        FaissIndexHNSWFlat** p_index,
        int d,
        int m,
        FaissMetricType metric) {
    try {
        *p_index = reinterpret_cast<FaissIndexHNSWFlat*>(
                new IndexHNSWFlat(d, m, static_cast<MetricType>(metric)));
    }
    CATCH_AND_HANDLE
}
