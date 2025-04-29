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

using faiss::IndexHNSWFlat;
using faiss::MetricType;

DEFINE_DESTRUCTOR(IndexHNSW)
DEFINE_INDEX_DOWNCAST(IndexHNSW)

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

int faiss_IndexHNSWFlat_entry_point(
        FaissIndexHNSWFlat* p_index,
        idx_t* entry_point,
        int* max_level) {
    try {
        const auto& hnsw = reinterpret_cast<IndexHNSWFlat*>(p_index)->hnsw;
        *entry_point = hnsw.entry_point;
        *max_level = hnsw.max_level;
    }
    CATCH_AND_HANDLE
}

int faiss_IndexHNSWFlat_levels(
        FaissIndexHNSWFlat* p_index,
        const int** levels) {
    try {
        const auto& hnsw = reinterpret_cast<IndexHNSWFlat*>(p_index)->hnsw;
        *levels = hnsw.levels.data();
    }
    CATCH_AND_HANDLE
}

int faiss_IndexHNSWFlat_neighbors(
        FaissIndexHNSWFlat* p_index,
        idx_t no,
        int level_no,
        const FaissHNSWNeighborIdx** neighbors,
        size_t* neighbor_count) {
    try {
        size_t begin, end;
        const auto& hnsw = reinterpret_cast<IndexHNSWFlat*>(p_index)->hnsw;
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
    CATCH_AND_HANDLE
}