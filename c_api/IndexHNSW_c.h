/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c -*-

#ifndef FAISS_INDEX_HNSW_C_H
#define FAISS_INDEX_HNSW_C_H

#include "Clustering_c.h"
#include "Index_c.h"
#include "faiss_c.h"
#include "impl/AuxIndexStructures_c.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef int32_t FaissHNSWNeighborIdx;

/** The HNSW index is a normal random-access index with a HNSW
 * link structure built on top */
FAISS_DECLARE_CLASS_INHERITED(IndexHNSW, Index)
FAISS_DECLARE_DESTRUCTOR(IndexHNSW)
FAISS_DECLARE_INDEX_DOWNCAST(IndexHNSW)

/** Flat index topped with with a HNSW structure to access elements
 *  more efficiently.
 */
FAISS_DECLARE_CLASS_INHERITED(IndexHNSWFlat, Index)
FAISS_DECLARE_DESTRUCTOR(IndexHNSWFlat)
FAISS_DECLARE_INDEX_DOWNCAST(IndexHNSWFlat)

int faiss_IndexHNSWFlat_new(FaissIndexHNSWFlat** p_index);

int faiss_IndexHNSWFlat_new_with(FaissIndexHNSWFlat** p_index, int d, int m);

int faiss_IndexHNSWFlat_new_with_metric(
        FaissIndexHNSWFlat** p_index,
        int d,
        int m,
        FaissMetricType metric);

int faiss_IndexHNSWFlat_entry_point(
        FaissIndexHNSWFlat* p_index,
        idx_t* entry_point,
        int* max_level);

int faiss_IndexHNSWFlat_levels(FaissIndexHNSWFlat* p_index, const int** levels);

int faiss_IndexHNSWFlat_neighbours(
        FaissIndexHNSWFlat* p_index,
        idx_t no,
        int level_no,
        const FaissHNSWNeighborIdx** neighbors,
        size_t* neighbor_count);

#ifdef __cplusplus
}
#endif

#endif