#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
Matlab version: https://github.com/zhunzhong07/person-re-ranking
"""

import os, sys
import time
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
import gc
import faiss

import torch
import torch.nn.functional as F

from .faiss_utils import search_index_pytorch, search_raw_array_pytorch, \
                            index_init_gpu, index_init_cpu


def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i,:k1+1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
    fi = np.where(backward_k_neigh_index==i)[0]
    return forward_k_neigh_index[fi]


def compute_jaccard_distance(target_features, k1=20, k2=6, print_flag=True, search_option=0, use_float16=False, sparse=False):
    end = time.time()
    if print_flag:
        print('Computing jaccard distance...')

    ngpus = faiss.get_num_gpus()
    N = target_features.size(0)
    mat_type = np.float16 if use_float16 else np.float32
    search_option = 2

    if (search_option==0):
        # GPU + PyTorch CUDA Tensors (1)
        res = faiss.StandardGpuResources()
        res.setDefaultNullStreamAllDevices()
        _, initial_rank = search_raw_array_pytorch(res, target_features, target_features, k1)
        initial_rank = initial_rank.cpu().numpy()
    elif (search_option==1):
        # GPU + PyTorch CUDA Tensors (2)
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, target_features.size(-1))
        index.add(target_features.cpu().numpy())
        _, initial_rank = search_index_pytorch(index, target_features, k1)
        res.syncDefaultStreamCurrentDevice()
        initial_rank = initial_rank.cpu().numpy()
    elif (search_option==2):
        # GPU
        index = index_init_gpu(ngpus, target_features.size(-1))
        index.add(target_features.cpu().numpy())
        _, initial_rank = index.search(target_features.cpu().numpy(), k1)
    else:
        # CPU
        index = index_init_cpu(target_features.size(-1))
        index.add(target_features.cpu().numpy())
        _, initial_rank = index.search(target_features.cpu().numpy(), k1)


    nn_k1 = []
    nn_k1_half = []
    for i in range(N):
        nn_k1.append(k_reciprocal_neigh(initial_rank, i, k1))
        nn_k1_half.append(k_reciprocal_neigh(initial_rank, i, int(np.around(k1/2))))

    if sparse:
        # scipy sparse has poor float16 support; the sparse path always uses float32
        V_rows, V_cols, V_vals = [], [], []
    else:
        V = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        k_reciprocal_index = nn_k1[i]
        k_reciprocal_expansion_index = k_reciprocal_index
        for candidate in k_reciprocal_index:
            candidate_k_reciprocal_index = nn_k1_half[candidate]
            if (len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index)) > 2/3*len(candidate_k_reciprocal_index)):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)  ## element-wise unique
        dist = 2-2*torch.mm(target_features[i].unsqueeze(0).contiguous(), target_features[k_reciprocal_expansion_index].t())
        if sparse:
            V_rows.append(np.full(len(k_reciprocal_expansion_index), i, dtype=np.int64))
            V_cols.append(k_reciprocal_expansion_index.astype(np.int64))
            V_vals.append(F.softmax(-dist, dim=1).view(-1).cpu().numpy().astype(np.float32))
        elif use_float16:
            V[i,k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy().astype(mat_type)
        else:
            V[i,k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy()

    del nn_k1, nn_k1_half

    if sparse:
        V = csr_matrix((np.concatenate(V_vals), (np.concatenate(V_rows), np.concatenate(V_cols))),
                       shape=(N, N), dtype=np.float32)
        del V_rows, V_cols, V_vals

    if k2 != 1:
        if sparse:
            # densify only the k2 selected rows per iteration so np.mean sees the
            # exact same operands as the dense path (bitwise-identical results);
            # a csr matmul with 1/k2 weights changes summation order and its ~1e-6
            # drift is enough to flip DBSCAN decisions on eps-boundary pairs
            qe_rows, qe_cols, qe_vals = [], [], []
            for i in range(N):
                mean_row = np.asarray(V[initial_rank[i, :k2], :].todense(), dtype=np.float32).mean(axis=0)
                nz = np.where(mean_row != 0)[0]
                qe_rows.append(np.full(len(nz), i, dtype=np.int64))
                qe_cols.append(nz)
                qe_vals.append(mean_row[nz])
            V = csr_matrix((np.concatenate(qe_vals), (np.concatenate(qe_rows), np.concatenate(qe_cols))),
                           shape=(N, N), dtype=np.float32)
            del qe_rows, qe_cols, qe_vals
        else:
            V_qe = np.zeros_like(V, dtype=mat_type)
            for i in range(N):
                V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:], axis=0)
            V = V_qe
            del V_qe

    del initial_rank

    if sparse:
        # entries never stored are exactly those with temp_min == 0, i.e. jaccard
        # distance 1; sklearn's sparse precomputed DBSCAN treats missing entries
        # as beyond eps, so dropping them cannot change the clustering result.
        Vc = V.tocsc()
        rows_out, cols_out, vals_out = [], [], []
        acc = np.zeros(N, dtype=np.float32)
        for i in range(N):
            indNonZero = V.indices[V.indptr[i]:V.indptr[i+1]]
            valNonZero = V.data[V.indptr[i]:V.indptr[i+1]]
            touched = []
            for v_ik, col in zip(valNonZero, indNonZero):
                rows_j = Vc.indices[Vc.indptr[col]:Vc.indptr[col+1]]
                vals_j = Vc.data[Vc.indptr[col]:Vc.indptr[col+1]]
                acc[rows_j] += np.minimum(v_ik, vals_j)
                touched.append(rows_j)
            if len(touched) == 0:
                continue
            touched = np.unique(np.concatenate(touched))
            temp_min = acc[touched]
            dist_row = 1-temp_min/(2-temp_min)
            np.maximum(dist_row, 0, out=dist_row)  # same clip as the dense pos_bool step
            rows_out.append(np.full(len(touched), i, dtype=np.int64))
            cols_out.append(touched)
            vals_out.append(dist_row)
            acc[touched] = 0
        del Vc, V, acc
        # explicit zeros (e.g. the diagonal) must stay stored: for sparse
        # precomputed metrics, missing means "beyond eps", stored 0 means distance 0
        jaccard_dist = csr_matrix((np.concatenate(vals_out), (np.concatenate(rows_out), np.concatenate(cols_out))),
                                  shape=(N, N), dtype=np.float32)
        del rows_out, cols_out, vals_out
        from sklearn.neighbors import sort_graph_by_row_values
        sort_graph_by_row_values(jaccard_dist, copy=False, warn_when_not_sorted=False)
        if print_flag:
            print("Jaccard distance computing time cost: {}".format(time.time()-end))
        return jaccard_dist

    invIndex = []
    for i in range(N):
        invIndex.append(np.where(V[:,i] != 0)[0])  #len(invIndex)=all_num

    jaccard_dist = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        temp_min = np.zeros((1, N), dtype=mat_type)
        # temp_max = np.zeros((1,N), dtype=mat_type)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]]+np.minimum(V[i, indNonZero[j]], V[indImages[j], indNonZero[j]])
            # temp_max[0,indImages[j]] = temp_max[0,indImages[j]]+np.maximum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])

        jaccard_dist[i] = 1-temp_min/(2-temp_min)
        # jaccard_dist[i] = 1-temp_min/(temp_max+1e-6)

    del invIndex, V

    pos_bool = (jaccard_dist < 0)
    jaccard_dist[pos_bool] = 0.0
    if print_flag:
        print("Jaccard distance computing time cost: {}".format(time.time()-end))

    return jaccard_dist
