#!/usr/bin/env python3
"""One implementation of the five statistical methods and three reconstructions."""
import copy
import numpy as np
import pandas as pd
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.optimize import linear_sum_assignment
from scipy.special import xlogy
from scipy.stats import binomtest
import community_simulator as original

METHODS = ['raw', 'cophenetic', 'rf', 'mci', 'mc']

def split_matrix(splits, names):
    order = sorted(splits, key=lambda s: tuple(sorted(s)))
    return np.array([[name in s for name in names] for s in order], dtype=np.int16)


def entropy(sizes, n):
    p = np.asarray(sizes) / n
    return -(xlogy(p, p) + xlogy(1-p, 1-p)) / np.log(2)


def mi_lookup(a, b, n):
    # Four-cell contingency table of the two binary split memberships.
    c = np.arange(n+1)[None, None, :]
    a = a[:, None, None]; b = b[None, :, None]
    cells = [c, a-c, b-c, n-a-b+c]
    expectations = [a*b/n, a*(n-b)/n, (n-a)*b/n, (n-a)*(n-b)/n]
    valid = np.ones(np.broadcast_shapes(a.shape,b.shape,c.shape), dtype=bool)
    result = np.zeros(valid.shape)
    for cell, expected in zip(cells, expectations):
        valid &= cell >= 0
        safe = np.maximum(cell, 0)
        result += xlogy(safe/n, np.maximum(safe, 1e-300)/expected) / np.log(2)
    return np.where(valid, np.maximum(result, 0), np.nan)


def tree_stats(M, H, perms):
    n = M.shape[1]
    A, B = M.sum(1), H.sum(1)
    indices = np.vstack([np.arange(n), perms])
    # Permute host labels identically in both matrix and topology tests.
    HP = H[:,indices].transpose(1,0,2)
    overlap = (M.astype(np.float64)[None,:,:] @ HP.astype(np.float64).transpose(0,2,1)).astype(np.int16)
    equal = ((overlap == A[None,:,None]) & (A[None,:,None] == B[None,None,:]))
    equal |= ((overlap == 0) & (A[None,:,None]+B[None,None,:] == n))
    matches = equal.any(axis=2).sum(axis=1)
    rf = 1 - 2*matches/(len(A)+len(B))
    lookup = mi_lookup(A, B, n)
    scores = lookup[np.arange(len(A))[None,:,None], np.arange(len(B))[None,None,:], overlap]
    information = np.empty(len(indices))
    for k, score in enumerate(scores):
        row, col = linear_sum_assignment(score, maximize=True)
        information[k] = score[row,col].sum()
    information *= 2/(entropy(A,n).sum()+entropy(B,n).sum())
    return rf, information


def rooted_cluster_matrices(tree, Z, names):
    """Nontrivial rooted clades, including n-1 clades; never complement them."""
    n = len(names)
    host = [frozenset(t.name for t in c.get_terminals())
            for c in tree.find_clades() if not c.is_terminal() and c is not tree.root]
    members = {i: frozenset([name]) for i, name in enumerate(names)}
    microbial = []
    for k, row in enumerate(Z):
        cluster = members[int(row[0])] | members[int(row[1])]
        members[n+k] = cluster
        if len(cluster) < n:
            microbial.append(cluster)
    return split_matrix(microbial, names), split_matrix(host, names)


def normalized_matching_cluster(M, H, perms):
    """Minimum assignment of symmetric-difference costs on rooted clades.

    Normalize by sum of all clade cardinalities in both trees, an explicit
    upper bound on assignment cost (0 <= nMC <= 1). This is bound-scaled
    MC, not TreeCmp's random-tree-mean normalization or an exact diameter.
    Denominator is invariant under the shared host-label permutations.
    Unequal numbers of clades are padded with empty clades.
    """
    n = M.shape[1]
    count = max(len(M), len(H))
    M = np.pad(M, ((0, count-len(M)), (0, 0)))
    H = np.pad(H, ((0, count-len(H)), (0, 0)))
    A, B = M.sum(1), H.sum(1)
    denominator = int(A.sum()+B.sum())
    indices = np.vstack([np.arange(n), perms])
    HP = H[:, indices].transpose(1, 0, 2)
    overlap = (M.astype(np.float64)[None, :, :] @ HP.astype(np.float64).transpose(0, 2, 1)).astype(np.int16)
    costs = A[None, :, None]+B[None, None, :]-2*overlap
    values = np.zeros(len(indices))
    if denominator:
        for k, cost in enumerate(costs):
            row, col = linear_sum_assignment(cost)
            values[k] = cost[row, col].sum()/denominator
    return values


def midpoint_root_for_mc(tree):
    """Root an independent copy; signed original NJ estimates are untouched."""
    rooted = copy.deepcopy(tree)
    clipped = 0
    for clade in rooted.find_clades():
        if clade.branch_length is not None and clade.branch_length < 0:
            clade.branch_length = 0.
            clipped += 1
    zero_diameter = not any((c.branch_length or 0.) > 0 for c in rooted.find_clades()
                            if c is not rooted.root)
    if not zero_diameter:
        rooted.root_at_midpoint()
    rooted.rooted = True
    return rooted, clipped, zero_diameter


def rooted_clades(tree, names):
    return split_matrix([frozenset(t.name for t in c.get_terminals())
                         for c in tree.find_clades()
                         if c is not tree.root and not c.is_terminal()], names)


def corr_vectors(host_matrix, matrices, perms):
    n = host_matrix.shape[0]
    i,j = np.tril_indices(n,-1)
    indices = np.vstack([np.arange(n), perms])
    x = host_matrix[indices[:,i],indices[:,j]]
    x -= x.mean(1,keepdims=True)
    norm = np.linalg.norm(x,axis=1,keepdims=True)
    x = np.divide(x,norm,out=np.zeros_like(x),where=norm>0)
    out = []
    for mat in matrices:
        y = mat[i,j].copy(); y -= y.mean()
        scale = np.linalg.norm(y)
        out.append(x @ (y/scale if scale>0 else np.zeros_like(y)))
    return out


def reconstruct_microbiota(distance, names, method='average', return_tree=False):
    """Reconstruct the microbiota tree and distances in original sample order.

    NJ produces an unrooted tree; use tip-to-tip path lengths and unrooted
    nontrivial bipartitions. Retain signed NJ edge estimates, with counts in
    replicate tables. Complete linkage uses dendrogram cophenetic distances.
    """
    if method == 'nj':
        matrix = DistanceMatrix(list(names),
                                [distance[i, :i+1].tolist() for i in range(len(names))])
        microbial_tree = DistanceTreeConstructor().nj(matrix)
        labels, reconstructed = original.get_patristic_distance_matrix(microbial_tree)
        order = [labels.index(name) for name in names]
        reconstructed = reconstructed[np.ix_(order, order)]
        splits = original.get_host_splits(microbial_tree, names)
        edges = [float(c.branch_length) for c in microbial_tree.find_clades()
                 if c is not microbial_tree.root and c.branch_length is not None]
        diagnostic = dict(nj_negative_edges=sum(x < -1e-12 for x in edges),
                          nj_min_edge_length=min(edges),
                          tree_distance_kind='nj_path_length')
    else:
        Z = linkage(squareform(distance), method=method)
        reconstructed = squareform(cophenet(Z))
        splits = original.linkage_splits(Z, names)
        diagnostic = dict(nj_negative_edges=0,nj_min_edge_length=np.nan,
                          tree_distance_kind='cophenetic')
    if return_tree:
        return reconstructed, splits, diagnostic, microbial_tree if method == "nj" else Z
    return reconstructed, splits, diagnostic


def evaluate(tree, profiles, perms, hc='average', metric='BrayCurtis'):
    names, host = original.get_patristic_distance_matrix(tree)
    bray = squareform(pdist(profiles > 0, 'dice') if metric == 'BinaryDice'
                      else pdist(profiles, 'braycurtis'))
    coph, microbial_splits, reconstruction, microbial = reconstruct_microbiota(bray, names, hc, return_tree=True)
    raw, transformed = corr_vectors(host,[bray,coph],perms)
    M = split_matrix(microbial_splits,names)
    H = split_matrix(original.get_host_splits(tree,names),names)
    rf, mci = tree_stats(M,H,perms)
    if hc == 'nj':
        rooted, clipped, zero_diameter = midpoint_root_for_mc(microbial)
        CM, CH = rooted_clades(rooted, names), rooted_clades(tree, names)
    else:
        CM, CH = rooted_cluster_matrices(tree, microbial, names)
        clipped, zero_diameter = 0, False
    mc = normalized_matching_cluster(CM, CH, perms)
    stats = [raw, transformed, -rf, mci, -mc]
    rows=[]
    for name, st in zip(METHODS,stats):
        p = (1+np.sum(st[1:] >= st[0]-1e-12))/(len(perms)+1)
        if name in ['raw','cophenetic'] and (not np.isfinite(st[0]) or np.ptp(bray)==0):
            p=1.
        two_sided = (float((1+np.sum(np.abs(st[1:])>=abs(st[0])-1e-12))/(len(perms)+1))
                     if name in ['raw','cophenetic'] else np.nan)
        rows.append(dict(method=name, statistic=float(-st[0] if name in ['rf', 'mc'] else st[0]),
                         p=float(p), reject=bool(p<.05), p_two_sided=two_sided))
    if hc == 'nj':
        rows[-1].update(mc_rooting='midpoint', mc_rooting_edges_clipped=clipped,
                        mc_rooting_zero_diameter=zero_diameter)
    distortion = np.sqrt(np.mean((bray[np.tril_indices(len(names),-1)]-coph[np.tril_indices(len(names),-1)])**2))
    for row in rows:
        row.update(raw_r=float(raw[0]),coph_r=float(transformed[0]),distortion_rmse=float(distortion),
                   mean_richness=float((profiles>0).sum(1).mean()))
        if hc != "average": row.update(reconstruction)
    return rows


def communities(tree,p,rng):
    names,_=original.get_patristic_distance_matrix(tree)
    profiles=original.simulate_tip_profiles(tree,p,rng)
    return np.vstack([profiles[name] for name in names])


def independent_profiles(tree,p,rng):
    # Independently initiated/evolved communities; unlabeled depth draws ensure exchangeability.
    depths=np.array([tree.distance(tree.root,t) for t in tree.get_terminals()])
    rows=[]
    for _ in depths:
        root=original.initialize_root_profile(p,rng)
        rows.append(original.evolve_profile_along_branch(root,float(rng.choice(depths)),p,rng))
    return np.vstack(rows)


def nni(tree,k,rng):
    tree=copy.deepcopy(tree)
    for _ in range(k):
        candidates=[(p,c) for p in tree.find_clades() if len(p.clades)==2
                    for c in p.clades if len(c.clades)==2]
        p,c=candidates[int(rng.integers(len(candidates)))]
        sib=next(x for x in p.clades if x is not c)
        child=c.clades[int(rng.integers(2))]
        p.clades[p.clades.index(sib)]=child
        c.clades[c.clades.index(child)]=sib
    return tree


def restore_order(tree,names):
    # evaluate() uses tree traversal order; reorder community rows to match it explicitly.
    return [names.index(t.name) for t in tree.get_terminals()]


def summarize(df):
    keys=['suite','n_hosts','richness','noise','sigma','rate','uncertainty_kind','uncertainty','hc','null_model','method']
    rows=[]
    for key,g in df.groupby(keys,dropna=False):
        n=len(g);k=int(g.reject.sum());ci=binomtest(k,n).proportion_ci(method='wilson')
        row=dict(zip(keys,key)); row.update(n=n,significant=k,proportion=k/n,ci_low=ci.low,ci_high=ci.high,
                     mean_statistic=float(g.statistic.mean()),mean_raw_r=float(g.raw_r.mean()),mean_coph_r=float(g.coph_r.mean()),
                     mean_distortion_rmse=float(g.distortion_rmse.mean()))
        rows.append(row)
    return pd.DataFrame(rows)
