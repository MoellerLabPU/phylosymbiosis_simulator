#!/usr/bin/env python3
"""Shared community evolution and tree helpers, retaining the uploaded model."""
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

@dataclass
class SimulationParams:
    tree_path: str
    n_species_pool: int = 1000
    root_richness: int = 100
    max_community_size: Optional[int] = None
    geom_p: float = 0.2
    brownian_sigma: float = 0.3
    abundance_model: str = "brownian"
    ou_alpha: float = 0.5
    extinction_rate: float = 0.08
    colonization_rate: float = 0.04
    colonist_weight_multiplier: float = 0.01
    min_positive_abundance: float = 1e-12


def normalize_profile(profile: np.ndarray) -> np.ndarray:
    total = profile.sum()
    if total <= 0:
        raise ValueError("Profile has zero total abundance.")
    return profile / total


def lower_triangle_values(mat: np.ndarray) -> np.ndarray:
    idx = np.tril_indices_from(mat, k=-1)
    return mat[idx]


def get_patristic_distance_matrix(tree) -> Tuple[List[str], np.ndarray]:
    tips = tree.get_terminals()
    names = [tip.name for tip in tips]
    n = len(tips)
    dist = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            d = tree.distance(tips[i], tips[j])
            dist[i, j] = d
            dist[j, i] = d

    return names, dist


def enforce_max_richness_weighted(
    profile: np.ndarray,
    max_k: Optional[int],
    rng: np.random.Generator,
) -> np.ndarray:
    if max_k is None:
        return profile

    present_idx = np.where(profile > 0)[0]
    richness = len(present_idx)

    if richness <= max_k:
        return profile

    abund = profile[present_idx].astype(float)
    total = abund.sum()

    if total <= 0:
        keep_idx = rng.choice(present_idx, size=max_k, replace=False)
    else:
        probs = abund / total
        keep_idx = rng.choice(present_idx, size=max_k, replace=False, p=probs)

    new_profile = np.zeros_like(profile)
    new_profile[keep_idx] = profile[keep_idx]
    return new_profile


def initialize_root_profile(params: SimulationParams, rng: np.random.Generator) -> np.ndarray:
    profile = np.zeros(params.n_species_pool, dtype=float)
    chosen = rng.choice(params.n_species_pool, size=params.root_richness, replace=False)
    abundances = rng.geometric(p=params.geom_p, size=params.root_richness).astype(float)
    profile[chosen] = abundances

    if params.max_community_size is not None:
        profile = enforce_max_richness_weighted(profile, params.max_community_size, rng)

    return normalize_profile(profile)


def transition_log_abundance(log_abund, branch_length, params, rng):
    """Transition log weights before the original turnover and normalization.

    OU: dY = alpha*(mu-Y)*dt + sigma*dW, mu = -log(root_richness).
    White noise: independent N(mu, sigma**2/(2*alpha)) draws on every
    positive-length branch. Taxon occupancy still follows inherited turnover.
    """
    if branch_length < 0:
        raise ValueError("Negative branch length detected.")
    model = params.abundance_model
    sigma = params.brownian_sigma
    alpha = params.ou_alpha
    if sigma < 0 or alpha < 0:
        raise ValueError("Sigma and OU alpha must be nonnegative.")
    if model == 'brownian' or (model == 'ou' and alpha == 0):
        return log_abund + rng.normal(0.0, sigma * math.sqrt(branch_length),
                                      size=log_abund.shape[0])
    if model not in ('ou', 'white_noise'):
        raise ValueError(f"Unknown abundance model: {model}")
    if branch_length == 0:
        return log_abund.copy()
    mu = -math.log(params.root_richness)
    if model == 'ou':
        decay = math.exp(-alpha * branch_length)
        mean = mu + decay * (log_abund - mu)
        sd = sigma * math.sqrt(-math.expm1(-2 * alpha * branch_length) / (2 * alpha))
    else:
        if alpha == 0:
            raise ValueError("White-noise variance requires positive OU alpha.")
        mean = mu
        sd = sigma / math.sqrt(2 * alpha)
    return mean + rng.normal(0.0, sd, size=log_abund.shape[0])


def evolve_profile_along_branch(
    parent_profile: np.ndarray,
    branch_length: float,
    params: SimulationParams,
    rng: np.random.Generator,
) -> np.ndarray:
    if branch_length is None:
        branch_length = 1.0
    if branch_length < 0:
        raise ValueError("Negative branch length detected.")

    profile = parent_profile.copy()
    present = profile > 0
    absent = ~present

    if present.any():
        log_abund = np.log(profile[present] + params.min_positive_abundance)
        log_abund = transition_log_abundance(log_abund, branch_length, params, rng)
        updated = np.exp(log_abund)

        p_ext = 1.0 - math.exp(-params.extinction_rate * branch_length)
        extinct = rng.random(updated.shape[0]) < p_ext
        updated[extinct] = 0.0
        profile[present] = updated

    absent_indices = np.where(absent)[0]
    if len(absent_indices) > 0:
        p_col = 1.0 - math.exp(-params.colonization_rate * branch_length)
        colonize = rng.random(len(absent_indices)) < p_col
        chosen_absent = absent_indices[colonize]
        if len(chosen_absent) > 0:
            new_abund = rng.geometric(
                p=params.geom_p, size=len(chosen_absent)
            ).astype(float) / 10.0 * params.colonist_weight_multiplier
            profile[chosen_absent] = new_abund

    profile[profile < params.min_positive_abundance] = 0.0

    if params.max_community_size is not None:
        profile = enforce_max_richness_weighted(
            profile,
            params.max_community_size,
            rng,
        )

    if profile.sum() <= 0:
        fallback = rng.choice(params.n_species_pool, size=1, replace=False)
        profile[fallback] = 1.0

    return normalize_profile(profile)


def simulate_tip_profiles(
    tree,
    params: SimulationParams,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    root_profile = initialize_root_profile(params, rng)
    tip_profiles: Dict[str, np.ndarray] = {}

    def recurse(clade, parent_profile: np.ndarray) -> None:
        branch_length = clade.branch_length if clade.branch_length is not None else 0.0
        current_profile = evolve_profile_along_branch(parent_profile, branch_length, params, rng)

        if clade.is_terminal():
            if clade.name is None:
                raise ValueError("All terminal tips must have names.")
            tip_profiles[clade.name] = current_profile
            return

        for child in clade.clades:
            recurse(child, current_profile)

    for child in tree.root.clades:
        recurse(child, root_profile)

    return tip_profiles


def canonical_split(subset: set, all_leaves: set) -> frozenset:
    other = all_leaves - subset
    subset_sorted = tuple(sorted(subset))
    other_sorted = tuple(sorted(other))

    if len(subset) < len(other):
        return frozenset(subset)
    if len(other) < len(subset):
        return frozenset(other)
    return frozenset(subset if subset_sorted <= other_sorted else other)


def get_host_splits(tree, tip_names: List[str]) -> set:
    all_leaves = set(tip_names)
    splits = set()

    def recurse(clade):
        if clade.is_terminal():
            return {clade.name}
        desc = set()
        for child in clade.clades:
            desc |= recurse(child)
        if 1 < len(desc) < len(all_leaves) - 1:
            splits.add(canonical_split(desc, all_leaves))
        return desc

    recurse(tree.root)
    return splits


def linkage_splits(Z: np.ndarray, tip_names: List[str]) -> set:
    n = len(tip_names)
    all_leaves = set(tip_names)
    clusters = {i: {tip_names[i]} for i in range(n)}
    splits = set()

    for row_idx, row in enumerate(Z):
        a = int(row[0])
        b = int(row[1])
        new_id = n + row_idx
        members = clusters[a] | clusters[b]
        if 1 < len(members) < len(all_leaves) - 1:
            splits.add(canonical_split(members, all_leaves))
        clusters[new_id] = members

    return splits


def normalized_robinson_foulds(splits1: set, splits2: set) -> float:
    denom = len(splits1) + len(splits2)
    if denom == 0:
        return 1.0
    rf = len(splits1 - splits2) + len(splits2 - splits1)
    return rf / denom
