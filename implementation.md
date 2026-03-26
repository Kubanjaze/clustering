# Phase 49 — Clustering (k-means + Butina on Fingerprints)

**Version:** 1.1 | **Tier:** Standard | **Date:** 2026-03-26

## Goal
Cluster the 45-compound CETP library using two algorithms:
1. k-means on ECFP4 fingerprint bit vectors (k=5)
2. Butina clustering on Tanimoto distances (cutoff=0.4)

Compare cluster composition (scaffold families) and evaluate cluster quality (silhouette score).

CLI: `python main.py --input data/compounds.csv`

Outputs: clustering_results.csv, clustering_plot.png

## Logic
- Compute ECFP4 fingerprints (radius=2, nBits=2048, useChirality=True)
- k-means: k=5 clusters on fingerprint bit vectors; silhouette score
- Butina: RDKit BulkTanimotoSimilarity → distance matrix → ClusterData(cutoff=0.4)
- Compare: cluster purity by scaffold family, cluster sizes
- Plot: PCA 2D projection colored by scaffold family + k-means + Butina (3-panel)

## Results

### k-means (k=5, silhouette=0.287)

| Cluster | Size | Dominant Family |
|---|---|---|
| C0 | 7 | ind |
| C1 | 19 | benz |
| C2 | 6 | pyr |
| C3 | 7 | quin |
| C4 | 6 | bzim |

### Butina (cutoff=0.4, 31 clusters)

| Clusters | Description |
|---|---|
| C0 (n=9) | benz — largest real cluster |
| C1 (n=7) | naph |
| C2–C30 (n=1 each) | 29 singletons across all families |

## Key Insights
- k-means (sil=0.287) recovers scaffold families cleanly: 5 clusters ≈ 5 non-benz families, with benz dominating C1
- Butina with cutoff=0.4 over-fragments: 31 clusters for 45 compounds (64% singletons)
  - This confirms the library is chemically diverse — most compounds sit >0.6 Tanimoto distance from each other
  - Only benz (C0, n=9) and naph (C1, n=7) form cohesive clusters, matching their larger family sizes
- k-means is more appropriate for downstream stratification; Butina better for diversity selection
- PCA variance explained: ~15-20% in 2D (high-dim fingerprint data expected to compress poorly)

## Deviations from Plan
- None; plan implemented as specified.
