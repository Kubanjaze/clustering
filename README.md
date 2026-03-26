# clustering — Phase 49

k-means and Butina clustering of the 45-compound CETP inhibitor library using ECFP4 fingerprints. PCA projection shows cluster composition by scaffold family.

## Usage

```bash
PYTHONUTF8=1 python main.py --input data/compounds.csv
```

## Outputs

- `output/clustering_results.csv` — cluster assignments and dominant families
- `output/clustering_plot.png` — PCA scatter: scaffold family / k-means / Butina
