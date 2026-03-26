import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import argparse, os, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from rdkit.ML.Cluster import Butina
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
RDLogger.DisableLog("rdApp.*")

FAMILY_COLORS = {"benz": "#4C72B0", "naph": "#DD8452", "ind": "#55A868",
                 "quin": "#C44E52", "pyr": "#8172B2", "bzim": "#937860", "other": "#808080"}

CLUSTER_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
                  "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62"]


def load_compounds(path):
    df = pd.read_csv(path)
    records, n_bad = [], 0
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(str(row["smiles"]))
        if mol is None:
            n_bad += 1
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, useChirality=True)
        fam = str(row["compound_name"]).split("_")[0]
        records.append({
            "compound_name": str(row["compound_name"]),
            "family": fam if fam in FAMILY_COLORS else "other",
            "pic50": float(row["pic50"]) if "pic50" in row else np.nan,
            "fp_obj": fp,
            "fp": list(fp),
        })
    print(f"  {len(records)} valid ({n_bad} skipped)")
    return pd.DataFrame(records)


def butina_clusters(fps, cutoff=0.4):
    """RDKit Butina clustering. Returns list of cluster IDs per compound."""
    n = len(fps)
    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - s for s in sims])
    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)
    # clusters is a tuple of tuples (cluster_centroid_idx, member_idx, ...)
    labels = np.zeros(n, dtype=int)
    for cid, cl in enumerate(clusters):
        for idx in cl:
            labels[idx] = cid
    return labels, len(clusters)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--k", type=int, default=5, help="Number of k-means clusters")
    parser.add_argument("--butina-cutoff", type=float, default=0.4, help="Butina distance cutoff")
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nLoading: {args.input}")
    df = load_compounds(args.input)
    X = np.array(df["fp"].tolist(), dtype=float)
    fps = list(df["fp_obj"])

    # PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X)
    var_exp = pca.explained_variance_ratio_

    # k-means
    print(f"Running k-means (k={args.k})...")
    km = KMeans(n_clusters=args.k, random_state=42, n_init=10)
    km_labels = km.fit_predict(X)
    sil = silhouette_score(X, km_labels)
    print(f"  Silhouette score: {sil:.3f}")

    # Butina
    print(f"Running Butina clustering (cutoff={args.butina_cutoff})...")
    but_labels, n_but_clusters = butina_clusters(fps, cutoff=args.butina_cutoff)
    print(f"  {n_but_clusters} Butina clusters")

    df["kmeans_cluster"] = km_labels
    df["butina_cluster"] = but_labels

    # Results CSV
    rows = []
    for cid in range(args.k):
        mask = km_labels == cid
        families = df.loc[mask, "family"].value_counts().to_dict()
        rows.append({
            "algorithm": "kmeans",
            "cluster_id": cid,
            "size": int(mask.sum()),
            "dominant_family": df.loc[mask, "family"].value_counts().index[0],
            "family_breakdown": str(families),
        })
    for cid in range(n_but_clusters):
        mask = but_labels == cid
        families = df.loc[mask, "family"].value_counts().to_dict()
        rows.append({
            "algorithm": "butina",
            "cluster_id": cid,
            "size": int(mask.sum()),
            "dominant_family": df.loc[mask, "family"].value_counts().index[0],
            "family_breakdown": str(families),
        })
    res_df = pd.DataFrame(rows)
    res_df.to_csv(os.path.join(args.output_dir, "clustering_results.csv"), index=False)
    print(f"Saved: {args.output_dir}/clustering_results.csv")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ax0: scaffold family
    for fam, color in FAMILY_COLORS.items():
        mask = df["family"] == fam
        if mask.sum() == 0:
            continue
        axes[0].scatter(X2[mask, 0], X2[mask, 1], c=color, label=fam, s=60, alpha=0.85, edgecolors="white", lw=0.5)
    axes[0].set_title("Scaffold Family", fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=7, loc="best")
    axes[0].set_xlabel(f"PC1 ({var_exp[0]:.1%})", fontsize=9)
    axes[0].set_ylabel(f"PC2 ({var_exp[1]:.1%})", fontsize=9)
    axes[0].spines["top"].set_visible(False); axes[0].spines["right"].set_visible(False)

    # ax1: k-means
    for cid in range(args.k):
        mask = km_labels == cid
        axes[1].scatter(X2[mask, 0], X2[mask, 1], c=CLUSTER_COLORS[cid % len(CLUSTER_COLORS)],
                        label=f"C{cid} (n={mask.sum()})", s=60, alpha=0.85, edgecolors="white", lw=0.5)
    axes[1].set_title(f"k-means (k={args.k}, sil={sil:.3f})", fontsize=11, fontweight="bold")
    axes[1].legend(fontsize=7, loc="best")
    axes[1].set_xlabel(f"PC1 ({var_exp[0]:.1%})", fontsize=9)
    axes[1].set_ylabel(f"PC2 ({var_exp[1]:.1%})", fontsize=9)
    axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False)

    # ax2: Butina
    for cid in range(n_but_clusters):
        mask = but_labels == cid
        axes[2].scatter(X2[mask, 0], X2[mask, 1], c=CLUSTER_COLORS[cid % len(CLUSTER_COLORS)],
                        label=f"C{cid} (n={mask.sum()})", s=60, alpha=0.85, edgecolors="white", lw=0.5)
    axes[2].set_title(f"Butina (cutoff={args.butina_cutoff}, {n_but_clusters} clusters)", fontsize=11, fontweight="bold")
    axes[2].legend(fontsize=7, loc="best")
    axes[2].set_xlabel(f"PC1 ({var_exp[0]:.1%})", fontsize=9)
    axes[2].set_ylabel(f"PC2 ({var_exp[1]:.1%})", fontsize=9)
    axes[2].spines["top"].set_visible(False); axes[2].spines["right"].set_visible(False)

    plt.suptitle("Compound Library Clustering (PCA projection of ECFP4)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "clustering_plot.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {args.output_dir}/clustering_plot.png")

    print(f"\n--- k-means Clusters (k={args.k}, silhouette={sil:.3f}) ---")
    km_summary = res_df[res_df["algorithm"] == "kmeans"][["cluster_id", "size", "dominant_family"]].to_string(index=False)
    print(km_summary)

    print(f"\n--- Butina Clusters ({n_but_clusters} clusters, cutoff={args.butina_cutoff}) ---")
    but_summary = res_df[res_df["algorithm"] == "butina"][["cluster_id", "size", "dominant_family"]].to_string(index=False)
    print(but_summary)

    print("\nDone.")


if __name__ == "__main__":
    main()
