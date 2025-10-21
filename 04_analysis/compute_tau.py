import argparse
import os
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse

def calculate_tau(adata, groupby):
    """
    Calculate Tau per gene and identify the group with highest mean expression.

    Parameters
    ----------
    adata : AnnData
        AnnData object subsetted to genes of interest.
    groupby : str
        Column in `adata.obs` to group cells by.

    Returns
    -------
    pd.DataFrame
        Columns: ['gene', 'tau', 'max_group'].
    """
    # Handle sparse/dense matrices
    X = adata.X.toarray() if issparse(adata.X) else adata.X

    # Mean expression per group
    df_expr = pd.DataFrame(X, index=adata.obs[groupby], columns=adata.var_names)
    df_expr = df_expr.loc[~df_expr.index.isna()]
    mean_expr = df_expr.groupby(level=0).mean().T  # genes × groups

    if mean_expr.empty:
        raise ValueError("Mean-expression table is empty. Check grouping labels.")

    max_expr   = mean_expr.max(axis=1)
    max_group  = mean_expr.idxmax(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        tau = (1 - mean_expr.div(max_expr, axis=0)).sum(axis=1)
        tau /= (mean_expr.shape[1] - 1)
        tau = tau.fillna(0)

    return pd.DataFrame({"gene": mean_expr.index,
                         "tau": tau.values,
                         "max_group": max_group.values})

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--abc_h5ad", required=True, help="ABC AnnData file")
    p.add_argument("--mapping_tsv", required=True,
                   help="TSV with columns 'cl', 'subclass_label', 'class_label'")
    p.add_argument("--filter_h5ad", default=None,
                   help="Optional AnnData whose var_names define genes to keep")
    p.add_argument("--outdir", default=".", help="Output directory")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # --- Load data ----------------------------------------------------
    ABC = sc.read_h5ad(args.abc_h5ad)
    cl_mapping = pd.read_csv(args.mapping_tsv, sep="\t")

    subclass_map = cl_mapping.set_index("cl")["subclass_label"].to_dict()
    class_map    = cl_mapping.set_index("cl")["class_label"].to_dict()

    ABC.obs["cl"] = ABC.obs["cl"].astype(int)
    ABC.obs["subclass_label"] = ABC.obs["cl"].map(subclass_map)
    ABC.obs["class_label"]    = ABC.obs["cl"].map(class_map)

    # --- Optional gene filter ----------------------------------------  # CHANGED
    if args.filter_h5ad:
        filter_ad = sc.read_h5ad(args.filter_h5ad)
        ABC = ABC[:, ABC.var_names.isin(filter_ad.var_names)]

    # --- Tau calculations --------------------------------------------
    tau_sub  = calculate_tau(ABC, "subclass_label")
    tau_cl   = calculate_tau(ABC, "class_label")

    tau_sub.to_csv(os.path.join(args.outdir, "tau_subclass.csv"), index=False)
    tau_cl.to_csv(os.path.join(args.outdir, "tau_class.csv"),   index=False)

if __name__ == "__main__":
    main()
