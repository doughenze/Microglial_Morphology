import scanpy as sc
import pandas as pd
import os
import string
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from tqdm import tqdm
import Mapping
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial
from pathlib import Path
import argparse

import skimage
import cv2
from skimage.morphology import disk, opening, closing
from scipy.ndimage import binary_fill_holes, label, distance_transform_edt
from skimage.segmentation import find_boundaries, watershed
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.segmentation import watershed
import anndata
from adjustText import adjust_text

import Mapping
import os
import string
import glob

import cv2
import geopandas as gpd
import igraph as ig
#import leidenalg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns
import tifffile
import umap
from anndata import AnnData as ad
from matplotlib import patches as mpatches
#from matplotlib_scalebar.scalebar import ScaleBar
import scanpy as sc
import anndata
from shapely.affinity import translate
from shapely.geometry import Polygon, MultiPolygon, box, shape
from skimage import img_as_bool, img_as_ubyte
from skimage.measure import find_contours, regionprops, regionprops_table
from skimage.morphology import skeletonize, opening, disk
from skimage import exposure
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from skimage.segmentation import find_boundaries
from tqdm import tqdm
import matplotlib.patches as patches

from matplotlib_venn import venn2
import gseapy as gp
import seaborn as sns
import scipy.stats as stats
from scipy.ndimage import binary_fill_holes, label
import pandas as pd
from statsmodels.stats.multitest import multipletests


_GENE_COLOURS = {
    "Tmem119": "gold",
    "P2ry12":  "lime",
    "Hexb":    "cyan",
    "Slc1a2":  "magenta",
    "Gria2":   "orange",
}
_GENE_LIST = list(_GENE_COLOURS.keys())

_BATCHID_TO_PATH = {
 '3-mo-male-1':'202405250811_3-mo-male-mouse-1-cerebellum-IHC_VMSC12602/region_1',
 '3-mo-male-2':'202406171454_3m-male-2-IHC_VMSC11602/region_0',
 '3-mo-male-3-rev2':'202407021559_3-mo-male-3-rev2_VMSC12602/region_0',
 '3-mo-female-1-rev2':'202407010924_3-month-female-1-rev2_VMSC12602/region_0',
 '3-mo-female-2':'202405311300_3month-female-2-IHC_VMSC12602/region_0',
 '3-mo-female-3':'202406171409_3m-female-3-IHC_VMSC12602/region_0',
 '24-mo-male-1':'202406101010_24month-male-1-IHC_VMSC12602/region_0',
 '24-mo-male-2':'202406141135_24m-male-2-IHC_VMSC12602/region_0',
 '24-mo-male-4-rev2':'202407011057_24-month-male-4-rev2_VMSC11602/region_0',
 '24-mo-female-1':'202406071120_24m-female-1-IHC_VMSC11602/region_0',
 '24-mo-female-3':'202406071304_24m-female-3-IHC_VMSC12602/region_0',
 '24-mo-female-5':'202406141019_24m-female-5-IHC_VMSC11602/region_0'
}

_PATH_TO_BATCHID = {v.rstrip("/"): k for k, v in _BATCHID_TO_PATH.items()}

def extract_batch_id(exp_path) -> str:
    """
    Map any full experiment path—whether or not it ends with ‘/’—
    to the corresponding key in _BATCHID_TO_PATH.
    """
    # normalise: remove trailing slashes, convert to forward-slash string
    exp_path = Path(exp_path).as_posix().rstrip("/")

    for fragment, batch_id in _PATH_TO_BATCHID.items():
        if fragment in exp_path:
            return batch_id
    raise ValueError(f"Could not map {exp_path!r} to any batch ID.")

def plot_random_cells_from_cluster(data,
                                   cluster_id,
                                   *,
                                   column="leiden",
                                   num_cells=9,
                                   mode="composite",                # "composite" | "segmented" | "mask"
                                   random_state=30,
                                   conversion_rate=1.0,
                                   micron_length=50,
                                   output_pdf=None,
                                   root="/oak/stanford/groups/quake/shared/Vizgen/dough/output/"):

    sub = data[data[column] == cluster_id]
    if len(sub) > num_cells:
        sub = sub.sample(n=num_cells, random_state=random_state)

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    for ax, (_, row) in zip(axes, sub.iterrows()):
        # ------------------------------------------------------------------
        # 1.  tile image (RGB, RGBA or binary) -----------------------------
        # ------------------------------------------------------------------
        tile = load_label_image(row.batchID, row.x, row.y, mode=mode)

        if mode == "segmented":
            ax.imshow(tile[:, :, :3], alpha=tile[:, :, 3] / 255.0, interpolation="nearest")
        else:
            ax.imshow(tile, cmap="gray" if mode == "mask" else None, interpolation="nearest")

        # ------------------------------------------------------------------
        # 2.  mask-mode → overlay transcript dots -------------------------
        # ------------------------------------------------------------------
        if mode == "mask":
            # absolute pixel centre of this tile in the mosaic coordinate frame
            tfm = np.loadtxt(f"{root}{_BATCHID_TO_PATH[row.batchID]}/images/micron_to_mosaic_pixel_transform.csv")
            x_pix = round(row.x * tfm[0, 0] + tfm[0, 2])
            y_pix = round(row.y * tfm[1, 1] + tfm[1, 2])
            x0, y0 = x_pix - 500, y_pix - 500     # top-left mosaic coords of the tile

            # load per-batch transcript table
            tpath = f"transcript_out_slice_by_slice_v5/{row.batchID}_complete.csv"
            tx = pd.read_csv(tpath)
            #print(tx)
            #print(row.Name)
            # keep only {'cell' == row.Name} & gene ∈ whitelist
            tx = tx.loc[(tx["cell"] == row.Name) & (tx["gene"].isin(_GENE_LIST)),
                        ["translate_x", "translate_y", "gene"]]
            
            #print(tx)

            # shift to tile-local coords & clip to bounds
            tx["tile_x"] = tx["translate_x"]# - x0
            tx["tile_y"] = tx["translate_y"]# - y0
            tx = tx[(tx["tile_x"].between(0, 999)) & (tx["tile_y"].between(0, 999))]

            # scatter one colour per gene
            for g, colour in _GENE_COLOURS.items():
                subg = tx[tx["gene"] == g]
                ax.scatter(subg["tile_x"], subg["tile_y"],
                           s=50, marker="o", c=colour, edgecolors="none", alpha=0.9)

        # ------------------------------------------------------------------
        # 3.  cosmetics ----------------------------------------------------
        # ------------------------------------------------------------------
        ax.axis("on")
        ax.set_title(f"Batch {row.batchID} | x={row.x:.0f}, y={row.y:.0f}")
        add_scale_bar(ax, conversion_rate, micron_length,
                      location=(10, tile.shape[0] - 20))

    plt.tight_layout()
    if output_pdf:
        plt.savefig(output_pdf, format="pdf", bbox_inches="tight")
    plt.show()
    
_clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))

def _as_8bit_with_clahe(img: np.ndarray) -> np.uint8:
    """Rescale to [0,255] then apply CLAHE."""
    im = cv2.normalize(im, None, 0, 255,
                           cv2.NORM_MINMAX).astype(np.uint8)
    #img8 = exposure.rescale_intensity(img, in_range='image', out_range=(0, 255)).astype(np.uint8)
    return im#_clahe.apply(img8)

def _crop(tile: np.ndarray, x_pix: int, y_pix: int, box_size: int) -> np.ndarray:
    """Return a (2·box_size)² crop centred on (x_pix, y_pix)."""
    xs, xe = max(x_pix - box_size, 0), min(x_pix + box_size, tile.shape[1])
    ys, ye = max(y_pix - box_size, 0), min(y_pix + box_size, tile.shape[0])

    out = np.zeros((2 * box_size, 2 * box_size), dtype=tile.dtype)
    out_y0, out_x0 = max(0, -(y_pix - box_size)), max(0, -(x_pix - box_size))
    out[out_y0:out_y0 + (ye - ys), out_x0:out_x0 + (xe - xs)] = tile[ys:ye, xs:xe]
    return out

def extract_sub_image_with_padding(image, bbox, padding=10):
    min_row, min_col, max_row, max_col = bbox
    min_row = max(min_row - padding, 0)
    min_col = max(min_col - padding, 0)
    max_row = min(max_row + padding, image.shape[0])
    max_col = min(max_col + padding, image.shape[1])
    return image[min_row:max_row, min_col:max_col], (min_row, min_col)

def add_scale_bar(ax, conversion_rate, micron_length, color='red', location=(10, 10), thickness=50, fontsize=12):
    """
    Add a scale bar to the given axis.

    Args:
    - ax: The axis on which to draw the scale bar.
    - conversion_rate: The pixel-to-micron conversion factor (pixels per micron).
    - micron_length: The length of the scale bar in microns.
    - color: The color of the scale bar.
    - location: A tuple specifying the (x, y) location of the scale bar.
    - thickness: The thickness of the scale bar in pixels.
    """
    # Calculate the length of the scale bar in pixels
    pixel_length = conversion_rate * micron_length
    
    # Create a rectangle for the scale bar
    scale_bar = patches.Rectangle(location, pixel_length, thickness, linewidth=0, edgecolor=color, facecolor=color)
    
    # Add the scale bar to the plot
    ax.add_patch(scale_bar)
    text_x = location[0] + pixel_length + 10  # Place text to the right of the scale bar
    text_y = location[1] + thickness / 2      # Vertically center the text with the scale bar
    ax.text(text_x, text_y, f'{micron_length} μm', color=color, fontsize=fontsize, va='center')
    
_clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))

def _as_8bit_with_clahe(img: np.ndarray) -> np.ndarray:
    """
    Scale to 0-255 uint8, then apply CLAHE (histogram equalisation).
    """
    img8 = exposure.rescale_intensity(img, in_range="image", out_range=(0, 255)).astype(np.uint8)
    return _clahe.apply(img8)


def load_label_image(batchID: str,
                     x_ax: float,
                     y_ax: float,
                     *,
                     mode: str = "composite",        # "composite" | "segmented" | "mask"
                     box_size: int = 500,
                     root: str = "/oak/stanford/groups/quake/shared/Vizgen/dough/output/"):

    # ── 1. coordinate transform ───────────────────────────────────────────────
    tfm = np.loadtxt(f"{root}{_BATCHID_TO_PATH[batchID]}/images/micron_to_mosaic_pixel_transform.csv")
    x_pix = round(x_ax * tfm[0, 0] + tfm[0, 2])
    y_pix = round(y_ax * tfm[1, 1] + tfm[1, 2])

    # ── 2. load full-frame channels ───────────────────────────────────────────
    iba1_raw = Mapping.load_tiff_image(f"{root}{_BATCHID_TO_PATH[batchID]}/binary_image.tif")

    primary_path  = Path(root) / _BATCHID_TO_PATH[batchID] / "binary_image_astro.tif"
    fallback_path = Path(root) / _BATCHID_TO_PATH[batchID] / "images" / "mosaic_Anti-Chicken_z3.tif"

    if primary_path.exists():                           # <-- added existence check
        gfap_raw = Mapping.load_tiff_image(primary_path)
    else:                        # <-- added fallback
        gfap_raw = Mapping.load_tiff_image(fallback_path)
    #gfap_raw = Mapping.load_tiff_image(fallback_path)
    #gfap_raw = Mapping.load_tiff_image(f"{root}{_BATCHID_TO_PATH[batchID]}/binary_image_astro.tif")
    #gfap_raw = Mapping.load_tiff_image(f"{root}{_BATCHID_TO_PATH[batchID]}/images/mosaic_Anti-Chicken_z3.tif")
    dapi_raw = Mapping.load_tiff_image(f"{root}{_BATCHID_TO_PATH[batchID]}/images/mosaic_DAPI_z3.tif")
    #dapi_paths = sorted(glob.glob(f"{root}{batchID}/images/mosaic_DAPI_z*.tif"))
    #dapi_stack = np.stack([Mapping.load_tiff_image(path) for path in dapi_paths])

    # ── 3. crop around the requested point ────────────────────────────────────
    iba1 = _crop(iba1_raw, x_pix, y_pix, box_size)
    gfap = _crop(gfap_raw, x_pix, y_pix, box_size)
    dapi = _crop(dapi_raw, x_pix, y_pix, box_size)
    #dapi_crops = [_crop(dapi_plane, x_pix, y_pix, box_size) for dapi_plane in dapi_stack]

    # ── 4. output selector ────────────────────────────────────────────────────
    if mode == "composite":
        return np.dstack([
            #_as_8bit_with_clahe(gfap),      # R ← GFAP
            cv2.normalize(gfap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            #_as_8bit_with_clahe(iba1),      # G ← IBA1
            cv2.normalize(iba1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            #_as_8bit_with_clahe(dapi)       # B ← DAPI
            cv2.normalize(dapi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        ])

    elif mode == "segmented":
        iba1_mask = _get_iba1_target_mask(iba1, box_size)
        gfap_mask = _get_gfap_mask(gfap)

        # build RGBA overlay (uint8, 0–255)
        overlay = np.zeros((2 * box_size, 2 * box_size, 4), dtype=np.uint8)

        # GFAP  – solid red, fully opaque
        overlay[gfap_mask == 1] = (255, 0, 0, 255)

        # IBA1 – white, α = 0.5 (≈128)
        iba1_pixels = (iba1_mask == 1)
        
        overlay[iba1_pixels, :3] = (255, 255, 255)
        overlay[iba1_pixels, 3]  = 128

        return overlay                      # RGBA

    elif mode == "mask":                    # IBA1 minus GFAP
        iba1_mask = _get_iba1_target_mask(iba1, box_size)
        dapi_mask = _get_dapi_target_mask(dapi, box_size)
        gfap_mask = _get_gfap_mask(gfap)
        iba1_mask[(gfap_mask == 1) & (dapi_mask == 0)] = 0       # subtract overlap
        iba1_mask_fill = binary_fill_holes(iba1_mask)
        #final_mask = iba1_mask_fill + dapi_mask
        #final_mask = (iba1_mask_fill == 1) | (dapi_mask == 1)
        return iba1_mask_fill.astype(np.uint8)

    else:
        raise ValueError(f"Unknown mode '{mode}'.")
        
        
# segment gfap, dapi, and IBA1
def roi_picker(im, point=(500, 500), dapi=False):
    """
    This accepts a binary image and finds the largest object which is closest to the center point of the image
    Best approximation for a cell
    """
    
    from scipy.ndimage import binary_fill_holes, label, distance_transform_edt
    labeled_array, num_features = label(im)
    
    if dapi:
        # Apply distance transform
        distance = distance_transform_edt(labeled_array > 0)
        
        # Generate markers using connected components after thresholding
        coords = peak_local_max(distance, footprint=np.ones((9, 9)), labels=(labeled_array > 0))
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = label(mask)
        # Apply watershed
        watershed_labels = watershed(-distance, markers, mask=(labeled_array > 0))
        
        # Update the labeled array with the watershed labels
        labeled_array = watershed_labels

        # Initialize variables to track the closest region
        closest_region_label = None
        min_distance = float('inf')
        regions = regionprops(labeled_array)

        # Find the closest region to the specified point
        for region in regions:
            if region.area > 500:
                # Calculate the distance from the point to the region's centroid
                region_centroid = np.array(region.centroid)
                distance = np.linalg.norm(region_centroid - np.array(point))

                # Update if this region is closer than previous ones
                if distance < min_distance:
                    min_distance = distance
                    closest_region_label = region.label

        if closest_region_label is not None:
            isolated_component = labeled_array == closest_region_label
            label_image = np.zeros_like(labeled_array)
            label_image[isolated_component] = 1
            return label_image
        else:
            return labeled_array
    else:
        # The original functionality to find the largest region near the point
        largest_component_label = None
        max_area = 0
        regions = regionprops(labeled_array)
        for region in regions:
            if region.area > 500:
                boundary_coords = np.column_stack(np.where(labeled_array == region.label))
                distances = np.linalg.norm(boundary_coords - np.array(point), axis=1)
                min_dist = np.min(distances)

                if min_dist < 50:
                    if region.area > max_area:
                        max_area = region.area
                        largest_component_label = region.label

        if largest_component_label is not None:
            isolated_component = labeled_array == largest_component_label
            label_image = np.zeros_like(labeled_array)
            label_image[isolated_component] = 1
            return label_image
        else:
            return labeled_array
def _get_iba1_target_mask(iba1_tile: np.ndarray, box_size: int) -> np.ndarray:
    #im = cv2.normalize(iba1_tile, None, 0, 255,
    #                       cv2.NORM_MINMAX).astype(np.uint8)
    im = ((iba1_tile - iba1_tile.min()) / (iba1_tile.max() - iba1_tile.min()) * 255).astype(np.uint8)
    subtract = cv2.fastNlMeansDenoising(im)
    pre = cv2.adaptiveThreshold(255 - subtract, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 201, 2)  # max projection window size.

    opened = opening(255 - pre, disk(3))
    pre    = closing(opened, disk(3))

    filled, _ = binary_fill_holes(pre), None
    labeled, _ = label(filled)

    centre = (box_size, box_size)
    target, max_area = None, 0
    for r in regionprops(labeled):
        if r.area > 500:
            bnd = find_boundaries(labeled == r.label, mode='outer')
            d   = np.linalg.norm(np.column_stack(np.where(bnd)) - np.array(centre), axis=1).min()
            if d < 50 and r.area > max_area:
                target, max_area = r.label, r.area

    mask = np.zeros_like(labeled, dtype=np.uint8)
    if target is not None:
        mask[labeled == target] = 1
    return mask

def _get_dapi_target_mask(dapi_tile: np.ndarray, box_size: int) -> np.ndarray:
    from skimage.measure import label
    #if dapi_tile.dtype == np.uint16:
    im = cv2.normalize(dapi_tile, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    den = cv2.fastNlMeansDenoising(im)
    th  = cv2.adaptiveThreshold(255-den, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 155, 2)
    mask = closing(opening(255-th, disk(2)), disk(2)).astype(bool)
    dist = distance_transform_edt(mask)
    coords  = peak_local_max(dist, footprint=np.ones((9,9)), labels=mask)
    markers = label(np.zeros_like(mask, bool))
    markers[tuple(coords.T)] = np.arange(1, len(coords)+1)
    ws = watershed(-dist, markers, mask=mask)

    # pick nucleus whose centroid is nearest (500,500)
    centre = np.array([box_size, box_size])
    best, min_d = None, np.inf
    for r in regionprops(ws):
        d = np.linalg.norm(np.array(r.centroid) - centre)
        if d < min_d and r.area > 500:
            min_d, best = d, r.label
    return (ws == best) if best else mask

def _get_gfap_mask(gfap_tile: np.ndarray, area_thr: int = 1000) -> np.ndarray:
    # identical mechanics to IBA1, minus the centre-distance test
    im = ((gfap_tile - gfap_tile.min()) / (gfap_tile.max() - gfap_tile.min()) * 255).astype(np.uint8)
    im = cv2.normalize(gfap_tile, None, 0, 255,
                           cv2.NORM_MINMAX).astype(np.uint8)
    subtract = cv2.fastNlMeansDenoising(im,h=10)   #Changed
    pre = cv2.adaptiveThreshold(255 - subtract, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 245, 6)

    opened = opening(255 - pre, disk(5))
    pre    = closing(opened, disk(3))

    filled = binary_fill_holes(pre)
    labeled, _ = label(filled)

    mask = np.zeros_like(labeled, dtype=np.uint8)
    for r in regionprops(labeled):
        if r.area > area_thr:
            mask[labeled == r.label] = 1
    return mask


JOBS = [
    # idx 0‒2  : cluster 4
    dict(cluster_id="4", mode="composite", random_state=None,
         pdf="ramified_4_examples_comp.pdf"),
    dict(cluster_id="4", mode="segmented", random_state=None,
         pdf="ramified_4_examples_seg.pdf"),
    dict(cluster_id="4", mode="mask", random_state=None,
         pdf="ramified_4_examples_mask.pdf"),
    # idx 3‒5  : cluster 3
    dict(cluster_id="3", mode="composite", random_state=12,
         pdf="ramified_3_examples_comp.pdf"),
    dict(cluster_id="3", mode="segmented", random_state=12,
         pdf="ramified_3_examples_seg.pdf"),
    dict(cluster_id="3", mode="mask", random_state=12,
         pdf="ramified_3_examples_mask.pdf"),
    # idx 6‒8  : cluster 2
    dict(cluster_id="2", mode="composite", random_state=None,
         pdf="ramified_2_examples_comp.pdf"),
    dict(cluster_id="2", mode="segmented", random_state=None,
         pdf="ramified_2_examples_seg.pdf"),
    dict(cluster_id="2", mode="mask", random_state=None,
         pdf="ramified_2_examples_mask.pdf"),
    # idx 9‒11 : cluster 1
    dict(cluster_id="1", mode="composite", random_state=11,
         pdf="amoeboid_1_examples_comp.pdf"),
    dict(cluster_id="1", mode="segmented", random_state=11,
         pdf="amoeboid_1_examples_seg.pdf"),
    dict(cluster_id="1", mode="mask", random_state=11,
         pdf="amoeboid_1_examples_mask.pdf"),
    # idx 12‒14: cluster 0
    dict(cluster_id="0", mode="composite", random_state=10,
         pdf="amoeboid_0_examples_comp.pdf"),
    dict(cluster_id="0", mode="segmented", random_state=10,
         pdf="amoeboid_0_examples_seg.pdf"),
    dict(cluster_id="0", mode="mask", random_state=10,
         pdf="amoeboid_0_examples_mask.pdf"),
]


def load_inputs():
    root = Path("/oak/stanford/groups/quake/shared/Vizgen/dough/output/")
    ad   = sc.read_h5ad("../../../Transciptomic_labels_and_morphology_labels_full.h5ad")
    ad.X = ad.layers["total_counts"].copy()
    sc.pp.filter_cells(ad, min_counts=75)

    tfm_path = root / "202407010924_3-month-female-1-rev2_VMSC12602/region_0" \
                     / "images/micron_to_mosaic_pixel_transform.csv"
    # first row, second column = micron-per-pixel
    mosaic_to_micron = pd.read_csv(tfm_path, delim_whitespace=True).iloc[0, 1]
    return ad, mosaic_to_micron


# -----------------------------------------------------------------------------#
#  main                                                                        #
# -----------------------------------------------------------------------------#
def main(idx: int):
    job = JOBS[idx]
    ad, conv = load_inputs()

    kwargs = dict(
        column="ordered_morph",
        num_cells=9,
        conversion_rate=conv,
        mode=job["mode"],
        output_pdf=job["pdf"],
    )
    if job["random_state"] is not None:
        kwargs["random_state"] = job["random_state"]

    print(f"task {idx}: cluster {job['cluster_id']} | mode {job['mode']}")
    plot_random_cells_from_cluster(ad.obs, cluster_id=job["cluster_id"], **kwargs)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task-index", type=int, required=True,
                   help="value of $SLURM_ARRAY_TASK_ID")
    args = p.parse_args()
    if not (0 <= args.task_index < len(JOBS)):
        raise IndexError(f"task-index {args.task_index} out of range 0-{len(JOBS)-1}")
    main(args.task_index)