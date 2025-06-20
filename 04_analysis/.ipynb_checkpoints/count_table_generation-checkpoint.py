"""
Count microglial (cyto / nuc) transcripts slice-by-slice.

Usage
-----
python slice_counter.py --exp /abs/path/to/Shapes_Spatial/3-mo-male-1/ \
                        --h5  ../03_morph_embedding/Shape_500.h5ad \
                        --outdir  transcript_out_slice_by_slice

All heavy lifting is confined to `run_experiment()`, so you can call that
function from a notebook or another script if you like.

Author  : Douglas Henze
Created : 2025-05-13
"""

import argparse, os, sys, json, pathlib
from pathlib import Path
from typing import Tuple, List
import string
from functools import partial, lru_cache

import scanpy as sc
import pandas as pd
import numpy as np
import cv2, skimage, Mapping, matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp
import geopandas as gpd
from shapely.geometry import Point
from skimage.morphology import disk, opening, closing
from scipy.ndimage import binary_fill_holes, label, distance_transform_edt
from skimage.segmentation import find_boundaries, watershed
from skimage.feature import peak_local_max
from skimage.measure import regionprops


# attempting a monkey-patch to make this run much faster:
# Only perform this step if their is enough memory
# ─────────────────────────────────────────────────────────────
#   1.  Monkey-patch Mapping.load_tiff_image   (do this ONCE)
# ─────────────────────────────────────────────────────────────
Mapping._orig_load = Mapping.load_tiff_image                # keep original

@lru_cache(maxsize=None)
def _load_once(path: str):
    path = os.path.abspath(path)            # canonical key
    return Mapping._orig_load(path)         # real disk read – first time only

Mapping.load_tiff_image = _load_once        # << patch in
# ─────────────────────────────────────────────────────────────


def find_filtered_transcripts(experiment_path):
    region_types = ['region_0', 'region_1']
    for region in region_types:
        file_path = f'{experiment_path}baysor/detected_transcripts.csv'
        if os.path.exists(file_path):
            return pd.read_csv(file_path,index_col=0)
    return None

def extract_sub_image_with_padding(image, bbox, padding=10):
    min_row, min_col, max_row, max_col = bbox
    min_row = max(min_row - padding, 0)
    min_col = max(min_col - padding, 0)
    max_row = min(max_row + padding, image.shape[0])
    max_col = min(max_col + padding, image.shape[1])
    return image[min_row:max_row, min_col:max_col], (min_row, min_col)

def load_images(batchID, x_ax, y_ax, raw_im, raw_dapi,transcripts):
    root = '/hpc/projects/group.quake/doug/Shapes_Spatial/'
    
    transform_file = f'{root}{batchID}/images/micron_to_mosaic_pixel_transform.csv'
    transform_df = pd.read_table(transform_file, sep=' ', header=None)
    transformation_matrix = transform_df.values
    
    x_ax = round(x_ax * transformation_matrix[0, 0] + transformation_matrix[0, 2])
    y_ax = round(y_ax * transformation_matrix[1, 1] + transformation_matrix[1, 2])
    
    box_size = 500
    x_start = x_ax - box_size
    x_end = x_ax + box_size
    y_start = y_ax - box_size
    y_end = y_ax + box_size
    
    # Extract the sub-image, ensuring the indices are within bounds
    sub_image = np.zeros((2 * box_size, 2 * box_size), dtype=raw_im.dtype)
    sub_dapi = np.zeros((2 * box_size, 2 * box_size), dtype=raw_dapi.dtype)
    
    raw_x_start = max(x_start, 0)
    raw_x_end = min(x_end, raw_im.shape[1])
    raw_y_start = max(y_start, 0)
    raw_y_end = min(y_end, raw_im.shape[0])
    
    sub_x_start = max(0, -x_start)
    sub_x_end = sub_x_start + (raw_x_end - raw_x_start)
    sub_y_start = max(0, -y_start)
    sub_y_end = sub_y_start + (raw_y_end - raw_y_start)
    
    sub_image[sub_y_start:sub_y_end, sub_x_start:sub_x_end] = raw_im[raw_y_start:raw_y_end, raw_x_start:raw_x_end]
    sub_dapi[sub_y_start:sub_y_end, sub_x_start:sub_x_end] = raw_dapi[raw_y_start:raw_y_end, raw_x_start:raw_x_end]
    
    transcripts_sub = transcripts.loc[
        (transcripts.mosaic_x < raw_x_end) & (transcripts.mosaic_x > raw_x_start) &
        (transcripts.mosaic_y > raw_y_start) & (transcripts.mosaic_y < raw_y_end)
    ].copy()  # Explicitly create a copy

    # Now assign the new values without triggering the warning
    transcripts_sub['translate_x'] = transcripts_sub.mosaic_x - raw_x_start
    transcripts_sub['translate_y'] = transcripts_sub.mosaic_y - raw_y_start
    return sub_image, sub_dapi, transcripts_sub, (raw_y_start,raw_y_end, raw_x_start, raw_x_end)

def segment_image(im,window_size,foreground=True, dapi=False):
    """
    This accepts a raw unage abd oerfirnes segmentation following a denoising. Segmentation methodology is adaptive thresholding
    We then remove small objects upon using an opening and closing of the segmented mask
    """
    if im.dtype == 'uint16':
        im = ((im - im.min()) / (im.max() - im.min()) * 255).astype(np.uint8)
    subtract = cv2.fastNlMeansDenoising(im)
    if foreground:
        pre = cv2.adaptiveThreshold((255 - subtract), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, window_size, 2)
    else:
        pre = cv2.adaptiveThreshold((subtract), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, window_size, 2)
    opened = opening(255 - pre, disk(3))
    pre = closing(opened, disk(3))
    filled_image = binary_fill_holes(pre).astype(np.uint8)
        
    return filled_image

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
        
def count_gene_overlaps(transcripts, dapi, micro, filled_dapi):
    """
    Counts the occurrences of barcodes (rows) for each gene that overlap with dapi_1, 
    with the binary difference of micro_1 - dapi_1, and with the binary difference of micro_1 - filled_dapi.
    
    Parameters:
        transcripts (pd.DataFrame): A DataFrame containing 'genes', 'translate_x', and 'translate_y' columns.
        dapi_1 (np.array): Binary image representing the region of interest (e.g., DAPI stained area).
        micro_1 (np.array): Binary image representing a larger or different region of interest.
        filled_dapi (np.array): Binary image representing another region of interest.
        
    Returns:
        result (pd.DataFrame): A subset of the input DataFrame 'transcripts' containing only the barcodes that overlap with
                               the binary difference between micro_1 and dapi_1 or the binary difference between 
                               micro_1 and filled_dapi, along with overlap counts per gene.
        dapi_only (pd.DataFrame): A subset of the input DataFrame 'transcripts' containing only the barcodes that overlap exclusively with dapi_1.
    """
    
    transcripts = transcripts.copy()
    # Calculate the differences
    binary_diff_dapi = np.logical_and(micro.astype(bool), np.logical_not(dapi.astype(bool)))
    binary_diff_filled_dapi = np.logical_and(micro.astype(bool), np.logical_not(filled_dapi.astype(bool)))
    
    results = transcripts[
        (transcripts['translate_x'].astype(int) >= 0) & (transcripts['translate_x'].astype(int) < binary_diff_filled_dapi.shape[1]) &
        (transcripts['translate_y'].astype(int) >= 0) & (transcripts['translate_y'].astype(int) < binary_diff_filled_dapi.shape[0])
    ]

    results = results[
        binary_diff_filled_dapi[results['translate_y'].astype(int), results['translate_x'].astype(int)]
    ]
    
    dapi_only = transcripts[
        (transcripts['translate_x'].astype(int) >= 0) & (transcripts['translate_x'].astype(int) < dapi.shape[1]) &
        (transcripts['translate_y'].astype(int) >= 0) & (transcripts['translate_y'].astype(int) < dapi.shape[0])
    ]

    dapi_only = dapi_only[
        dapi.astype(bool)[dapi_only['translate_y'].astype(int), dapi_only['translate_x'].astype(int)]
    ]
    

    return results, dapi_only

def calculate_areas(dataframe, dapi_labeled_array, non_dapi_labeled_array):
    """
    Calculate the total area of the DAPI (nucleus) and non-DAPI (non-nucleus) regions.

    Parameters:
        dataframe (pd.DataFrame): A DataFrame with one row, which will be used to create the output DataFrame.
        dapi_labeled_array (np.array): A labeled array where each unique integer represents a different object in the nucleus (DAPI).
        non_dapi_labeled_array (np.array): A labeled array where each unique integer represents a different object in the non-nucleus (Non-DAPI).

    Returns:
        result_df (pd.DataFrame): A DataFrame with the same index as the input DataFrame, containing two columns:
                                  'DAPI Area' and 'Non-DAPI Area'.
    """
    
    # Calculate total area for DAPI (nucleus)
    dapi_total_area = sum(region.area for region in regionprops(dapi_labeled_array))
    
    # Calculate total area for Non-DAPI (non-nucleus)
    non_dapi_total_area = sum(region.area for region in regionprops(non_dapi_labeled_array))
    
    # Create a new DataFrame to store the results
    result_df = pd.DataFrame({
        'DAPI Area': [dapi_total_area],
        'Non-DAPI Area': [non_dapi_total_area]
    }, index=dataframe.index)
    
    return result_df

def generate_counts_matrix(dataframe, var_names):
    """
    Generate a counts matrix where columns are genes and rows contain the number of barcodes 
    for that gene present in the dataframe.

    Parameters:
        dataframe (pd.DataFrame): The input DataFrame containing barcode information.
        var_names (list or pd.Index): List of gene names (matching adata.var_names).

    Returns:
        pd.DataFrame: A DataFrame with genes as columns and the number of barcodes for each gene.
    """
    # Filter the dataframe to only include genes in var_names
    filtered_df = dataframe[dataframe['gene'].isin(var_names)]
    
    # Count the number of barcodes for each gene
    counts = filtered_df['gene'].value_counts().reindex(var_names, fill_value=0)
    
    # Convert the counts to a DataFrame
    counts_df = counts.to_frame().T
    
    return counts_df

def rename_index(df, adata,blanks):
    """
    This accepts a dataframe of counts, an adata with a set number of genes, and a list of blanks.
    
    We then output a true counts table format like what is seen in an adata.X
    """
    
    counts_matrix_result = generate_counts_matrix(df, adata.var_names.tolist()+blanks)
    counts_matrix_result.index = adata.obs.index
    return counts_matrix_result

def generate_transcript_spreadsheet(transcripts, dapi, micro, ad_test):
    # Calculate the union of dapi_1 and micro_1
    union_mask = np.logical_or(dapi.astype(bool), micro.astype(bool))
    
    filtered_transcripts = transcripts[
        (transcripts['translate_x'].astype(int) >= 0) & (transcripts['translate_x'].astype(int) < union_mask.shape[1]) &
        (transcripts['translate_y'].astype(int) >= 0) & (transcripts['translate_y'].astype(int) < union_mask.shape[0])
    ]

    filtered_transcripts = filtered_transcripts[
        union_mask[filtered_transcripts['translate_y'].astype(int), filtered_transcripts['translate_x'].astype(int)]
    ]

    # Create a new DataFrame for the spreadsheet
    spreadsheet_df = pd.DataFrame({
        'x': filtered_transcripts['translate_x'].astype(int),
        'y': filtered_transcripts['translate_y'].astype(int),
        'gene': filtered_transcripts['gene'],
        'cell': ad_test.obs.Name.iloc[0]
    })
    return spreadsheet_df

def apply_bbox_to_image(image, bbox):
    """
    Apply a bounding box to a separate image by cropping or highlighting the region.
    
    Parameters:
    - image: The image to which the bounding box will be applied (as a NumPy array).
    - bbox: The bounding box as a tuple (min_row, min_col, max_row, max_col).
    
    Returns:
    - Cropped or highlighted image.
    """
    min_row, min_col, max_row, max_col = bbox
    
    # Crop the region from the new image
    cropped_image = image[min_row:max_row, min_col:max_col]

    # Show the original image with the bounding box    
    return cropped_image

BOX_SIZE = 500
TILE_SHAPE = (2*BOX_SIZE, 2*BOX_SIZE)

def _crop_tile(im_full, x0, y0):
    """Crop & zero-pad a 1000×1000 box from im_full."""
    H, W = im_full.shape[:2]
    x1, y1 = x0 + 2*BOX_SIZE, y0 + 2*BOX_SIZE

    tile = np.zeros(TILE_SHAPE, dtype=im_full.dtype)
    raw_x0, raw_x1 = max(x0, 0), min(x1, W)
    raw_y0, raw_y1 = max(y0, 0), min(y1, H)
    sub_x0         = max(0, -x0)
    sub_y0         = max(0, -y0)

    tile[sub_y0:sub_y0+(raw_y1-raw_y0),
         sub_x0:sub_x0+(raw_x1-raw_x0)] = im_full[raw_y0:raw_y1, raw_x0:raw_x1]
    return tile


def load_plane_tiles(root: str,
                     batch: str,
                     z: int,
                     x_pix: int,
                     y_pix: int):
    """
    Returns
    -------
    raw_rabbit, raw_dapi : np.ndarray (uint8/16, 1000×1000)
    """
    rabbit_path = root + batch + f"/images/mosaic_Anti-Rabbit_z{z}.tif"
    dapi_path   = root + batch + f"/images/mosaic_DAPI_z3.tif" # center of the imaging stack so we remove out of focus stuff.
    chicken_path = root + batch + f"/images/mosaic_Anti-Chicken_z{z}.tif"

    for p in (rabbit_path, dapi_path, chicken_path):
        if not os.path.exists(p): raise FileNotFoundError(p)

    im_rabbit = Mapping.load_tiff_image(rabbit_path)
    im_dapi   = Mapping.load_tiff_image(dapi_path)
    im_chicken = Mapping.load_tiff_image(chicken_path) 

    x0, y0 = x_pix - BOX_SIZE, y_pix - BOX_SIZE
    tile_rabbit = _crop_tile(im_rabbit, x0, y0)
    tile_dapi   = _crop_tile(im_dapi,   x0, y0)
    tile_chicken = _crop_tile(im_chicken, x0, y0) 
    return tile_rabbit, tile_dapi, tile_chicken 

# just error check to make sure the transcripts are in the right tile
def crop_transcripts_to_tile(df: pd.DataFrame,
                             x_pix: int,
                             y_pix: int) -> pd.DataFrame:
    """
    Given mosaic coords in df.mosaic_x/y, keep only those in the
    1000×1000 tile and add translate_x/y columns **relative** to it.
    """
    x0, y0 = x_pix - BOX_SIZE, y_pix - BOX_SIZE
    keep = (
        (df.mosaic_x >= x0) & (df.mosaic_x < x0 + 2*BOX_SIZE) &
        (df.mosaic_y >= y0) & (df.mosaic_y < y0 + 2*BOX_SIZE)
    )
    sub = df.loc[keep].copy()
    sub["translate_x"] = sub.mosaic_x - x0
    sub["translate_y"] = sub.mosaic_y - y0
    return sub

def segment_plane_micro(im: np.ndarray,
                        maxproj_micro: np.ndarray,
                        win: int = 121,
                        area_thresh: int = 500) -> np.ndarray:
    """
    Segment the Anti-Rabbit slice for this Z plane.

    Returns
    -------
    mask : bool ndarray, shape (1000,1000)
        Union of *all* components ≥ area_thresh that also overlap the
        2-D max-projection microglia mask.
    """
    from skimage.measure import label
    # --- 1 · pre-processing -----------------------------------------
    if im.dtype == np.uint16:
        im = cv2.normalize(im, None, 0, 255,
                           cv2.NORM_MINMAX).astype(np.uint8)

    den = cv2.fastNlMeansDenoising(im)
    th  = cv2.adaptiveThreshold(255 - den, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, win, 2)
    mask = closing(opening(255 - th, disk(3)), disk(3)).astype(bool)

    # --- 2 · collect ALL sizable blobs ------------------------------
    labels = label(mask)
    keep   = np.zeros_like(mask, bool)

    for r in regionprops(labels):
        if r.area >= area_thresh:
            component = labels == r.label
            keep |= component

    # --- 3 · intersect with max-projection mask ---------------------
    keep &= maxproj_micro            # << the key new line
    return keep

def segment_plane_chicken(im, win=121, area_thresh=500):
    """
    Segment Anti-Chicken channel (e.g. vasculature or other exclusion zone).
    Returns a boolean mask of the *entire* Anti-Chicken region in the tile.
    """
    from skimage.measure import label
    if im.dtype == np.uint16:
        im = cv2.normalize(im, None, 0, 255,
                           cv2.NORM_MINMAX).astype(np.uint8)
    den = cv2.fastNlMeansDenoising(im)
    th  = cv2.adaptiveThreshold(255-den, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, win, 2)
    mask = closing(opening(255-th, disk(3)), disk(3)).astype(bool)
    # keep *all* blobs above size threshold
    labels = label(mask)
    keep = np.zeros_like(mask, bool)
    for r in regionprops(labels):
        if r.area > area_thresh:
            keep |= labels == r.label
    return keep

def segment_plane_dapi(im, win=255, area_thresh=500):
    """Watershed-split nuclei, return mask of the ONE nucleus closest to centre."""
    from skimage.measure import label
    if im.dtype == np.uint16:
        im = cv2.normalize(im, None, 0, 121, cv2.NORM_MINMAX).astype(np.uint8)

    den = cv2.fastNlMeansDenoising(im)
    th  = cv2.adaptiveThreshold(255-den, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, win, 2)
    mask = closing(opening(255-th, disk(2)), disk(2)).astype(bool)
    dist = distance_transform_edt(mask)
    coords  = peak_local_max(dist, footprint=np.ones((9,9)), labels=mask)
    markers = label(np.zeros_like(mask, bool))
    markers[tuple(coords.T)] = np.arange(1, len(coords)+1)
    ws = watershed(-dist, markers, mask=mask)

    # pick nucleus whose centroid is nearest (500,500)
    centre = np.array([BOX_SIZE, BOX_SIZE])
    best, min_d = None, np.inf
    for r in regionprops(ws):
        d = np.linalg.norm(np.array(r.centroid) - centre)
        if d < min_d and r.area > area_thresh:
            min_d, best = d, r.label
    return (ws == best) if best else mask

def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a,  mask_b).sum()
    return inter / union if union else 0.0

def plane_is_valid(plane_mask: np.ndarray,
                   maxproj_mask: np.ndarray,
                   min_iou: float = .15) -> bool:
    """
    Accept the Z plane if it overlaps the 2-D max-projection mask.
    """
    return mask_iou(plane_mask, maxproj_mask) >= min_iou

def transcripts_in_plane(df: pd.DataFrame, z: int) -> pd.DataFrame:
    """
    Keep only transcript rows that belong to this Z slice.
    """
    return df.loc[df.global_z == z]


def keep_overlapping_spots(df_plane: pd.DataFrame,
                           plane_mask: np.ndarray) -> pd.DataFrame:
    """
    Keep spots whose (translate_x, translate_y) sit inside plane_mask.
    """
    inside = plane_mask[
        df_plane.translate_y.astype(int),
        df_plane.translate_x.astype(int)
    ]
    return df_plane[inside]

    
def write_debug_png(dst, rabbit, mask_micro, mask_dapi, mask_chicken, spots):
    #dst.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.5,3.5))
    overlay = np.dstack([rabbit.copy()]*3)
    overlay[mask_micro] = (  0,255,  0)   # green micro
    overlay[mask_dapi ] = (255,  0,255)   # magenta nucleus
    overlay[mask_chicken] = (  0,255, 255) # cyan astrocytes
    for _, r in spots.iterrows():
        rr, cc = int(r.translate_y), int(r.translate_x)
        cv2.drawMarker(overlay, (cc, rr), (255,0,0),
                       markerType=cv2.MARKER_CROSS,
                       markerSize=4, thickness=1)
    ax.imshow(overlay); ax.axis("off")
    fig.tight_layout();
    plt.show(fig)
    #fig.savefig(dst, dpi=150); plt.close(fig)
    
def count_by_plane(batch, transcripts, maxproj_micro,
                   root, x_pix, y_pix, min_iou=.15):
    
    """
    Counts the number of transcripts that overlap with the segmentations on each plane, the drivers here are batch so that we can load the whole image,
    transcripts so we know what to match to the segmentation and the max projected micro so that we can limit the segmentation to part of the original cell,
    not random signal found in the periphery.
    
    """

    rows_cyto, rows_nuc, rows_tot = [], [], []
    for z in transcripts.global_z.astype(int).unique():

        raw_rabbit, raw_dapi, raw_chicken = load_plane_tiles(
            root, batch, z, x_pix, y_pix)

        micro_mask   = segment_plane_micro(raw_rabbit, maxproj_micro)
        if mask_iou(micro_mask, maxproj_micro) < min_iou:
            continue                                    # bad slice → skip

        dapi_mask    = segment_plane_dapi(raw_dapi)
        chicken_mask = segment_plane_chicken(raw_chicken)

        # ---------------- transcript filtering ------------------------
        df_z = transcripts_in_plane(transcripts, z)
        df_z = crop_transcripts_to_tile(df_z, x_pix, y_pix)

        # index arrays once to avoid double work
        yy = df_z.translate_y.astype(int).values
        xx = df_z.translate_x.astype(int).values

        in_chicken = chicken_mask[yy, xx]
        in_micro   = micro_mask  [yy, xx] & ~in_chicken
        in_dapi    = dapi_mask   [yy, xx]  & micro_mask[yy, xx]  # has to be in dapi and in the IBA1 stain prevents bleed or out of focus.

        cyto_rows  = df_z[in_micro & ~in_dapi].copy()
        nuc_rows   = df_z[in_dapi].copy() # this is fine because we are grabbing the nucleus that our baysor segmentation tracked towards

        cyto_rows["compartment"] = "cyto"
        nuc_rows ["compartment"] = "nuc"

        rows_cyto.append(cyto_rows)
        rows_nuc .append(nuc_rows)
        
        #cyto_df = pd.concat(rows_cyto, ignore_index=True)
        #nuc_df  = pd.concat(rows_nuc , ignore_index=True)
        #tot_df  = pd.concat([cyto_df, nuc_df], ignore_index=True)
        
    cyto_df = (pd.concat(rows_cyto, ignore_index=True)
               if rows_cyto else pd.DataFrame(columns=transcripts.columns.tolist()
                                                          + ["compartment"]))
    nuc_df  = (pd.concat(rows_nuc , ignore_index=True)
               if rows_nuc  else pd.DataFrame(columns=transcripts.columns.tolist()
                                                          + ["compartment"]))
    tot_df  = pd.concat([cyto_df, nuc_df], ignore_index=True)
    
    #if cyto_df.empty and nuc_df.empty:
    #    print(f"[warn] {batch} cell {cellname}: no valid planes or spots")
        
        #write_debug_png("test and string doesnt matter",
        #                raw_rabbit, micro_mask, dapi_mask, chicken_mask, df_z)
    return (cyto_df,
            nuc_df,
            tot_df)


# warming the cache to prevent I/O during the experiment:

def _warm_cache(batch: str, z_vals, root: str):
    channels = ("Anti-Rabbit", "Anti-Chicken")
    for z in z_vals:
        for ch in channels:
            p = f"{root}/{batch}/images/mosaic_{ch}_z{z}.tif"
            if os.path.exists(p):
                Mapping.load_tiff_image(p)   # first call → disk, later calls → RAM
    # single-plane helpers
    Mapping.load_tiff_image(f"{root}/{batch}/binary_image.tif")
    Mapping.load_tiff_image(f"{root}/{batch}/images/mosaic_DAPI_z3.tif")

# ----------------------------------------------------------------------
#  Driver for *one* experiment
# ----------------------------------------------------------------------
def run_experiment(exp_path: Path,
                   ad_parent: "sc.AnnData",
                   outdir: Path,
                   root_spatial: Path = Path("/hpc/projects/group.quake/doug/Shapes_Spatial")):
    """
    Parameters
    ----------
    exp_path : Path
        Folder that contains `images/` and `baysor/` for ONE sample.
        e.g. /hpc/projects/.../3-mo-male-1/
    ad_parent : AnnData
        Pre-loaded AnnData filtered to microglia only (Shape_500.h5ad).
    outdir : Path
        Where CSVs will be written.
    root_spatial : Path
        Root of the slide folder hierarchy (defaults to Doug's path).
    """
    exp_path   = str(exp_path)                      
    root_str   = str(root_spatial) + '/'                
    batch      = exp_path.rstrip("/").split("/")[-1]
    print(f"[{batch}]   starting")

    ad_viz = ad_parent[ad_parent.obs.batchID == batch]

    # 1 · affine transform ------------------------------------------------
    tmat_file = f"{exp_path}/images/micron_to_mosaic_pixel_transform.csv"
    tmat      = pd.read_table(tmat_file, sep=" ", header=None).iloc[:2]

    # 2 · transcripts & blank list ----------------------------------------
    tx        = find_filtered_transcripts(exp_path+'/')
    print(exp_path)
    print(tx)
    print(tx.columns.tolist())
    tx["mosaic_x"] = tx.global_x * tmat.iloc[0,0] + tmat.iloc[0,2]
    tx["mosaic_y"] = tx.global_y * tmat.iloc[1,1] + tmat.iloc[1,2]
    blank_names    = tx[tx.gene.str.contains("Blank")].gene.unique().tolist()
    
    # warming the cache so that the loop runs very quickly.
    z_vals = tx.global_z.unique().astype(int)
    _warm_cache(batch, z_vals, root_str.rstrip("/"))
    print(f"[{batch}]  cache warm-up done ({len(z_vals)} z-planes)")

    # 3 · images that are needed *once* per experiment --------------------
    print("before max_proj")
    raw_im   = Mapping.load_tiff_image(f"{root_str}/{batch}/binary_image.tif")
    raw_dapi = Mapping.load_tiff_image(f"{root_str}/{batch}/images/mosaic_DAPI_z3.tif")
    print("after max_proj")

    # 4 · results containers ---------------------------------------------
    cols = ad_viz.var_names.tolist() + blank_names
    df_cyto  = pd.DataFrame(columns=cols)
    df_nuc   = pd.DataFrame(columns=cols)
    df_tot   = pd.DataFrame(columns=cols)
    df_spots = pd.DataFrame(columns=['x','y','gene','cell','compartment'])

    # 5 · iterate microglia ----------------------------------------------
    for idx in tqdm(range(len(ad_viz)), desc=f"[{batch}]  cells"):

        ad_cell  = ad_viz[idx, :]
        cellname = ad_cell.obs.Name.iloc[0]

        # 5a · crop tile --------------------------------------------------
        tile_raw, tile_dapi, tile_tx, _ = load_images(
            batch, ad_cell.obs.x.iloc[0], ad_cell.obs.y.iloc[0],
            raw_im, raw_dapi, tx)

        # 5b · segment max-proj masks ------------------------------------
        filled_raw  = segment_image(tile_raw,  205, foreground=True)
        filled_dapi = segment_image(tile_dapi, 255, foreground=True, dapi=True)
        micro_1     = roi_picker(filled_raw)
        dapi_1      = roi_picker(filled_dapi, dapi=True)

        # 5c · plane-by-plane counts -------------------------------------
        x_pix = int(round(ad_cell.obs.x.iloc[0] * tmat.values[0,0] + tmat.values[0,2]))
        y_pix = int(round(ad_cell.obs.y.iloc[0] * tmat.values[1,1] + tmat.values[1,2]))

        cyto_df, nuc_df, tot_df = count_by_plane(
            batch, tile_tx, micro_1.astype(bool),
            root_str, x_pix, y_pix)

        for df in (cyto_df, nuc_df, tot_df):
            df["cell"] = cellname

        # 5d · gene-count matrices --------------------------------------
        df_cyto = pd.concat([df_cyto, rename_index(cyto_df, ad_cell, blank_names)])
        df_nuc  = pd.concat([df_nuc,  rename_index(nuc_df , ad_cell, blank_names)])
        df_tot  = pd.concat([df_tot,  rename_index(tot_df , ad_cell, blank_names)])
        df_spots= pd.concat([df_spots, tot_df])   # every spot

    # 6 · write output ----------------------------------------------------
    outdir.mkdir(parents=True, exist_ok=True)
    df_cyto.to_csv(outdir / f"{batch}_non_nuc.csv")
    df_nuc .to_csv(outdir / f"{batch}_nuc.csv")
    df_tot .to_csv(outdir / f"{batch}_nuc_y_non_nuc.csv")
    df_spots.to_csv(outdir / f"{batch}_complete.csv")
    print(f"[{batch}]   done   ({len(ad_viz)} cells)")


# ----------------------------------------------------------------------
#  Command-line interface
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Slice-by-slice transcript counter")
    p.add_argument("--exp",      required=True, type=Path,
                   help="Path to one experiment folder (…/3-mo-male-1/)")
    p.add_argument("--h5",       required=True, type=Path,
                   help="Shape_500.h5ad (will be loaded & filtered to microglia)")
    p.add_argument("--outdir",   default=Path("transcript_out_slice_by_slice_v2"),
                   type=Path, help="Output directory")
    return p.parse_args()


def main():
    args = parse_args()

    # load AnnData once for this worker
    ad = sc.read_h5ad(args.h5)
    # make sure we are just grabbing microglia
    ad = ad[ad.obs.updated_celltype == "Microglia"]

    run_experiment(args.exp.resolve(),
                   ad_parent=ad,
                   outdir=args.outdir.resolve())

if __name__ == "__main__":
    main()