# The permutation in this code is adapted from: https://github.com/xingjiepan/MERFISH_analysis/blob/main/cell_cell_contact/permutation.py

import os
import sys
from tqdm import tqdm
import scanpy as sc
import pandas as pd
import string
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from tqdm import tqdm
import Mapping
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial
import skimage
import cv2
from skimage.morphology import disk, opening, closing
from scipy.ndimage import binary_fill_holes, label, distance_transform_edt
from skimage.segmentation import find_boundaries, watershed
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.segmentation import watershed
import scipy.spatial as spatial
from scipy.sparse import csr_matrix

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
    
    #print(f'load {batchID}')
    #raw_im = Mapping.load_tiff_image(root + batchID + '/binary_image.tif')
    #dapi_im = Mapping.load_tiff_image(root + batchID + '/images/mosaic_DAPI_z3.tif')
    
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

def rename_index(df, adata,transcript_df):
    counts_matrix_result = generate_counts_matrix(df,  transcript_df.gene.unique().tolist())
    counts_matrix_result.index = adata.obs.index
    return counts_matrix_result

def generate_transcript_spreadsheet(transcripts, dapi, micro, ad_test):
    # Calculate the union of dapi_1 and micro_1
    union_mask = np.logical_or(dapi.astype(bool), micro.astype(bool))  
    

    # Filter transcripts to include only those within the union of dapi_1 and micro_1
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

def generate_gene_coloc_matrices(df, counts_matrix_full, micro_1, gene_col, coord_cols, genes, 
                                 permutation_method, transform_matrix, N_permutations=1, contact_radius=1):
    """
    Generate a cell type contact matrix.
    """
    assert(permutation_method in ['no_permutation', 'global_permutation'])

    points = np.array(df[coord_cols])
    point_tree = spatial.cKDTree(points)
    neighbors_contact = point_tree.query_ball_point(points, contact_radius)

    N_genes = len(genes)
    gene_ids = list(range(N_genes))
    gene_id_map = pd.DataFrame({'gene_id': gene_ids, 'gene': genes})
    barcode_ids_of_genes = list(df[[gene_col]].merge(gene_id_map, left_on=gene_col, right_on='gene', how='left')['gene_id'])

    if permutation_method == 'no_permutation':
        return count_gene_coloc(neighbors_contact, barcode_ids_of_genes, N_genes)

    elif permutation_method == 'global_permutation':
        permuted_contact_tensor = np.zeros((N_permutations, N_genes, N_genes))
        
        for i in range(N_permutations):
            try:
                permutated_cell_counts = generate_transcript_positions(counts_matrix_full, micro_1, n_iterations=1)
                
                permutated_cell_counts['x'] = permutated_cell_counts['x'] / transform_matrix.iloc[0, 0]
                permutated_cell_counts['y'] = permutated_cell_counts['y'] / transform_matrix.iloc[1, 1]

                barcode_ids_of_genes_perm = list(permutated_cell_counts[[gene_col]].merge(
                    gene_id_map, left_on=gene_col, right_on='gene', how='left')['gene_id'])

                points_perm = np.array(permutated_cell_counts[['x', 'y']])
                point_tree_perm = spatial.cKDTree(points_perm)
                neighbors_contact_perm = point_tree_perm.query_ball_point(points_perm, contact_radius)
                
                permuted_ct_contact_count_mtx = count_gene_coloc(
                    neighbors_contact_perm, barcode_ids_of_genes_perm, N_genes
                )
                
                permuted_contact_tensor[i] = permuted_ct_contact_count_mtx

            except ValueError as e:
                print(f"Error in permutation iteration {i}: {e}. Skipping this permutation.")
                continue

        return permuted_contact_tensor

    
# the adaptations this code needs is to have the gene ids of each of the transcripts
def count_gene_coloc(neighbors, barcode_ids_of_genes, N_genes, permutated_gene_ids=None,
                             use_sparse_matrix=False):
    '''Count the contacts between genes.
    Return a sparse matrix of the number of contacts between
    pairs of genes.
    '''
    # If no permutation is specified, use identical permutation
    if None is permutated_gene_ids:
        permutated_gene_ids = list(range(len(neighbors)))
    
    # Initialize the counting matrix
    if use_sparse_matrix:
        ct_contact_count_mtx = csr_matrix((N_genes, N_genes), dtype=int)
    else:
        ct_contact_count_mtx = np.zeros((N_genes, N_genes), dtype=int)
    
    # Count the contacts
    for i in range(len(neighbors)):
        p_i = permutated_gene_ids[i]
        ct_i = barcode_ids_of_genes[p_i]
        
        for j in neighbors[i]:
            if i != j: # Ignore the cell itself
                p_j = permutated_gene_ids[j]
                ct_j = barcode_ids_of_genes[p_j]
            
                ct_contact_count_mtx[ct_i, ct_j] += 1
            
    return ct_contact_count_mtx

def generate_transcript_positions(counts_matrix_full, micro_1, n_iterations=1):
    """
    Simulates transcript positions for each gene in the count matrix by randomly placing points
    within the positive regions of micro_1. Returns a DataFrame containing the gene name and 
    the x and y coordinates for each simulated transcript.

    Parameters:
        counts_matrix_full (pd.DataFrame): DataFrame with columns representing genes and values 
                                           indicating transcript counts for each gene.
        micro_1 (np.array): Binary mask array for the valid region.
        n_iterations (int): Number of bootstrap iterations to perform. Default is 1.

    Returns:
        pd.DataFrame: A DataFrame containing 'gene', 'x', and 'y' columns for each simulated transcript.
    """
    # Get all valid positions within the micro_1 mask where dots can be placed
    valid_positions = np.column_stack(np.where(micro_1 > 0))
    
    if valid_positions.size == 0:
        raise ValueError(f"No valid positions available in micro_1 mask for batch.")

    results = []
    
    for _ in range(n_iterations):
        for gene in counts_matrix_full.columns:
            transcript_count = int(counts_matrix_full[gene].iloc[0])  # Get count of transcripts for this gene
            
            if transcript_count > 0:
                random_indices = np.random.choice(len(valid_positions), transcript_count, replace=True)
                selected_positions = valid_positions[random_indices]
                for pos in selected_positions:
                    y, x = pos
                    results.append({'gene': gene, 'x': x, 'y': y})

    if not results:
        raise ValueError(f"No transcripts generated for batch due to all-zero counts in counts_matrix_full.")
    
    return pd.DataFrame(results)


def process_experiment(experiment):
    batch = experiment.split('/')[-2]
    
    ad_parent = sc.read_h5ad('Shape_500.h5ad')
    ad_parent = ad_parent[ad_parent.obs.updated_celltype == 'Microglia']
    
    place_in_order = {
        0: '2',
        1: '3',
        2: '1',
        3: '0',
        4: '4',
    }

    ad_parent.obs['ordered_morph'] = ad_parent.obs.morph_leiden.map(place_in_order)
    ad_viz = ad_parent[ad_parent.obs.batchID == batch]
    
    transform_file = f'{experiment}/images/micron_to_mosaic_pixel_transform.csv'
    transform_matrix = pd.read_table(transform_file, sep=' ', header=None).iloc[:2]
    transcripts = find_filtered_transcripts(experiment)
    transcripts['mosaic_x'], transcripts['mosaic_y'] = (
        transcripts.global_x * transform_matrix.iloc[0, 0] + transform_matrix.iloc[0, 2],
        transcripts.global_y * transform_matrix.iloc[1, 1] + transform_matrix.iloc[1, 2]
    )
    blank_names = transcripts[transcripts.gene.str.contains('Blank')].gene.unique().tolist()

    root = '/hpc/projects/group.quake/doug/Shapes_Spatial/'
    raw_im = Mapping.load_tiff_image(root + batch + '/binary_image.tif')
    raw_dapi = Mapping.load_tiff_image(root + batch + '/images/mosaic_DAPI_z3.tif')
    
    gene_col = 'gene'
    genes = np.unique(transcripts.gene.unique().tolist())
    N_genes = len(genes)
    N_permutations = 1000

    # Iterate through each ordered_morph classification
    for morph_class in ad_viz.obs['ordered_morph'].unique():
        print(f"Processing morph class {morph_class} in batch {batch}")
        
        ad_viz_morph = ad_viz[ad_viz.obs['ordered_morph'] == morph_class]
        
        merged_soma_coloc_counts_no_perm = np.zeros((N_genes, N_genes), dtype=int)
        merged_soma_coloc_counts_perm = np.zeros((N_permutations, N_genes, N_genes), dtype=int)
        
        merged_branches_coloc_counts_no_perm = np.zeros((N_genes, N_genes), dtype=int)
        merged_branches_coloc_counts_perm = np.zeros((N_permutations, N_genes, N_genes), dtype=int)
        
        merged_total_coloc_counts_no_perm = np.zeros((N_genes, N_genes), dtype=int)
        merged_total_coloc_counts_perm = np.zeros((N_permutations, N_genes, N_genes), dtype=int)
        
        for i in tqdm(range(len(ad_viz_morph)), desc=f"Processing {batch} - Morph {morph_class}"):
            ad_test = ad_viz_morph[i, :]
            small_raw, small_dapi, small_transcripts, image_loc = load_images(
                batch, ad_test.obs.x.iloc[0], ad_test.obs.y.iloc[0],
                raw_im, raw_dapi, transcripts
            )

            filled_raw = segment_image(small_raw, 205, foreground=True)
            filled_dapi = segment_image(small_dapi, 255, foreground=True, dapi=True)

            micro_1 = roi_picker(filled_raw)
            dapi_1 = roi_picker(filled_dapi, dapi=True)
            non_dapi_1 = np.logical_and(micro_1.astype(bool), np.logical_not(filled_dapi.astype(bool))).astype(np.uint8)
            total_1 = np.logical_or(dapi_1.astype(bool), micro_1.astype(bool))

            counts_non_nuclei, counts_nuclei = count_gene_overlaps(
                small_transcripts, dapi_1, micro_1, filled_dapi
            )
            counts_non_nuclei['x'] = counts_non_nuclei['translate_x']
            counts_non_nuclei['y'] = counts_non_nuclei['translate_y']
            
            counts_nuclei['x'] = counts_nuclei['translate_x']
            counts_nuclei['y'] = counts_nuclei['translate_y']
            
            counts_matrix_result_branch = rename_index(counts_non_nuclei,ad_test,transcripts)
            counts_matrix_result_dapi = rename_index(counts_nuclei,ad_test,transcripts)

            sub_full_trans = generate_transcript_spreadsheet(small_transcripts, dapi_1, micro_1, ad_test)
            counts_matrix_full = rename_index(sub_full_trans, ad_test, transcripts)

            sub_full_trans['x'] = sub_full_trans['x'] / transform_matrix.iloc[0, 0]
            sub_full_trans['y'] = sub_full_trans['y'] / transform_matrix.iloc[1, 1]
            
            counts_non_nuclei['x'] = counts_non_nuclei['x'] / transform_matrix.iloc[0, 0]
            counts_non_nuclei['y'] = counts_non_nuclei['y'] / transform_matrix.iloc[1, 1]
            
            counts_nuclei['x'] = counts_nuclei['x'] / transform_matrix.iloc[0, 0]
            counts_nuclei['y'] = counts_nuclei['y'] / transform_matrix.iloc[1, 1]
            
            # soma counts
            gene_soma_coloc_counts_no_perm = generate_gene_coloc_matrices(
                counts_nuclei, counts_matrix_result_dapi, dapi_1, gene_col, 
                ['x', 'y'], genes, 'no_permutation', transform_matrix
            )
            gene_soma_coloc_counts_perm = generate_gene_coloc_matrices(
                counts_nuclei, counts_matrix_result_dapi, dapi_1, gene_col, 
                ['x', 'y'], genes, 'global_permutation', transform_matrix, 
                N_permutations=N_permutations
            )
            
            merged_soma_coloc_counts_no_perm += gene_soma_coloc_counts_no_perm
            merged_soma_coloc_counts_perm += gene_soma_coloc_counts_perm.astype(int)
            
            # branch counts
            gene_branches_coloc_counts_no_perm = generate_gene_coloc_matrices(
                counts_non_nuclei, counts_matrix_result_branch, non_dapi_1, gene_col, 
                ['x', 'y'], genes, 'no_permutation', transform_matrix
            )
            gene_branches_coloc_counts_perm = generate_gene_coloc_matrices(
                counts_non_nuclei, counts_matrix_result_branch, non_dapi_1, gene_col, 
                ['x', 'y'], genes, 'global_permutation', transform_matrix, 
                N_permutations=N_permutations
            )
            
            merged_branches_coloc_counts_no_perm += gene_branches_coloc_counts_no_perm
            merged_branches_coloc_counts_perm += gene_branches_coloc_counts_perm.astype(int)
            
            # total counts
            gene_total_coloc_counts_no_perm = generate_gene_coloc_matrices(
                sub_full_trans, counts_matrix_full, total_1, gene_col, 
                ['x', 'y'], genes, 'no_permutation', transform_matrix
            )
            gene_total_coloc_counts_perm = generate_gene_coloc_matrices(
                sub_full_trans, counts_matrix_full, total_1, gene_col, 
                ['x', 'y'], genes, 'global_permutation', transform_matrix, 
                N_permutations=N_permutations
            )
            
            merged_total_coloc_counts_no_perm += gene_total_coloc_counts_no_perm
            merged_total_coloc_counts_perm += gene_total_coloc_counts_perm.astype(int)
            
        merged_soma_coloc_counts_perm_sum = np.sum(merged_soma_coloc_counts_perm, axis=0).astype(int)
        means_soma = np.mean(merged_soma_coloc_counts_perm, axis=0)
        stds_soma = np.std(merged_soma_coloc_counts_perm, axis=0)
        
        merged_branches_coloc_counts_perm_sum = np.sum(merged_branches_coloc_counts_perm, axis=0).astype(int)
        means_branches = np.mean(merged_branches_coloc_counts_perm, axis=0)
        stds_branches = np.std(merged_branches_coloc_counts_perm, axis=0)
        
        merged_total_coloc_counts_perm_sum = np.sum(merged_total_coloc_counts_perm, axis=0).astype(int)
        means_total = np.mean(merged_total_coloc_counts_perm, axis=0)
        stds_total = np.std(merged_total_coloc_counts_perm, axis=0)
        # Save outputs for each morph class separately
        morph_output_dir = f'permutation_coloc/{batch}/morph_{morph_class}'
        os.makedirs(morph_output_dir, exist_ok=True)
        
        output_file_soma_no_perm = os.path.join(morph_output_dir, 'soma_no_permutation.npy')
        output_file_soma_perm = os.path.join(morph_output_dir, 'soma_full_permutation.npy')
        output_file_soma_mean = os.path.join(morph_output_dir, f'soma_full_permutation_mean.npy')
        output_file_soma_std = os.path.join(morph_output_dir, f'soma_full_permutation_std.npy')
        
        output_file_branches_no_perm = os.path.join(morph_output_dir, 'branches_no_permutation.npy')
        output_file_branches_perm = os.path.join(morph_output_dir, 'branches_full_permutation.npy')
        output_file_branches_mean = os.path.join(morph_output_dir, f'branches_full_permutation_mean.npy')
        output_file_branches_std = os.path.join(morph_output_dir, f'branches_full_permutation_std.npy')
        
        output_file_total_no_perm = os.path.join(morph_output_dir, 'total_no_permutation.npy')
        output_file_total_perm = os.path.join(morph_output_dir, 'total_full_permutation.npy')
        output_file_total_mean = os.path.join(morph_output_dir, f'total_full_permutation_mean.npy')
        output_file_total_std = os.path.join(morph_output_dir, f'total_full_permutation_std.npy')
        
        np.save(output_file_soma_no_perm, merged_soma_coloc_counts_no_perm)
        np.save(output_file_soma_perm, merged_soma_coloc_counts_perm_sum)
        np.save(output_file_soma_mean, means_soma)
        np.save(output_file_soma_std, stds_soma)
        
        np.save(output_file_branches_no_perm, merged_branches_coloc_counts_no_perm)
        np.save(output_file_branches_perm, merged_branches_coloc_counts_perm_sum)
        np.save(output_file_branches_mean, means_branches)
        np.save(output_file_branches_std, stds_branches)
        
        np.save(output_file_total_no_perm, merged_total_coloc_counts_no_perm)
        np.save(output_file_total_perm, merged_total_coloc_counts_perm_sum)
        np.save(output_file_total_mean, means_total)
        np.save(output_file_total_std, stds_total)

if __name__ == '__main__':
    experiment_paths = sys.argv[1:]
    for experiment in experiment_paths:
        process_experiment(experiment)