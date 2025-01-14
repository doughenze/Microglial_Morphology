import os
import argparse
import numpy as np
import tifffile as tiff
from glob import glob
from tqdm import tqdm
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, binary_opening, binary_closing, disk
import cv2
from skimage import measure, morphology, exposure

def load_tiff_files_in_tiles(directory, stain_name, tile_size, min_size):
    """
    Load .tif files in tiles and create a max projection image.
    """
    file_pattern = os.path.join(directory, f"*{stain_name}*.tif")
    tiff_files = sorted(glob(file_pattern))
    
    with tiff.TiffFile(tiff_files[0]) as tif:
        image_shape = tif.pages[0].shape
    
    raw_image = np.zeros(image_shape, dtype=np.uint8)
    labeled_image = np.zeros(image_shape, dtype=np.int32)
    num_tiles_x = (image_shape[0] + tile_size - 1) // tile_size
    num_tiles_y = (image_shape[1] + tile_size - 1) // tile_size
    total_tiles = num_tiles_x * num_tiles_y
    current_label = 1
    
    with tqdm(total=total_tiles, desc="Processing tiles") as pbar:
        for tile_start_x in range(0, image_shape[0], tile_size):
            for tile_start_y in range(0, image_shape[1], tile_size):
                tile_end_x = min(tile_start_x + tile_size, image_shape[0])
                tile_end_y = min(tile_start_y + tile_size, image_shape[1])
                
                max_tile = None
                
                for file in tiff_files:
                    with tiff.TiffFile(file) as tif:
                        img = tif.asarray()[tile_start_x:tile_end_x, tile_start_y:tile_end_y]
                    
                    if max_tile is None:
                        max_tile = img
                    else:
                        max_tile = np.maximum(max_tile, img)
                
                # Enhance and segment the tile
                enhanced_tile = enhance_image(max_tile)
                background_subtracted_tile = subtract_background(enhanced_tile)
                edges_tile = edge_detection(background_subtracted_tile)
                segmented_tile = otsu_thresholding(edges_tile)
                filled_tile = segmented_tile.copy()
                
                raw_image[tile_start_x:tile_end_x, tile_start_y:tile_end_y] = np.maximum(
                    raw_image[tile_start_x:tile_end_x, tile_start_y:tile_end_y],
                    enhanced_tile
                )
                
                # Label the tile
                labeled_tile, num_features = measure.label(filled_tile, return_num=True, connectivity=2)
                
                # Relabel the tile with unique labels
                labeled_tile = labeled_tile + current_label
                labeled_tile[labeled_tile == current_label] = 0  # Set the background to 0
                labeled_image[tile_start_x:tile_end_x, tile_start_y:tile_end_y] = np.maximum(
                    labeled_image[tile_start_x:tile_end_x, tile_start_y:tile_end_y],
                    labeled_tile
                )
                
                current_label += num_features
                
                pbar.update(1)
    labeled_image = remove_small_objects(labeled_image, min_size=min_size)
    
    return raw_image, labeled_image

def enhance_image(image):
    """
    Enhance the contrast of the image using histogram equalization.
    """
    enhanced_image = exposure.equalize_adapthist(image, clip_limit=0.01)
    enhanced_image = (enhanced_image * 255).astype(np.uint8)
    return enhanced_image

def subtract_background(image):
    """
    Subtract the background using background subtraction techniques in OpenCV.
    """
    backSub = cv2.createBackgroundSubtractorMOG2()
    fg_mask = backSub.apply(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    subtracted_image = cv2.bitwise_and(image, image, mask=fg_mask)
    return subtracted_image

def edge_detection(image):
    """
    Apply edge detection using the Sobel operator.
    """
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return sobel

def otsu_thresholding(image):
    """
    Apply Otsu's thresholding method to segment the image.
    """
    blurred_image = cv2.GaussianBlur(image, (51, 51), 0)
    _, thresholded_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresholded_image

def fill_holes(image):
    """
    Fill small holes in the binary image using morphological operations.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)  
    return closed_image

def save_images(output_dir, original_image, labeled_image):
    """
    Save the original and labeled images as .tif files.
    """
    os.makedirs(output_dir, exist_ok=True)
    original_image_path = os.path.join(output_dir, "binary_image.tif")
    labeled_image_path = os.path.join(output_dir, "labeled_image.tif")
    
    tiff.imwrite(original_image_path, original_image.astype(np.uint8))
    tiff.imwrite(labeled_image_path, labeled_image.astype(np.uint32))  # Save labeled image as uint32 to handle large labels

def main():
    parser = argparse.ArgumentParser(description="Process and label .tif images.")
    parser.add_argument("directory", type=str, help="Directory containing the .tif files.")
    parser.add_argument("stain_name", type=str, help="Stain name to filter the .tif files.")
    parser.add_argument("tile_size", type=int, help="Tile size for processing the images.")
    parser.add_argument("output_dir", type=str, help="Directory to save the output images.")
    parser.add_argument("--min_size", type=int, default=100, help="Minimum size of cells to keep.")
    
    args = parser.parse_args()
    
    max_projection, labeled_image = load_tiff_files_in_tiles(args.directory, args.stain_name, args.tile_size, args.min_size)
    
    save_images(args.output_dir, max_projection, labeled_image)

if __name__ == "__main__":
    main()