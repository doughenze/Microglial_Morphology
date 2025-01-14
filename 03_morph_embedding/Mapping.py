import os
import numpy as np
import pandas as pd
import math
import tifffile
import geopandas as gpd
import shapely.geometry as sg
from rasterio.features import rasterize
from skimage.measure import label
from tqdm import tqdm

# Function to load TIFF images
def load_tiff_image(file_path):
    image = tifffile.imread(file_path)
    return image

# Function to process geometries and extract pixel values from label image
def extract_label_pixel_values(boundaries, label_image):
    data = []
    
    progress_bar = tqdm(total=len(boundaries), desc="Extracting pixel values")
    
    for idx, row in boundaries.iterrows():
        geom = row['Geometry']
        entity_id = row['EntityID']
        
        if not geom.is_empty and geom.geom_type == 'MultiPolygon':
            for polygon in geom.geoms:
                if not polygon.is_empty and polygon.geom_type == 'Polygon':
                    # Extract polygon coordinates
                    xx, yy = polygon.exterior.coords.xy
                    min_x, min_y = polygon.bounds[0], polygon.bounds[1]

                    # Calculate bounding box coordinates
                    xs = (np.min(xx), np.max(xx))
                    ys = (np.min(yy), np.max(yy))

                    # Check for negative coordinates
                    total = np.concatenate([xx, yy])
                    
                    if np.any(total <= 0):
                        img = np.array([[0, 0], [0, 0]])
                    elif np.shape(label_image[math.floor(ys[0]):math.ceil(ys[1]), math.floor(xs[0]):math.ceil(xs[1])])[1] == 0:
                        img = np.array([[0, 0], [0, 0]])
                    else:
                        adjusted_coords = [(x - min_x, y - min_y) for x, y in zip(xx, yy)]
                        adjusted_polygon = sg.Polygon(adjusted_coords)

                        img = label_image[math.floor(ys[0]):math.ceil(ys[1]), math.floor(xs[0]):math.ceil(xs[1])]
                        
                        
                    
                    # Append to data list
                    data.append({
                        'EntityID': entity_id,
                        'Label_pixels': np.unique(img).tolist()
                    })
        
        progress_bar.update(1)
    
    progress_bar.close()
    
    return data

# Function to process geometries and extract pixel values from label image
def extract_label_pixel_values_baysor(boundaries, label_image):
    
    data = []
    
    image_height = label_image.shape[0]
    
    progress_bar = tqdm(total=len(boundaries), desc="Extracting pixel values")
    
    for idx, row in boundaries.iterrows():
        geom = row['Geometry']
        entity_id = row['Name']
        

        if not geom.is_empty and geom.geom_type == 'Polygon':
                    # Extract polygon coordinates
            xx, yy = geom.exterior.coords.xy
            min_x, min_y = geom.bounds[0], geom.bounds[1]
            
            #flip y because Baysor coordinates are cartesian
            #yy = [image_height - y for y in yy]
            #min_y = image_height - min_y

                    # Calculate bounding box coordinates
            xs = (np.min(xx), np.max(xx))
            ys = (np.min(yy), np.max(yy))

                    # Check for negative coordinates
            total = np.concatenate([xx, yy])
                    
            if np.any(total <= 0):
                img = np.array([[0, 0], [0, 0]])
            elif np.shape(label_image[math.floor(ys[0]):math.ceil(ys[1]), math.floor(xs[0]):math.ceil(xs[1])])[1] == 0:
                img = np.array([[0, 0], [0, 0]])
            else:
                adjusted_coords = [(x - min_x, y - min_y) for x, y in zip(xx, yy)]
                adjusted_polygon = sg.Polygon(adjusted_coords)

                img = label_image[math.floor(ys[0]):math.ceil(ys[1]), math.floor(xs[0]):math.ceil(xs[1])]
                        
            # Calculate the total number of pixels
            total_pixels = img.size
        
            # Get unique pixel values and their counts
            unique_values, counts = np.unique(img, return_counts=True)
        
            # Calculate the percentage for each unique pixel value
            percentages = ((counts / total_pixels) * 100)
                    
                    # Append to data list
            data.append({
                'Name': entity_id,
                'Label_pixels': unique_values.tolist(),
                'Percentage': percentages.tolist()
            })
        
        progress_bar.update(1)
    
    progress_bar.close()
    
    return data


if __name__ == "__main__":
    pass