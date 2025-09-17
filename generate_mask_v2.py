import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from rtree import index
import cv2

def create_masks(image_shapefile, building_shapefile, output_dir="wms_tiles/2024_orthoHR/masks", gsd=0.08):
    """
    Generate binary masks for each image based on building footprints, using spatial indexing.
    
    Parameters:
    - image_shapefile: Path to shapefile with image footprints (e.g., 'wms_tiles/2024_orthoHR/2024_orthoHR_tiles.shp')
    - building_shapefile: Path to shapefile with building footprints (e.g., 'buildings.shp')
    - output_dir: Directory to save mask GeoTIFFs
    - gsd: Ground Sample Distance (m/pixel), default 0.08m for 2024_orthoHR
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filtered_type = ["PND", "OBW"]
    
    # Read image footprints
    print("Loading image shapefile...")
    image_gdf = gpd.read_file(image_shapefile)
    if image_gdf.crs is None or str(image_gdf.crs) != "EPSG:28992":
        print(f"Warning: Image shapefile CRS is {image_gdf.crs}. Reprojecting to EPSG:28992.")
        image_gdf = image_gdf.to_crs("EPSG:28992")

    # Read building footprints
    print("Loading building shapefile...")
    building_gdf = gpd.read_file(building_shapefile)
    
    # Filter BGT data
    if 'bgttype' in building_gdf.columns:
        print(f"Total buildings before filtering: {len(building_gdf)}")
        building_gdf = building_gdf[building_gdf['bgttype'].isin(filtered_type)]
        print(f"Total buildings after filtering: {len(building_gdf)}")
    else:
        print("Warning: 'bgttype' column not found in building shapefile. Using all features.")
    
    if building_gdf.crs is None or str(building_gdf.crs) != "EPSG:28992":
        print(f"Warning: Building shapefile CRS is {building_gdf.crs}. Reprojecting to EPSG:28992.")
        building_gdf = building_gdf.to_crs("EPSG:28992")

    # Ensure geometries are valid
    print("Validating geometries...")
    building_gdf = building_gdf[building_gdf.geometry.is_valid]
    image_gdf = image_gdf[image_gdf.geometry.is_valid]
    
    if building_gdf.empty:
        print("Error: No valid buildings found after filtering!")
        return

    # Build spatial index for buildings
    print("Building spatial index for buildings...")
    ridx = index.Index()
    for i, geom in enumerate(building_gdf.geometry):
        if geom is not None and not geom.is_empty:
            ridx.insert(i, geom.bounds)

    # Initialize progress bar and counters
    total_tiles = len(image_gdf)
    print(f"Processing {total_tiles} tiles...")
    successful_masks = 0
    failed_masks = 0
    skipped_masks = 0
    mask_paths = []

    # Process each tile
    with tqdm(total=total_tiles, desc="Creating masks", unit="tile") as pbar:
        for idx, row in image_gdf.iterrows():
            tile_name = row['tile']  # e.g., tile_i_j.jpg
            tile_geom = row['geometry']
            
            # Skip if tile geometry is invalid
            if tile_geom is None or tile_geom.is_empty or not tile_geom.is_valid:
                print(f"Skipping tile {tile_name}: invalid geometry")
                mask_paths.append(None)
                skipped_masks += 1
                pbar.update(1)
                continue
                
            tile_bounds = tile_geom.bounds  # (minx, miny, maxx, maxy)

            # Calculate tile dimensions based on GSD
            width = int(round((tile_bounds[2] - tile_bounds[0]) / gsd))
            height = int(round((tile_bounds[3] - tile_bounds[1]) / gsd))
            
            # Ensure minimum dimensions
            if width <= 0 or height <= 0:
                print(f"Skipping tile {tile_name}: invalid dimensions ({width}x{height})")
                mask_paths.append(None)
                skipped_masks += 1
                pbar.update(1)
                continue

            # Use spatial index to get candidate buildings
            candidate_ids = list(ridx.intersection(tile_bounds))
            
            if not candidate_ids:
                print(f"Skipping tile {tile_name}: no buildings intersect (spatial index)")
                mask_paths.append(None)
                skipped_masks += 1
                pbar.update(1)
                continue

            # Filter candidates with exact intersection
            candidates = building_gdf.iloc[candidate_ids]
            intersecting_buildings = candidates[candidates.intersects(tile_geom)]
            
            if intersecting_buildings.empty:
                print(f"Skipping tile {tile_name}: no buildings intersect after exact check")
                mask_paths.append(None)
                skipped_masks += 1
                pbar.update(1)
                continue

            try:
                # Create affine transform for the tile
                transform = from_bounds(*tile_bounds, width=width, height=height)
                
                # Debug: Print some information
                print(f"Processing {tile_name}: {len(intersecting_buildings)} buildings, {width}x{height} pixels")
                
                # Get geometries that actually intersect the tile
                clipped_geometries = []
                for geom in intersecting_buildings.geometry:
                    if geom.intersects(tile_geom):
                        # Optionally clip to tile bounds for better accuracy
                        try:
                            clipped_geom = geom.intersection(tile_geom)
                            if not clipped_geom.is_empty:
                                clipped_geometries.append(clipped_geom)
                        except Exception as e:
                            print(f"Warning: Error clipping geometry: {e}")
                            clipped_geometries.append(geom)
                
                if not clipped_geometries:
                    print(f"Skipping tile {tile_name}: no valid clipped geometries")
                    mask_paths.append(None)
                    skipped_masks += 1
                    pbar.update(1)
                    continue

                # Rasterize building footprints (1 for buildings, 0 for background)
                mask = rasterize(
                    [(geom, 1) for geom in clipped_geometries],
                    out_shape=(height, width),
                    transform=transform,
                    fill=0,
                    dtype=rasterio.uint8,
                    all_touched=True  # This can help capture small buildings
                )
                
                # Debug: Check if mask has any non-zero values
                non_zero_pixels = np.count_nonzero(mask)
                print(f"Mask for {tile_name}: {non_zero_pixels} non-zero pixels out of {width*height}")
                
                if non_zero_pixels == 0:
                    print(f"Warning: Generated mask for {tile_name} is completely blank!")

                # Save mask as GeoTIFF
                output_file = os.path.join(output_dir, f"mask_{tile_name}")
                
                cv2.imwrite(output_file, mask* 255)
    
                print(f"Created mask {output_file}")
                mask_paths.append(output_file)
                successful_masks += 1
                
            except Exception as e:
                print(f"Error creating mask for {tile_name}: {e}")
                import traceback
                traceback.print_exc()
                mask_paths.append(None)
                failed_masks += 1

            pbar.update(1)

    # Add mask paths to the image geodataframe
    image_gdf['mask_path'] = mask_paths
    
    # Save updated shapefile with mask paths
    output_shapefile = os.path.join(os.path.dirname(image_shapefile), "2024_orthoHR_tiles_with_masks.shp")
    image_gdf.to_file(output_shapefile)
    print(f"Saved updated shapefile with mask paths to {output_shapefile}")

    # Print summary
    print(f"\nMask Creation Summary:")
    print(f"Total tiles processed: {total_tiles}")
    print(f"Successful masks: {successful_masks}")
    print(f"Failed masks: {failed_masks}")
    print(f"Skipped masks (no buildings): {skipped_masks}")
    
    return successful_masks > 0

def debug_single_tile(image_shapefile, building_shapefile, tile_index=0, gsd=0.08):
    """
    Debug function to process a single tile and provide detailed information.
    """
    filtered_type = ["PND", "OBW"]
    
    # Load data
    image_gdf = gpd.read_file(image_shapefile)
    building_gdf = gpd.read_file(building_shapefile)
    
    # Filter and reproject as needed
    if 'bgttype' in building_gdf.columns:
        building_gdf = building_gdf[building_gdf['bgttype'].isin(filtered_type)]
    
    if image_gdf.crs != "EPSG:28992":
        image_gdf = image_gdf.to_crs("EPSG:28992")
    if building_gdf.crs != "EPSG:28992":
        building_gdf = building_gdf.to_crs("EPSG:28992")
    
    # Get single tile
    if tile_index >= len(image_gdf):
        print(f"Error: tile_index {tile_index} >= number of tiles {len(image_gdf)}")
        return
        
    tile_row = image_gdf.iloc[tile_index]
    tile_geom = tile_row['geometry']
    tile_name = tile_row['tile']
    tile_bounds = tile_geom.bounds
    
    print(f"Debugging tile: {tile_name}")
    print(f"Tile bounds: {tile_bounds}")
    print(f"Tile area: {tile_geom.area:.2f} m²")
    
    # Find intersecting buildings
    intersecting = building_gdf[building_gdf.intersects(tile_geom)]
    print(f"Intersecting buildings: {len(intersecting)}")
    
    if len(intersecting) > 0:
        print(f"Building areas: {intersecting.geometry.area.describe()}")
        
        # Calculate dimensions
        width = int(round((tile_bounds[2] - tile_bounds[0]) / gsd))
        height = int(round((tile_bounds[3] - tile_bounds[1]) / gsd))
        print(f"Mask dimensions: {width}x{height}")
        
        # Create transform
        transform = from_bounds(*tile_bounds, width=width, height=height)
        print(f"Transform: {transform}")
        
        # Try rasterization
        mask = rasterize(
            [(geom, 1) for geom in intersecting.geometry],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=rasterio.uint8,
            all_touched=True
        )
        
        print(f"Mask stats: min={mask.min()}, max={mask.max()}, non-zero pixels={np.count_nonzero(mask)}")
        
        # Save debug mask
        debug_path = f"debug_mask_{tile_name.replace('.jpg', '.tif')}"
        with rasterio.open(
            debug_path, 'w', driver='GTiff', height=height, width=width,
            count=1, dtype=rasterio.uint8, crs='EPSG:28992', transform=transform
        ) as dst:
            dst.write(mask, 1)
        print(f"Debug mask saved to: {debug_path}")
    
    return intersecting

# Example usage
if __name__ == "__main__":
    image_shapefile = "wms_tiles/2024_orthoHR/2024_orthoHR_tiles.shp"
    building_shapefile = r"C:\Users\HDSL36\Desktop\Work\Projects\Change_detection\output\BGT\bgt_data.shp"
    
    # First, try debugging a single tile
    print("=== DEBUG MODE ===")
    # debug_single_tile(image_shapefile, building_shapefile, tile_index=0, gsd=0.08)
    
    print("\n=== FULL PROCESSING ===")
    success = create_masks(
        image_shapefile=image_shapefile,
        building_shapefile=building_shapefile,
        output_dir="wms_tiles/2024_orthoHR/masks",
        gsd=0.08
    )
    
    if success:
        print("Mask generation completed successfully!")
    else:
        print("Mask generation failed - check the debug output above.")