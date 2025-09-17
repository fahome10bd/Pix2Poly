import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from rtree import index

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
    image_gdf = gpd.read_file(image_shapefile)
    if image_gdf.crs is None or str(image_gdf.crs) != "EPSG:28992":
        print(f"Warning: Image shapefile CRS is {image_gdf.crs}. Reprojecting to EPSG:28992.")
        image_gdf = image_gdf.to_crs("EPSG:28992")

    # Read building footprints
    building_gdf = gpd.read_file(building_shapefile)
    # Filtering bgt
    building_gdf = building_gdf[building_gdf['bgttype'].isin(filtered_type)]
    
    if building_gdf.crs is None or str(building_gdf.crs) != "EPSG:28992":
        print(f"Warning: Building shapefile CRS is {building_gdf.crs}. Reprojecting to EPSG:28992.")
        building_gdf = building_gdf.to_crs("EPSG:28992")

    # Build spatial index for buildings
    print("Building spatial index for buildings...")
    ridx = index.Index()
    for i, geom in enumerate(building_gdf.geometry):
        ridx.insert(i, geom.bounds)

    # Initialize progress bar and counters
    total_tiles = len(image_gdf)
    print(f"Processing {total_tiles} tiles...")
    successful_masks = 0
    failed_masks = 0
    skipped_masks = 0

    # Process each tile
    with tqdm(total=total_tiles, desc="Creating masks", unit="tile") as pbar:
        for idx, row in image_gdf.iterrows():
            tile_name = row['tile']  # e.g., tile_i_j.jpg
            tile_geom = row['geometry']
            tile_bounds = tile_geom.bounds  # (minx, miny, maxx, maxy)

            # Calculate tile dimensions based on GSD
            width = int((tile_bounds[2] - tile_bounds[0]) / gsd)
            height = int((tile_bounds[3] - tile_bounds[1]) / gsd)

            # Use spatial index to get candidate buildings
            candidate_ids = list(ridx.intersection(tile_bounds))
            if not candidate_ids:
                print(f"Skipping tile {tile_name}: no buildings intersect")
                skipped_masks += 1
                pbar.update(1)
                continue

            # Filter candidates with exact intersection
            intersecting_buildings = building_gdf.iloc[candidate_ids][
                building_gdf.iloc[candidate_ids].intersects(tile_geom)
            ]
            if intersecting_buildings.empty:
                print(f"Skipping tile {tile_name}: no buildings intersect after exact check")
                skipped_masks += 1
                pbar.update(1)
                continue

            try:
                # Create affine transform for the tile
                transform = from_bounds(*tile_bounds, width=width, height=height)

                # Rasterize building footprints (1 for buildings, 0 for background)
                mask = rasterize(
                    [(geom, 1) for geom in intersecting_buildings.geometry],
                    out_shape=(height, width),
                    transform=transform,
                    fill=0,
                    dtype=rasterio.uint8
                )

                # Save mask as GeoTIFF
                output_file = os.path.join(output_dir, f"mask_{tile_name.replace('.jpg', '.tif')}")
                with rasterio.open(
                    output_file,
                    'w',
                    driver='GTiff',
                    height=height,
                    width=width,
                    count=1,
                    dtype=rasterio.uint8,
                    crs='EPSG:28992',
                    transform=transform
                ) as dst:
                    dst.write(mask, 1)

                print(f"Created mask {output_file}")
                successful_masks += 1
            except Exception as e:
                print(f"Error creating mask for {tile_name}: {e}")
                failed_masks += 1

            pbar.update(1)

    # Save updated shapefile with mask paths
    image_gdf['mask_path'] = [
        os.path.join(output_dir, f"mask_{tile_name.replace('.jpg', '.tif')}") if not building_gdf.iloc[list(idx.intersection(geom.bounds))].empty else None
        for tile_name, geom in zip(image_gdf['tile'], image_gdf['geometry'])
    ]
    output_shapefile = os.path.join(os.path.dirname(image_shapefile), "2024_orthoHR_tiles_with_masks.shp")
    image_gdf.to_file(output_shapefile)
    print(f"Saved updated shapefile with mask paths to {output_shapefile}")

    # Print summary
    print(f"\nMask Creation Summary:")
    print(f"Total tiles processed: {total_tiles}")
    print(f"Successful masks: {successful_masks}")
    print(f"Failed masks: {failed_masks}")
    print(f"Skipped masks (no buildings): {skipped_masks}")

# Example usage
if __name__ == "__main__":
    image_shapefile = "wms_tiles/2024_orthoHR/2024_orthoHR_tiles.shp"
    building_shapefile = r"C:\Users\HDSL36\Desktop\Work\Projects\Change_detection\output\BGT\bgt_data.shp" # Replace with your building shapefile path
    create_masks(
        image_shapefile=image_shapefile,
        building_shapefile=building_shapefile,
        output_dir="wms_tiles/2024_orthoHR/masks",
        gsd=0.08
    )
    print(" ")