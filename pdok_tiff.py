import requests
import xml.etree.ElementTree as ET
import os
from pathlib import Path
import json
from math import ceil
import re
import geopandas as gpd
from shapely.geometry import box
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.exceptions import RequestException
from shapely import wkt
from tqdm import tqdm

class PDOKWMSDownloader:
    def __init__(self, wms_url, srs, version="1.3.0", default_max_size=4096, max_workers=10):
        self.wms_url = wms_url
        self.srs = srs
        self.version = version
        self.default_max_size = default_max_size
        self.max_workers = max_workers
        self.max_width, self.max_height = self._get_max_size()
        self.available_layers = self._get_available_layers()

    def _get_max_size(self):
        """Fetch max width and height from WMS GetCapabilities."""
        capabilities_url = f"{self.wms_url}?SERVICE=WMS&REQUEST=GetCapabilities&VERSION={self.version}"
        try:
            response = requests.get(capabilities_url, timeout=10)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            namespaces = {'wms': 'http://www.opengis.net/wms'}
            max_width_elem = root.find('.//wms:Service/wms:MaxWidth', namespaces)
            max_height_elem = root.find('.//wms:Service/wms:MaxHeight', namespaces)
            max_width = int(max_width_elem.text) if max_width_elem is not None else self.default_max_size
            max_height = int(max_height_elem.text) if max_height_elem is not None else self.default_max_size
            print(f"Max Width: {max_width}, Max Height: {max_height}")
            return max_width, max_height
        except Exception as e:
            print(f"Error fetching capabilities: {e}. Using default max size {self.default_max_size}.")
            return self.default_max_size, self.default_max_size

    def _get_available_layers(self):
        """Extract all available layer names from GetCapabilities."""
        capabilities_url = f"{self.wms_url}?SERVICE=WMS&REQUEST=GetCapabilities&VERSION={self.version}"
        try:
            response = requests.get(capabilities_url, timeout=10)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            layers = []
            for layer_elem in root.findall('.//wms:Layer/wms:Name', {'wms': 'http://www.opengis.net/wms'}):
                layers.append(layer_elem.text)
            return layers
        except Exception as e:
            print(f"Error fetching layers: {e}. Using fallback layers.")
            return [
                "Actueel_orthoHR", "Actueel_ortho25", "2025_orthoHR", "2025_ortho25",
                "2025_quickorthoHR", "2025_quickortho25", "2024_orthoHR", "2024_ortho25",
                "2023_orthoHR", "2023_ortho25", "2022_orthoHR", "2022_ortho25",
                "2021_orthoHR", "2020_ortho25", "2019_ortho25", "2018_ortho25",
                "2017_ortho25", "2016_ortho25"
            ]

    def _get_gsd(self, layer_name):
        """Extract GSD from the layer's Title or Abstract in GetCapabilities."""
        capabilities_url = f"{self.wms_url}?SERVICE=WMS&REQUEST=GetCapabilities&VERSION={self.version}"
        try:
            response = requests.get(capabilities_url, timeout=10)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            namespaces = {'wms': 'http://www.opengis.net/wms'}
            for layer_elem in root.findall('.//wms:Layer', namespaces):
                name_elem = layer_elem.find('wms:Name', namespaces)
                if name_elem is not None and name_elem.text == layer_name:
                    title = layer_elem.find('wms:Title', namespaces).text or ''
                    abstract = layer_elem.find('wms:Abstract', namespaces).text or ''
                    text = title + ' ' + abstract
                    match = re.search(r'(\d+)\s*cm', text)
                    if match:
                        cm = int(match.group(1))
                        return cm / 100.0
                    match = re.search(r'(\d+)\s*en\s*(\d+)\s*cm', text)
                    if match:
                        return int(match.group(2)) / 100.0
            print(f"GSD not found for {layer_name}. Using default 0.08m.")
            return 0.08
        except Exception as e:
            print(f"Error fetching GSD: {e}. Using default 0.08m.")
            return 0.08

    def list_layers(self):
        """Print available layers for user selection."""
        print("Available layers:")
        for i, layer in enumerate(self.available_layers, 1):
            print(f"{i}. {layer}")
        return self.available_layers

    def _get_wms_url(self, layer_name, minx, miny, maxx, maxy, width, height):
        """Generate WMS GetMap URL for a tile."""
        return (
            f"{self.wms_url}?SERVICE=WMS&REQUEST=GetMap&VERSION={self.version}"
            f"&LAYERS={layer_name}&STYLES=&CRS={self.srs}"
            f"&BBOX={minx},{miny},{maxx},{maxy}"
            f"&WIDTH={width}&HEIGHT={height}&FORMAT=image/jpeg"
        )

    def _download_tile(self, layer_name, tile_minx, tile_miny, tile_maxx, tile_maxy, width, height, output_file, i, j):
        """Download a single tile and return its coordinates if successful."""
        url = self._get_wms_url(layer_name, tile_minx, tile_miny, tile_maxx, tile_maxy, width, height)
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            if b"ServiceException" in response.content:
                print(f"Error for tile {i}_{j}: {response.content.decode()}")
                return None, False
            if not response.content.startswith(b'\xff\xd8'):
                print(f"Error: Response for tile {i}_{j} is not a valid JPEG.")
                return None, False
            with open(output_file, 'wb') as f:
                f.write(response.content) 
            # print(f"Downloaded {output_file}")
            return {
                "tile": os.path.basename(output_file),
                "bbox": [tile_minx, tile_miny, tile_maxx, tile_maxy],
                "srs": self.srs,
                "url": url
            }, True
        except RequestException as e:
            print(f"Error downloading tile {i}_{j} for {layer_name}: {e}")
            return None, False

    def download_tif(self, area, layer_name=None, output_dir="wms_tiles"):
        """Download all tiles intersecting the bounding box in parallel with progress tracking."""
        minx, miny, maxx, maxy = area.bounds

        # Validate or select layer
        if layer_name is None or layer_name not in self.available_layers:
            self.list_layers()
            try:
                layer_index = int(input("Enter the number of the layer to download (or 0 to exit): ")) - 1
                if layer_index == -1:
                    print("Exiting download.")
                    return
                if 0 <= layer_index < len(self.available_layers):
                    layer_name = self.available_layers[layer_index]
                else:
                    raise ValueError("Invalid layer selection.")
            except ValueError as e:
                print(f"Error: {e}")
                return
        print(f"Downloading tiles for layer: {layer_name}")

        # Get GSD for the layer
        gsd = self._get_gsd(layer_name)
        print(f"GSD for {layer_name}: {gsd} m/pixel")

        # Create output directory
        output_dir = os.path.join(output_dir, layer_name)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Calculate maximum map extent per tile based on max size and GSD
        tile_map_width = self.max_width * gsd
        tile_map_height = self.max_height * gsd

        # Calculate bounding box dimensions
        bbox_width = maxx - minx
        bbox_height = maxy - miny

        # Snap starting position to GSD grid using ceil
        minx_grid = ceil(minx / gsd) * gsd
        miny_grid = ceil(miny / gsd) * gsd

        # Calculate number of tiles
        n_tiles_x = ceil(bbox_width / tile_map_width)
        n_tiles_y = ceil(bbox_height / tile_map_height)
        print(f"Total potential tiles: {n_tiles_x * n_tiles_y} ({n_tiles_x}x{n_tiles_y})")

        # Create input bounding box geometry
        input_bbox = area

        # Prepare tile download tasks
        tile_tasks = []
        skipped_tiles = 0
        for i in range(n_tiles_x):
            for j in range(n_tiles_y):
                tile_minx = minx + i * tile_map_width
                tile_maxx = min(tile_minx + tile_map_width, maxx)
                tile_miny = miny + j * tile_map_height
                tile_maxy = min(tile_miny + tile_map_height, maxy)

                # Check if tile intersects input BBOX
                tile_bbox = box(tile_minx, tile_miny, tile_maxx, tile_maxy)
                if not input_bbox.intersects(tile_bbox):
                    print(f"Skipping tile {i}_{j}: does not intersect input bounding box")
                    skipped_tiles += 1
                    continue

                tile_bbox_width = tile_maxx - tile_minx
                tile_bbox_height = tile_maxy - tile_miny
                width = ceil(tile_bbox_width / gsd)
                height = ceil(tile_bbox_height / gsd)

                # Cap at max size
                width = min(width, self.max_width)
                height = min(height, self.max_height)

                output_file = os.path.join(output_dir, f"tile_{i}_{j}.jpg")
                tile_tasks.append((layer_name, tile_minx, tile_miny, tile_maxx, tile_maxy, width, height, output_file, i, j))

        print(f"Total tiles to download after intersection check: {len(tile_tasks)}")
        print(f"Skipped tiles (outside input area): {skipped_tiles}")

        # Download tiles in parallel with progress bar
        tile_coords = []
        successful_tiles = 0
        failed_tiles = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_tile = {
                executor.submit(self._download_tile, *task): task for task in tile_tasks
            }
            with tqdm(total=len(tile_tasks), desc="Downloading tiles", unit="tile") as pbar:
                for future in as_completed(future_to_tile):
                    result, success = future.result()
                    if result:
                        tile_coords.append(result)
                        successful_tiles += 1
                    else:
                        failed_tiles += 1
                    pbar.update(1)

        # Print summary
        print(f"\nDownload Summary:")
        print(f"Total tiles attempted: {len(tile_tasks)}")
        print(f"Successful downloads: {successful_tiles}")
        print(f"Failed downloads: {failed_tiles}")
        print(f"Skipped tiles: {skipped_tiles}")

        # Save geo coordinates to JSON
        output_json = os.path.join(output_dir, f"{layer_name}_geo_coordinates.json")
        with open(output_json, 'w') as f:
            json.dump({"tiles": tile_coords, "note": f"Each tile for {layer_name} is georeferenced with its bounding box, projection, and URL."}, f, indent=4)
        print(f"Saved geo coordinates to {output_json}")

        # Create shapefile
        if tile_coords:
            geometries = [box(coord['bbox'][0], coord['bbox'][1], coord['bbox'][2], coord['bbox'][3]) for coord in tile_coords]
            gdf = gpd.GeoDataFrame({
                'tile': [coord['tile'] for coord in tile_coords],
                'url': [coord['url'] for coord in tile_coords]
            }, geometry=geometries, crs=self.srs)
            shapefile_path = os.path.join(output_dir, f"{layer_name}_tiles.shp")
            gdf.to_file(shapefile_path)
            print(f"Saved shapefile to {shapefile_path}")
        else:
            print(f"No tiles downloaded for {layer_name}. Try checking layer availability or contact BeheerPDOK@kadaster.nl.")

# Example usage:
if __name__ == "__main__":
    polygon = wkt.loads('MultiPolygon (((125315.14984061298309825 477665.88163518119836226, 128333.10384891713329125 477755.17021530849160627, 129368.85137839429080486 480505.25848323066020384, 127654.51063994933792856 483148.20045499998377636, 129368.85137839429080486 484148.23255242616869509, 129940.29829120928479824 485166.12236587790539488, 127395.45982914829801302 486769.93157312169205397, 127395.45982914829801302 486769.93157312169205397, 123994.61250886785273906 487367.37772398180095479, 121926.52967896759219002 487091.63334666175069287, 121512.91311298753134906 483093.33987552125472575, 123443.12375422778131906 481163.12923428096110001, 123810.78292398783378303 478589.51504596066661179, 123672.91073532780865207 478175.89847998059121892, 125315.14984061298309825 477665.88163518119836226)))')
    downloader = PDOKWMSDownloader(
        wms_url="https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0",
        srs="EPSG:28992",
        max_workers=6
    )
    downloader.download_tif(
        area=polygon,
        layer_name="2024_orthoHR",
        output_dir="wms_tiles"
    )