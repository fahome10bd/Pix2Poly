import concurrent.futures
import glob
import math
import multiprocessing
import multiprocessing as mp
import os
from shapely.ops import unary_union
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from configparser import ConfigParser
from functools import partial
from typing import List, Optional, Tuple, Union
import logging
from pathlib import Path

# import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import psutil
import requests
from osgeo import ogr
from shapely import wkt
# from dask.distributed import Client, LocalCluster, progress
from pyogrio import read_dataframe
from scipy.spatial import KDTree
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box
from shapely.ops import unary_union

# from tqdm import tqdm
from tqdm.auto import tqdm


class BGT_Data:
    base_url = "https://api.pdok.nl/lv/bgt/download/v1_0"
    post_endpoint = "/full/custom"
    status_endpoint = "/full/custom/{}/status"

    def __init__(self, area, output_dir, project_name="bgt_data"):
        
        self.polygon = area
        self.output_dir = output_dir
        self.project_name = project_name
        self.headers = {"Content-Type": "application/json"}
        self.gfs_path = os.path.join(os.path.dirname(__file__), "gfs")
        self.bgt_types = {
            "wegdeel": "WGD",
            "waterdeel": "WTD",
            "begroeidterreindeel": "BTD",
            "scheiding": "SHD",
            "ondersteunendwaterdeel": "OWT",
            "ondersteunendwegdeel": "OWG",
            "onbegroeidterreindeel": "OTD",
            "overigbouwwerk": "OBW",
            "pand": "PND",
            "kunstwerkdeel": "KWD",
            "overbruggingsdeel": "OBD",
        }

    def request_download(self):
        payload = {
            "featuretypes": list(self.bgt_types.keys()),
            "format": "citygml",
            "geofilter": self.polygon.wkt,
        }

        response = requests.post(
            self.base_url + self.post_endpoint, json=payload, headers=self.headers
        )
        if response.status_code == 202:
            self.download_request_id = response.json()["downloadRequestId"]
            return self.download_request_id
        else:
            raise Exception(
                f"Failed to request download: {response.status_code}, {response.text}"
            )

    def check_status(self):
        response = requests.get(
            self.base_url + self.status_endpoint.format(self.download_request_id)
        )
        return response.json()

    def download_file(self, download_url):
        full_download_url = "https://api.pdok.nl" + download_url
        zip_path = os.path.join(self.output_dir, self.project_name) + ".zip"
        with requests.get(full_download_url, stream=True) as response:
            total_size = int(
                response.headers.get("content-length", 0)
            )  # Get total file size
            block_size = 1024000  # Define the size of each chunk (1KB)
            downloaded_size = 0  # Track the size of data downloaded so far

            with open(zip_path, "wb") as file:
                # Stream the file in chunks and save to disk
                for data in response.iter_content(block_size):
                    file.write(data)
                    downloaded_size += len(data)

                    # Calculate the percentage downloaded and print it
                    progress = (downloaded_size / total_size) * 100
                    logging.info(f"\rDownloading file: {progress:.2f}%")

        logging.info("\nDownload complete.")
        return zip_path

    def extract_zip(self, zip_path, extract_to=None):
        if extract_to is None:
            extract_to = os.path.join(
                self.output_dir, os.path.splitext(self.project_name)[0]
            )
        else:
            extract_to = os.path.join(self.output_dir, extract_to)

        os.makedirs(extract_to, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        logging.info(f"Files extracted to {extract_to}.")
        return extract_to

    def convert_gml_to_shapefile(self, gml_path, gfs_path, filter_by_eind=False):
        # Get the filename without extension
        base_name = os.path.splitext(os.path.basename(gml_path))[0]

        # Set the GFS file to be used
        os.environ["GML_GFS_TEMPLATE"] = gfs_path

        # Open the GML file
        gml_ds = ogr.Open(gml_path)
        if gml_ds is None:
            logging.info(f"Error: Could not open {gml_path}")
            return

        # Get the first layer
        gml_layer = gml_ds.GetLayer()

        # Create the output shapefile
        driver = ogr.GetDriverByName("ESRI Shapefile")
        output_path = os.path.join(self.output_dir, f"{base_name}.shp")

        # Remove output shapefile if it already exists
        if os.path.exists(output_path):
            driver.DeleteDataSource(output_path)

        # Create the new shapefile
        shp_ds = driver.CreateDataSource(output_path)
        shp_layer = shp_ds.CreateLayer(
            base_name, gml_layer.GetSpatialRef(), gml_layer.GetGeomType()
        )

        # Copy the fields from GML to shapefile
        gml_layer_defn = gml_layer.GetLayerDefn()
        allowed_field_types = [
            ogr.OFTString,
            ogr.OFTInteger,
            ogr.OFTReal,
        ]

        field_map = {}  # To map original field names to truncated names
        for i in range(gml_layer_defn.GetFieldCount()):
            field_defn = gml_layer_defn.GetFieldDefn(i)
            if field_defn.GetType() in allowed_field_types:
                field_name = field_defn.GetName()
                truncated_name = field_name[:10]  # Shapefile field name limit
                new_field = ogr.FieldDefn(truncated_name, field_defn.GetType())
                shp_layer.CreateField(new_field)
                field_map[field_name] = truncated_name
            else:
                logging.warning(
                    f"Skipping unsupported field type: {field_defn.GetName()} ({field_defn.GetFieldTypeName(field_defn.GetType())})"
                )

        # Copy the features from GML to shapefile (only those with NULL 'eindRegistratie')
        unique_type = []
        invalid_geom_types = {
            "LINESTRING",
            "MULTIPOINT",
            "CIRCULARSTRING",
            "COMPOUNDCURVE",
        }
        for feature in gml_layer:
            if filter_by_eind:
                eind_registratie = feature.GetField("eindRegistratie")
                if eind_registratie is not None:
                    continue
            geometry = feature.GetGeometryRef()
            geom_name = geometry.GetGeometryName().upper()
            if geom_name not in unique_type:
                unique_type.append(geom_name)
            if geom_name in invalid_geom_types:
                continue

            # Create new feature to match the target layer definition
            new_feature = ogr.Feature(shp_layer.GetLayerDefn())

            # Copy allowed fields with proper truncated names
            for orig_name, trunc_name in field_map.items():
                new_feature.SetField(trunc_name, feature.GetField(orig_name))

            # Set geometry
            new_feature.SetGeometry(geometry.Clone())

            # Add feature to the shapefile layer
            shp_layer.CreateFeature(new_feature)

            new_feature = None  # Free the feature

        # Close the datasets
        gml_ds = None
        shp_ds = None

        logging.info(f"Converted {gml_path} to {output_path}")

    def prepare_shp_from_gml(self, extract_dir):
        # Convert all GML files in the extracted directory

        for gml_file in glob.glob(os.path.join(extract_dir, "*.gml")):
            base_name = os.path.splitext(os.path.basename(gml_file))[0]
            gfs_file = os.path.join(self.gfs_path, f"{base_name}_V.gfs")

            if os.path.exists(gfs_file):
                # Apply filtering only for 'bgt_pand'
                filter_eind = base_name.lower() == "bgt_pand"
                self.convert_gml_to_shapefile(
                    gml_file, gfs_file, filter_by_eind=filter_eind
                )
            else:
                logging.info(
                    f"Warning: No GFS file found for {gml_file}. Skipping conversion."
                )

        logging.info("All GML files have been converted to shapefiles.")

    def merge_geo_files(self, folder_path, file_name, ext=".shp"):
        geo_files = [f for f in os.listdir(folder_path) if f.endswith(ext)]
        data_frame_array = []
        for geo_file in geo_files:
            geo_path = os.path.join(folder_path, geo_file)
            geo_name = geo_file.split(".")[0]  # .split('_')[1]

            logging.info(f"Reading geo file: {geo_path}")
            gdf = read_dataframe(geo_path)
            for index, row in gdf.iterrows():
                geometry = row.geometry
                if not (
                    (
                        isinstance(geometry, Polygon)
                        or isinstance(geometry, MultiPolygon)
                    )
                ):
                    logging.info(geometry.type)
                    continue

                data = {
                    "cadid": row["gml_id"],
                    "blk": "Null",
                    "otype": geo_name,
                    "mapnr": "Null",
                    "bgtgvt": "Null",
                    "crtdate": row["creationDa"],
                    "operator": None,
                    "x": geometry.centroid.x,
                    "y": geometry.centroid.y,
                    "bgtid": "Null",
                    "bgttype": self.bgt_types[geo_name.split("_")[1]],
                    "fysiekvoor": "Null",
                    "tmain": "Null",
                    "level": "Null",
                    "laatste_wi": "Null",
                    "lastupdate": "Null",
                    "status": "Null",
                    "geom_opper": geometry.area,
                    "geomarea": geometry.area,
                    "nrpoints": 0,
                    "nrrings": 1,
                    "tooltiptex": "Null",
                    "geometry": geometry,
                    "height": pd.NA,
                    "imgsourcs": pd.NA,
                    "score": pd.NA,
                }
                data_frame_array.append(data)
            logging.info(f"Completed processing {geo_file}")
        bgt_df = pd.DataFrame(data_frame_array)
        bgt_gdf = gpd.GeoDataFrame(bgt_df, geometry="geometry")
        bgt_gdf.set_crs(epsg=28992, inplace=True)
        file_name = os.path.join(self.output_dir, file_name) + ext
        bgt_gdf.to_file(file_name, encoding="utf-8", engine="pyogrio")
        return bgt_gdf

    def _validate_zip(self, zip_path):
        """
        Validates the integrity of a ZIP file.

        Args:
            zip_path (str): The path to the ZIP file.

        Returns:
            bool: True if the ZIP file is valid, False otherwise.
        """
        if not os.path.exists(zip_path):
            logging.error(f"Validation failed: ZIP file not found at {zip_path}")
            return False

        try:
            with zipfile.ZipFile(zip_path, "r") as zip_file:
                # testzip() checks CRCs and headers. Returns name of first bad file or None.
                bad_file = zip_file.testzip()
                if bad_file is None:
                    logging.info(f"ZIP file validation successful: {zip_path}")
                    return True
                else:
                    logging.error(
                        f"Validation failed: Corrupt file found in ZIP: {bad_file} in {zip_path}"
                    )
                    return False
        except zipfile.BadZipFile:
            logging.error(
                f"Validation failed: File is not a valid ZIP archive or is corrupted: {zip_path}",
                exc_info=True,
            )
            return False
        except Exception as e:
            logging.error(
                f"Validation failed: An unexpected error occurred opening/testing {zip_path}: {e}",
                exc_info=True,
            )
            return False

    def run(self):
        self.output_dir = os.path.join(self.output_dir, "BGT")
        bgt_dir = (
            os.path.join(self.output_dir, self.project_name)
            + ".shp"
        )
        if os.path.exists(bgt_dir):
            return bgt_dir

        os.makedirs(self.output_dir, exist_ok=True)
        zip_path = os.path.join(self.output_dir, self.project_name) + ".zip"
        zip_exists = os.path.exists(zip_path)
        zip_valid = False
        download_needed = not zip_exists

        # --- Initial Validation if ZIP Exists ---
        if zip_exists:
            # self.progress_callback(5, f"Validating existing ZIP: {os.path.basename(zip_path)}...")
            zip_valid = self._validate_zip(zip_path)
            if zip_valid:
                # self.progress_callback(60, "Existing ZIP is valid.") # Jump progress
                download_needed = False  # No need to download if valid
            else:
                # self.progress_callback(5, f"Existing ZIP is invalid. Deleting and redownloading...")
                logging.warning(
                    f"Existing ZIP file {zip_path} is invalid. Deleting..."
                )
                try:
                    os.remove(zip_path)
                    download_needed = True  # Force download
                except OSError as e:
                    logging.error(f"Failed to delete invalid ZIP {zip_path}: {e}")
                    # self.progress_callback(5, "Error: Could not delete invalid ZIP.")
                    raise  # Stop if we can't delete the bad file
        if download_needed:
            self.request_download()
        while download_needed:
            status_response = self.check_status()
            status = status_response["status"]
            if status == "COMPLETED":
                download_url = status_response["_links"]["download"]["href"]
                logging.info("\nDownload ready. Starting download...")
                zip_path = self.download_file(download_url)
                logging.info("Downloaded BGT files on: ", zip_path)
                break
            elif status == "RUNNING":
                progress = status_response["progress"]
                logging.info(
                    f"\rDownload in progress: {progress}%"
                )  # Update in place
                time.sleep(2)  # Wait before checking again
            elif status == "PENDING":
                logging.info(
                    "Download is pending. Waiting for the process to start..."
                )
                time.sleep(5)  # Wait before checking again
            elif status == "FAILED":
                raise Exception(f"Download failed: {status_response['reason']}")
            else:
                raise Exception(f"Unexpected status: {status}")

        shp_dir = os.path.join(self.output_dir, "shp")
        # Check if the zip file extracted folder and files exists
        extract = True
        if os.path.exists(zip_path.split(".")[0]):
            # count existing file is the same as bgt type
            if len(glob.glob(os.path.join(zip_path.split(".")[0], "*.gml"))) == len(
                self.bgt_types
            ):
                logging.info("BGT files already extracted. Skipping extraction...")
                extract = False
        if extract:
            logging.info("Extracting Zip file...")
            extract_to = self.extract_zip(zip_path)
        else:
            extract_to = os.path.join(
                self.output_dir, os.path.splitext(self.project_name)[0]
            )
        # Check if the shape file exists
        process_shape = True
        if os.path.exists(shp_dir):
            if len(glob.glob(os.path.join(shp_dir, "*.shp"))) == len(
                self.bgt_types
            ):
                logging.info("Shape files already exists. Skipping processing...")
                process_shape = False
        if process_shape:
            logging.info("Processign GML files...")
            # Create a shp folder
            os.makedirs(shp_dir, exist_ok=True)
            temp_dir = self.output_dir
            self.output_dir = shp_dir
            self.prepare_shp_from_gml(extract_to)
            self.output_dir = temp_dir
        bgt_dir = (
            os.path.join(self.output_dir, self.project_name)
            + ".shp"
        )
        if not os.path.exists(bgt_dir):
            logging.info("Merging Shape files...")
            bgt_gdf = self.merge_geo_files(shp_dir, self.project_name)
            logging.info("Completed processing BGT files...")
            logging.info("BGT file location: ", bgt_dir)
            return bgt_dir, bgt_gdf
        else:
            logging.info("Shape file already exists. Skipping merge...")
            logging.info("Completed processing BGT files...")
            logging.info("BGT file location: ", bgt_dir)
            return bgt_dir, None

if __name__ == "__main__":
    polygon = wkt.loads('MultiPolygon (((125315.14984061298309825 477665.88163518119836226, 128333.10384891713329125 477755.17021530849160627, 129368.85137839429080486 480505.25848323066020384, 127654.51063994933792856 483148.20045499998377636, 129368.85137839429080486 484148.23255242616869509, 129940.29829120928479824 485166.12236587790539488, 127395.45982914829801302 486769.93157312169205397, 127395.45982914829801302 486769.93157312169205397, 123994.61250886785273906 487367.37772398180095479, 121926.52967896759219002 487091.63334666175069287, 121512.91311298753134906 483093.33987552125472575, 123443.12375422778131906 481163.12923428096110001, 123810.78292398783378303 478589.51504596066661179, 123672.91073532780865207 478175.89847998059121892, 125315.14984061298309825 477665.88163518119836226)))')
    output_dir = r"output"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    bgt_data = BGT_Data(area=polygon, output_dir=output_dir, project_name="bgt_data")
    bgt_path = bgt_data.run()
    print(" ")
    