import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import Draw
import numpy as np
import pandas as pd
import joblib
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as cx
from shapely.geometry import Point, box
import rioxarray as rxr
from pyproj import Proj, Transformer
from scipy.spatial import cKDTree
from tqdm import tqdm

# Model Saving
import joblib

# Feature Engineering
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Supress Warnings
import warnings
warnings.filterwarnings('ignore')

# Import Data Science Packages
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Import common GIS tools
import xarray as xr
import matplotlib.pyplot as plt
import rioxarray as rio
import rasterio
from matplotlib.cm import RdYlGn,jet,RdBu

# Geospatial raster data handling
import rioxarray as rxr

# Coordinate transformations
from pyproj import Proj, Transformer, CRS

# Others
import os
from tqdm import tqdm

# ckdTree for mapping
from scipy.spatial import cKDTree

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# Import Planetary Computer tools
import stackstac
import pystac_client
import planetary_computer
from odc.stac import stac_load

# Chatbot
from openai import OpenAI
import base64
from pypdf import PdfReader
import io
import requests

# set the number of columns we see
pd.set_option('display.max_columns',None)


import warnings
warnings.filterwarnings('ignore')


GOOGLE_KEY= (st.secrets.get("GOOGLE_MAPS_API_KEY", "") or os.getenv("GOOGLE_MAPS_API_KEY", "")).strip()
if not GOOGLE_KEY:
    st.error("GOOGLE_MAPS_API_KEY가 설정되지 않았습니다. `.streamlit/secrets.toml` 또는 환경변수에 키를 넣어주세요.")
    st.stop()



st.set_page_config(page_title="UHI Intensity Predictor", layout="wide")
st.title("Urban Heat Island (UHI) Intensity Predictor")
st.write("Draw a rectangle on the map to select an area and generate UHI predictions")



with st.sidebar:
    st.title("ChatGPT will help you")

    # Set OpenAI API key from Streamlit secrets
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    # Set a default model
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "ft:gpt-4o-2024-08-06:personal::CNA5x6eJ"

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    MODEL_ID = "ft:gpt-4o-2024-08-06:personal::CNA5x6eJ"


    # Accept user input
    if prompt := st.chat_input("What is up?", accept_file="multiple", file_type=["jpg", "jpeg", "png", "pdf", "txt"]):
        # Save the user's text message to the chat history
        st.session_state.messages.append({"role": "user", "content": prompt.text})
        
        # Display the user's message
        with st.chat_message("user"):
            st.markdown(prompt.text)

        # --- Process files and prepare content for the API ---
        
        # This list will hold all parts of the user's message (text, images, file content)
        content_parts = []
        
        # Add the user's typed text as the first part
        content_parts.append({"type": "text", "text": prompt.text})

        if prompt["files"]:
            for uploaded_file in prompt["files"]:
                try:
                    if uploaded_file.type in ["image/png", "image/jpeg"]:
                        # Handle image files
                        image_bytes = uploaded_file.getvalue()
                        base64_image = base64.b64encode(image_bytes).decode('utf-8')
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{uploaded_file.type};base64,{base64_image}"}
                        })
                    elif uploaded_file.type == "application/pdf":
                        # Handle PDF files
                        pdf_reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
                        pdf_text = ""
                        for page in pdf_reader.pages:
                            pdf_text += page.extract_text() or ""
                        content_parts.append({"type": "text", "text": f"Content from PDF '{uploaded_file.name}':\n{pdf_text}"})
                    elif uploaded_file.type.startswith("text/"):
                        # Handle text-based files
                        string_data = uploaded_file.getvalue().decode("utf-8")
                        content_parts.append({"type": "text", "text": f"Content from file '{uploaded_file.name}':\n{string_data}"})
                except Exception as e:
                    st.error(f"Error processing file {uploaded_file.name}: {e}")

        # --- Call the API and display the response ---

        with st.chat_message("assistant"):
            # Prepare the full message list for the API
            # System message + all previous messages + the current rich-content message
            api_messages = [{"role": "system", "content": "You are a helpful assistant."}]
            
            # Add all messages from history (they are already correctly formatted as strings)
            for msg in st.session_state.messages[:-1]: # Exclude the last message we just added
                api_messages.append(msg)
            
            # Add the final, rich-content user message
            api_messages.append({"role": "user", "content": content_parts})

            try:
                # Determine the model to use based on whether image content is present
                model_to_use = MODEL_ID
                if any(part["type"] == "image_url" for part in content_parts):
                    model_to_use = "gpt-4o" # Use a vision-capable model if images are present
                    st.session_state["openai_model"] = "gpt-4o" # Update session state as well

                response = client.chat.completions.create(
                    model=model_to_use,
                    messages=api_messages
                )
                response_text = response.choices[0].message.content
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            except Exception as e:
                st.error(f"An error occurred with the OpenAI API: {e}")



# Built-in Mapbox styles
streets_tiles = f"https://api.mapbox.com/styles/v1/mapbox/streets-v11/tiles/{{z}}/{{x}}/{{y}}?access_token={st.secrets['MAPBOX_TOKEN']}"
satellite_tiles = f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/tiles/{{z}}/{{x}}/{{y}}?access_token={st.secrets['MAPBOX_TOKEN']}"

# Custom tileset URL using your tileset id and token
custom_tileset_id = st.secrets["TILESET_ID"]
custom_tiles = f"https://api.mapbox.com/v4/{custom_tileset_id}/{{z}}/{{x}}/{{y}}.png?access_token={st.secrets['MAPBOX_TOKEN']}"

# Create initial map centered on default location
m = folium.Map(location=[37.77, -122.43], zoom_start=11, tiles=None)

# Add standard Mapbox layers
folium.TileLayer(
    tiles=streets_tiles,
    attr="Mapbox Streets",
    name="Mapbox Streets",
    overlay=False,
    control=True
).add_to(m)

folium.TileLayer(
    tiles=satellite_tiles,
    attr="Mapbox Satellite",
    name="Mapbox Satellite",
    overlay=False,
    control=True
).add_to(m)

# Add custom Mapbox Studio tileset layer
folium.TileLayer(
    tiles=custom_tiles,
    attr="Custom Mapbox Tileset",
    name="Custom Tileset",
    overlay=True,
    control=True
).add_to(m)


left_col, right_col = st.columns([1, 1])
bottom_left, bottom_right = st.columns([1, 1])

with left_col:
    # Add drawing plugin with only rectangle enabled
    draw = Draw(
        export=True, 
        draw_options={
            'rectangle': True, 
            'polyline': False, 
            'polygon': False, 
            'circle': False, 
            'marker': False
        }
    )
    draw.add_to(m)

    # Add layer control for toggling
    folium.LayerControl().add_to(m)



    # Display folium map in Streamlit
    map_data = st_folium(m, width=700, height=500)


    if map_data and map_data.get("last_active_drawing"):
        rect_coords = map_data["last_active_drawing"]["geometry"]["coordinates"][0]
        lats = [coord[1] for coord in rect_coords]
        lons = [coord[0] for coord in rect_coords]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)
        
        st.write("### Bounding Box Coordinates:")
        st.write(f"Min Latitude: {min_lat}")
        st.write(f"Max Latitude: {max_lat}")
        st.write(f"Min Longitude: {min_lon}")
        st.write(f"Max Longitude: {max_lon}")
        
        greensburg_location = (min_lon, min_lat, max_lon, max_lat)



# Google Places API에서 마커 오버레이
@st.cache_data(show_spinner=False)
def get_places_cached(api_key: str, center: tuple, radius: int, place_type: str):
    """
    Google Places API를 통해 특정 반경 내의 장소(병원, 공원 등)를 가져옴.
    캐시 적용되어 동일한 요청은 재호출하지 않음.
    """
    lat, lon = center
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lon}",
        "radius": radius,
        "type": place_type,
        "key": api_key
    }
    res = requests.get(url, params=params)
    results = res.json().get("results", [])

    return [
        (r["geometry"]["location"]["lat"],
         r["geometry"]["location"]["lng"],
         r["name"]) for r in results
    ]




with bottom_left:
    st.subheader("Hospital & Park(Google Places API Marker)")

    #  바운딩박스 결과로 중심 계산
    if 'min_lat' in locals():
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2
    else:
        center_lat, center_lon = 37.77, -96.8

    GOOGLE_KEY = st.secrets.get("GOOGLE_MAPS_API_KEY", "") or os.getenv("GOOGLE_MAPS_API_KEY", "")
    if not GOOGLE_KEY:
        st.warning("Google Maps API 키가 필요합니다.")
    else:
        #  병원과 공원 데이터 가져오기
        hospitals = get_places_cached(GOOGLE_KEY, (center_lat, center_lon), 3000, "hospital")
        parks = get_places_cached(GOOGLE_KEY, (center_lat, center_lon), 3000, "park")

        # Folium 지도 생성
        map_overlay = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="OpenStreetMap")

        # 병원 마커 추가
        for lat, lon, name in hospitals:
            folium.Marker(
                [lat, lon],
                popup=f"🏥 {name}",
                icon=folium.Icon(color="red", icon="plus-sign")
            ).add_to(map_overlay)

        # 공원 마커 추가
        for lat, lon, name in parks:
            folium.Marker(
                [lat, lon],
                popup=f"🌳 {name}",
                icon=folium.Icon(color="green", icon="tree-conifer")
            ).add_to(map_overlay)

        # Streamlit 출력
        st_folium(map_overlay, width=700, height=500)




#! Satellite Imagery Creation Functions
# create a function that can collect satellite data
def create_xarray_data(location, date_range, satellite, cloud_coverage, bands, resolution):
    """
    Fetch Sentinel-2 imagery as an xarray dataset.

    Parameters:
    - location: Tuple (min_lon, min_lat, max_lon, max_lat) defining the bounding box.
    - date_range: String in 'YYYY-MM-DD/YYYY-MM-DD' format defining the time window.
    - satellite: Name of the satellite collection (e.g., 'sentinel-2-l2a').
    - cloud_coverage: Max cloud cover percentage (integer, e.g., 30 for <30% clouds).
    - bands: List of band names (e.g., ['B02', 'B03', 'B04'] for RGB).
    - resolution: Desired spatial resolution in meters (e.g., 10m for Sentinel-2).

    Returns:
    - xarray.Dataset containing the requested satellite data.
    """

    # Define bounding box
    # Connect to Microsoft Planetary Computer STAC API
    stac = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")


    # Search for matching satellite imagery
    search = stac.search(
        bbox=location,
        datetime=date_range,
        collections=[satellite],
        query={"eo:cloud_cover": {"lt": cloud_coverage}},
    )

    # Convert search results to a list
    items = list(search.get_items())
    print(f'Satellite Images Found: {len(items)}')

    if not items:
        raise ValueError("No STAC items found for the given parameters.")

    # Sign the items for access
    signed_items = [planetary_computer.sign(item) for item in items]

    # Load data using stackstac
    data = stac_load(
        signed_items,
        bands=bands,
        crs="EPSG:4326",  # Latitude-Longitude
        resolution=resolution / 111320.0,  # Ensure resolution is in meters
        chunks={"x": 2048, "y": 2048},  # Optimize chunking
        dtype="uint16",
        patch_url=planetary_computer.sign,
        bbox=location
    )

    return data


def sentinel_1_xarray(location,date_range,satellite,bands):
    # Step 1: Connect to the Planetary Computer STAC API
    catalog_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
    catalog = pystac_client.Client.open(catalog_url)

    search = catalog.search(
        collections=[satellite],  # Sentinel-1 GRD product
        bbox=location,
        datetime=date_range,
        limit=10
    )

    # Step 3: Retrieve the items
    items = list(search.items())
    print(f"Found {len(items)} Sentinel-1 scenes.")

    signed_items = [planetary_computer.sign(item).to_dict() for item in items]

        # Define the pixel resolution for the final product
    # Define the scale according to our selected crs, so we will use degrees
    resolution = 10  # meters per pixel
    scale = resolution / 111320.0 # degrees per pixel for crs=4326

    # 'items' is a list of STAC Items from a prior search (e.g., via pystac_client)
    sentinel_1_july_26th = stac_load(
        items,                           # STAC Items from Sentinel-1 GRD search
        bands=bands,             # Sentinel-1 polarization bands
        crs="EPSG:4326",                # Latitude-Longitude coordinate system
        resolution=scale,               # Degrees (set 'scale' to desired resolution, e.g., 0.0001 for ~10 m)
        chunks={"x": 2048, "y": 2048},  # Dask chunks for lazy loading
        dtype="uint16",                 # GRD data typically uint16 (check metadata if float32 needed)
        patch_url=planetary_computer.sign,  # Sign URLs for Planetary Computer access
        bbox=location                     # Bounding box in EPSG:4326 [min_lon, min_lat, max_lon, max_lat]
    )

    return sentinel_1_july_26th

    
# function to create a median mosaic
def median_mosaic(data):
    median = data.median(dim='time').compute()
    return median

def create_lidar_data(satellite,location,date_range):
    client = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1/",
    )

    search = client.search(
        collections=[satellite],
        bbox=location,
        datetime=date_range
    )

    # Retrieve items
    items = {x.collection_id: x for x in search.get_all_items()}

    # Print available collection IDs for debugging
    print("Available collection IDs:", items.keys())

    item = items[satellite]

    asset_key = 'data'

    signed_url = planetary_computer.sign(item.assets[asset_key].href)

    file = rxr.open_rasterio(signed_url)

    return file


from rasterio.transform import from_bounds

def lidar_dtm_file(filename, image_data, bounds, location):
    """
    Saves a single-band raster (GeoTIFF) from an xarray dataset.

    Parameters:
    - filename (str): Name of the output file (e.g., "output.tif").
    - image_data (xarray.DataArray): The dataset containing raster data.
    - bounds (tuple): Bounding box (min_lon, min_lat, max_lon, max_lat).
    - location (str): Directory where the file will be saved.

    Returns:
    - None (Saves the GeoTIFF file)
    """

    filename = os.path.join(location, filename)

    # Extract width and height from dataset dimensions
    time, height, width = image_data.shape  # (4097, 4097)

    # Create raster transform using bounding box
    transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)

    # Assign CRS and transform to the dataset
    image_data.rio.write_crs("epsg:4326", inplace=True)
    image_data.rio.write_transform(transform, inplace=True)

    # Save the dataset as a GeoTIFF
    with rasterio.open(filename, 'w', driver='GTiff', width=width, height=height,
                       count=1, crs='epsg:4326', transform=transform, dtype='float32',
                       compress='lzw') as dst:
        dst.write(image_data.values.squeeze(), 1)  # Write the single band

    print(f"GeoTIFF saved: {filename}")


def lidar_file(filename, image_data, bounds, location):
    """
    Saves a single-band raster (GeoTIFF) from an xarray dataset.

    Parameters:
    - filename (str): Name of the output file (e.g., "output.tif").
    - image_data (xarray.Dataset): The dataset containing raster data with an 'elevation' variable.
    - bounds (tuple): Bounding box (min_lon, min_lat, max_lon, max_lat).
    - location (str): Directory where the file will be saved.

    Returns:
    - None (Saves the GeoTIFF file)
    """

    filename = os.path.join(location, filename)

    # Access the data variable and its shape from the Dataset
    elevation_data_array = image_data['elevation']
    # Correctly unpack height and width from the squeezed shape
    height, width = elevation_data_array.shape[-2:]


    # Create raster transform using bounding box
    transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)

    # Assign CRS and transform to the data variable
    elevation_data_array.rio.write_crs("epsg:4326", inplace=True)
    elevation_data_array.rio.write_transform(transform, inplace=True)

    # Save the data variable as a GeoTIFF
    with rasterio.open(filename, 'w', driver='GTiff', width=width, height=height,
                       count=1, crs='epsg:4326', transform=transform, dtype='float32',
                       compress='lzw') as dst:
        # Squeeze the data to remove dimensions of size 1 before writing
        dst.write(elevation_data_array.values.squeeze(), 1)  # Write the single band

    print(f"GeoTIFF saved: {filename}")



#! Helper functions
def map_satellite_data_xarray(xarray_data, locations_df, bands):
    lats = locations_df['Latitude'].values
    lons = locations_df['Longitude'].values

    result = pd.DataFrame({
        'Latitude': lats,
        'Longitude': lons
    })

    for band in bands:
        # Collect scalar band values for every location
        band_values = []
        for lat, lon in zip(lats, lons):
            # Extract scalar with .item() for single float
            val = xarray_data[band].sel(
                latitude=lat,
                longitude=lon,
                method='nearest'
            ).item()
            band_values.append(val)
        result[band] = band_values
    return result


def map_satellite_data(tiff_path,training_data, band_names):
    # Load the GeoTIFF data
    data = rxr.open_rasterio(tiff_path)
    tiff_crs = data.rio.crs

    # Read the CSV file
    latitudes = training_data['Latitude'].values
    longitudes = training_data['Longitude'].values

    # Convert lat/long to the GeoTIFF's CRS
    proj_wgs84 = Proj(init='epsg:4326')
    proj_tiff = Proj(tiff_crs)
    transformer = Transformer.from_proj(proj_wgs84, proj_tiff)

    # Reproject coordinates
    transformed_coords = [transformer.transform(lat, lon) for lat, lon in zip(latitudes, longitudes)]
    xs, ys = zip(*transformed_coords)

    # Get available band numbers
    band_numbers = data.coords["band"].values

    # Validate that the user provided the correct number of band names
    if len(band_names) != len(band_numbers):
        raise ValueError(f"Expected {len(band_numbers)} band names, but got {len(band_names)}")

    # Dictionary to store extracted band values
    band_data = {band_name: [] for band_name in band_names}

    # Extract values for each band
    for x, y in tqdm(zip(xs, ys), total=len(xs), desc="Extracting band values"):
        for band, band_name in zip(band_numbers, band_names):
            value = data.sel(x=x, y=y, band=band, method="nearest").values
            band_data[band_name].append(value)

    # Convert dictionary to DataFrame
    return pd.DataFrame(band_data)


def create_rounded_data(data):

    # convert the data to float values
    data = data.astype(float)

    # round by 3 decimal places
    data[['Longitude','Latitude']] = data[['Longitude','Latitude']].round(3)

    # group by the similar long/lat
    data = data.groupby(['Longitude','Latitude']).median().reset_index()

    return data


def add_nearest_data(uhi_data: pd.DataFrame, data_to_map: pd.DataFrame) -> pd.DataFrame:
    """
    Adds the nearest elevation value to the UHI dataset using a KDTree for fast nearest-neighbor search.

    Parameters:
        uhi_data (pd.DataFrame): DataFrame containing UHI data with 'Latitude' and 'Longitude' columns.
        elevation_data (pd.DataFrame): DataFrame containing elevation data with 'Latitude', 'Longitude', and 'z_grade' columns.

    Returns:
        pd.DataFrame: UHI dataset with an added 'Elevation' column.
    """
    # Extract coordinates from both datasets
    elevation_coords = np.array(list(zip(data_to_map["Latitude"], data_to_map["Longitude"])))
    uhi_coords = np.array(list(zip(uhi_data["Latitude"], uhi_data["Longitude"])))

    # Build KDTree for fast nearest-neighbor search
    tree = cKDTree(elevation_coords)

    # Find the nearest elevation point for each UHI point
    distances, indices = tree.query(uhi_coords)

    # Add the matched elevation values to the UHI dataset
    uhi_data = uhi_data.copy()

    for col in data_to_map.drop(columns=['Longitude','Latitude']):
        uhi_data[col] = data_to_map.iloc[indices][col].values

    return uhi_data

# Combine two datasets vertically (along columns) using pandas concat function.
def combine_two_datasets(dataset1,dataset2):
    '''
    Returns a  vertically concatenated dataset.
    Attributes:
    dataset1 - Dataset 1 to be combined
    dataset2 - Dataset 2 to be combined
    '''

    data = pd.concat([dataset1,dataset2], axis=1)
    return data


#UHI 
with right_col:
    if map_data and map_data.get("last_active_drawing"):
        with st.spinner("Creating prediction grid..."):
            # Create grid with 270x280 points
            #! Reduce the number of points?
            long = np.linspace(min_lon, max_lon, 270)
            lat = np.linspace(min_lat, max_lat, 280)
            longitudes, latitudes = np.meshgrid(long, lat)
            
            location_df = pd.DataFrame({
                'Longitude': longitudes.flatten(),
                'Latitude': latitudes.flatten()
            })
            
            st.success(f"Created grid with {len(location_df)} points")






        # Define band names
        sentinel_2_bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
        sentinel_1_bands = ['vv','vh']
        thermal_bands = ['lwir11']
        non_thermal_bands = ['red','green','blue','nir08']
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Sentinel-2 data
        status_text.text("Processing Sentinel-2 data...")
        progress_bar.progress(15)
        
        
        summer_2024 = '2024-06-01/2024-09-01'
        sentinel_2_satellite = "sentinel-2-l2a"
        cloud_coverage = 30
        sentinel_2_bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
        resolution = 10

        greensburg_sentinel_2_summer_2024_xarray = create_xarray_data(greensburg_location,summer_2024,sentinel_2_satellite,cloud_coverage,sentinel_2_bands,resolution)
        
        # get the median mosaic of the sentinel 2 summer data in greensburg
        greensburg_sentinel_2_summer_2024 = median_mosaic(greensburg_sentinel_2_summer_2024_xarray)
        
        
        greensburg_sentinel_2_summer_data = map_satellite_data_xarray(
        greensburg_sentinel_2_summer_2024,  # xarray.Dataset
        location_df,             # DataFrame of locations
        sentinel_2_bands                    # List of band names
        )
        
        
        greensburg_sentinel_2_summer_data.to_csv('created_data/sentinel_2_summer_data.csv')
        
        greensburg_sentinel_2_summer_data = pd.read_csv('created_data/sentinel_2_summer_data.csv',index_col=0)
        
        
        # Combine greensburg_location_df with only the band columns from greensburg_sentinel_2_summer_data
        greensburg_sentinel_2_summer_combined = combine_two_datasets(
        location_df,
        greensburg_sentinel_2_summer_data[sentinel_2_bands]
        )

        # Now call create_rounded_data with the combined DataFrame
        greensburg_sentinel_2_rounded = create_rounded_data(greensburg_sentinel_2_summer_combined)

        # Then add the nearest data
        greensburg_proximity_sentinel_2 = add_nearest_data(location_df, greensburg_sentinel_2_rounded)
        
        
        
        # Step 2: Sentinel-1 data
        status_text.text("Processing Sentinel-1 data...")
        progress_bar.progress(30)
        
        
        sentinel_1 = "sentinel-1-grd"                 # Sentinel-1 GRD
        low_cloud_coverage = 0                        # Max 10% cloud cover
        sentinel_1_all_bands = ['vv','vh']            # Sentinel 1 bands
        resolution = 10                               # 10m resolution

        greensburg_sentinel_1_summer_xarray = sentinel_1_xarray(greensburg_location,summer_2024,sentinel_1,sentinel_1_all_bands)
        
        greensburg_sentinel_1_summer_2024 = median_mosaic(greensburg_sentinel_1_summer_xarray)
        
        greensburg_sentinel_1_summer_data = map_satellite_data_xarray(
        greensburg_sentinel_1_summer_2024,  # xarray.Dataset
        location_df,             # DataFrame of locations
        sentinel_1_all_bands                    # List of band names
        )
        
        # sentinel_1_combined = pd.concat([location_df, greensburg_sentinel_1_summer_data], axis=1)
        
        greensburg_sentinel_1_rounded = create_rounded_data(greensburg_sentinel_1_summer_data)

        greensburg_sentinel_1_rounded_mapped = add_nearest_data(location_df,greensburg_sentinel_1_rounded)
        
        
        # Step 3: Landsat thermal data
        status_text.text("Processing Landsat thermal data...")
        progress_bar.progress(45)
        
        
        july_31 = '2024-06-25/2024-06-30'
        landsat = "landsat-c2-l2"
        cloud_coverage = 30
        thermal_bands = ['lwir11']
        resolution = 10

        greensburg_landsat_thermal_xarray = create_xarray_data(greensburg_location,july_31,landsat,cloud_coverage,thermal_bands,10)
        
        greensburg_landsat_thermal = median_mosaic(greensburg_landsat_thermal_xarray)
        
        # Scale Factors for the Surface Temperature band
        scale2 = 0.00341802
        offset2 = 149.0
        kelvin_celsius = 273.15 # convert from Kelvin to Celsius
        greensburg_landsat_thermal = greensburg_landsat_thermal.astype(float) * scale2 + offset2 - kelvin_celsius
        
        
        greensburg_landsat_thermal_data = map_satellite_data_xarray(
        greensburg_landsat_thermal,  # xarray.Dataset
        location_df,             # DataFrame of locations
        thermal_bands                    # List of band names
        )
        
        greensburg_landsat_thermal_data = combine_two_datasets(location_df,greensburg_landsat_thermal_data)
        
        
        
        # Step 4: Landsat non-thermal data
        status_text.text("Processing Landsat non-thermal data...")
        progress_bar.progress(60)
        
        
        landsat = "landsat-c2-l2"
        cloud_coverage = 30
        non_thermal_bands = ['red','green','blue','nir08']
        resolution = 10

        greensburg_landsat_non_thermal_summer_xarray = create_xarray_data(greensburg_location,summer_2024,landsat,cloud_coverage,non_thermal_bands,10)
        
        greensburg_landsat_non_thermal_summer = median_mosaic(greensburg_landsat_non_thermal_summer_xarray)
        
        # Scale Factors for the RGB and NIR bands
        scale1 = 0.0000275
        offset1 = -0.2
        greensburg_landsat_non_thermal_summer[['red','blue','green','nir08']] = greensburg_landsat_non_thermal_summer[['red','blue','green','nir08']].astype(float) * scale1 + offset1
        
        
        greensburg_non_thermal_data = map_satellite_data_xarray(
            greensburg_landsat_non_thermal_summer,  # xarray.Dataset
            location_df,             # DataFrame of locations
            non_thermal_bands                    # List of band names
        )

        # Combine greensburg_location_df with only the non-thermal band columns from greensburg_non_thermal_data
        greensburg_non_thermal_data_combined = combine_two_datasets(
            location_df,
            greensburg_non_thermal_data[non_thermal_bands]
        )
        
        greensburg_non_thermal_data_rounded = create_rounded_data(greensburg_non_thermal_data)

        greensburg_non_thermal_data_mapped = add_nearest_data(location_df,greensburg_non_thermal_data_rounded)
        
        
        # Step 5: LiDAR elevation data
        status_text.text("Processing LiDAR elevation data...")
        progress_bar.progress(75)
        
        #! DTM
        greensburg_natural_elevation_dtm = create_lidar_data('3dep-lidar-dtm',greensburg_location,'2010-01-01/2024-01-01')
        
        lidar_dtm_file('lidar_natural_elevation_dtm.tiff',greensburg_natural_elevation_dtm,greensburg_location,'created_data')
        
        #! HAG
        greensburg_natural_elevation_hag = create_lidar_data('3dep-lidar-hag',greensburg_location,'2010-01-01/2024-01-01')
        
        # Ensure greensburg_natural_elevation_hag is a Dataset with an 'elevation' data variable
        if isinstance(greensburg_natural_elevation_hag, xr.DataArray):
            greensburg_natural_elevation_hag = greensburg_natural_elevation_hag.to_dataset(name='elevation')
        elif not isinstance(greensburg_natural_elevation_hag, xr.Dataset) or 'elevation' not in greensburg_natural_elevation_hag.data_vars:
            raise TypeError("greensburg_natural_elevation_hag must be an xarray Dataset with an 'elevation' data variable")

        lidar_file('lidar_elevation_hag.tiff',greensburg_natural_elevation_hag,greensburg_location,'created_data')

        #! DSM
        greensburg_natural_elevation_dsm = create_lidar_data('3dep-lidar-dsm',greensburg_location,'2010-01-01/2024-01-01')
        
        # Ensure greensburg_natural_elevation_dsm is a Dataset with an 'elevation' data variable
        if isinstance(greensburg_natural_elevation_dsm, xr.DataArray):
            greensburg_natural_elevation_dsm = greensburg_natural_elevation_dsm.to_dataset(name='elevation')
        elif not isinstance(greensburg_natural_elevation_dsm, xr.Dataset) or 'elevation' not in greensburg_natural_elevation_dsm.data_vars:
            raise TypeError("greensburg_natural_elevation_dsm must be an xarray Dataset with an 'elevation' data variable")
        
        lidar_file('lidar_elevation_dsm.tiff',greensburg_natural_elevation_dsm,greensburg_location,'created_data')
        
        
        # DSM elevation
        dsm_data = map_satellite_data(
            'created_data/lidar_elevation_dsm.tiff',
            location_df,
            ['elevation']
        )
        dsm_combined = combine_two_datasets(location_df, dsm_data)
        dsm_combined = dsm_combined[dsm_combined['elevation'] > 0]
        dsm_mapped = add_nearest_data(location_df, create_rounded_data(dsm_combined))
        
        # DTM elevation
        dtm_data = map_satellite_data(
            'created_data/lidar_natural_elevation_dtm.tiff',
            location_df,
            ['elevation']
        )
        
        """
        dtm_combined = pd.concat([location_df, dtm_data], axis=1)
        dtm_combined = dtm_combined[dtm_combined['elevation'] > 0]
        dtm_rounded = create_rounded_data(dtm_combined)
        dtm_mapped = add_nearest_data(location_df, dtm_rounded)
        """
        dtm_combined = combine_two_datasets(location_df, dtm_data)
        dtm_combined = dtm_combined[(dtm_combined['elevation'] > 0)]
        dtm_rounded = create_rounded_data(dtm_combined)
        dtm_mapped = add_nearest_data(location_df, dtm_rounded)

        
        # HAG elevation
        hag_data = map_satellite_data(
            'created_data/lidar_elevation_hag.tiff',
            location_df,
            ['elevation']
        )
        hag_combined = combine_two_datasets(location_df, hag_data)
        hag_combined = hag_combined[hag_combined['elevation'] > 0]
        hag_rounded = create_rounded_data(hag_combined)
        hag_mapped = add_nearest_data(location_df, hag_rounded)





        status_text.text("Preparing features for prediction...")
        progress_bar.progress(85)
        
        X = pd.DataFrame()
        X[sentinel_2_bands] = greensburg_proximity_sentinel_2[sentinel_2_bands]
        X[sentinel_1_bands] = greensburg_sentinel_1_rounded_mapped[sentinel_1_bands]
        X[thermal_bands] = greensburg_landsat_thermal_data[thermal_bands]
        X[non_thermal_bands] = greensburg_non_thermal_data_mapped[non_thermal_bands]
        X['dsm elevation'] = dsm_mapped['elevation']
        X['dtm elevation'] = dtm_mapped['elevation']
        X['hag elevation'] = hag_mapped['elevation']




        status_text.text("Loading model and generating predictions...")
        progress_bar.progress(90)
        
        # Load pre-trained model
        model = joblib.load("model/joblib_rf_model.md")
        
        # Make predictions
        uhi_predictions = model.predict(X[model.feature_names_in_])
        
        # Create prediction dataframe
        prediction_df = pd.DataFrame({
            'Longitude': location_df['Longitude'],
            'Latitude': location_df['Latitude'],
            'UHI Values Prediction': uhi_predictions
        })





        status_text.text("Generating UHI intensity map...")
        progress_bar.progress(95)
        
        # Create GeoDataFrame
        uhi_gdf = gpd.GeoDataFrame(
            prediction_df['UHI Values Prediction'] * 80,
            geometry=gpd.points_from_xy(
                prediction_df['Longitude'], 
                prediction_df['Latitude']
            ),
            crs='EPSG:4326'
        )
        
        # Convert to Web Mercator
        uhi_gdf = uhi_gdf.to_crs(epsg=3857)
        
        # Create bounding box for plotting
        bbox_geom = box(min_lon, min_lat, max_lon, max_lat)
        bbox = gpd.GeoSeries([bbox_geom], crs="EPSG:4326").to_crs(epsg=3857)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot UHI points colored by value
        uhi_gdf.plot(
            ax=ax,
            column='UHI Values Prediction',
            cmap='coolwarm',
            legend=True,
            markersize=100,
            alpha=0.015
        )
        
        # Set bounds and add basemap
        ax.set_xlim(bbox.total_bounds[[0, 2]])
        ax.set_ylim(bbox.total_bounds[[1, 3]])
        cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery)
        
        plt.title("Urban Heat Island (UHI) Intensity Prediction")
        
        # Display in Streamlit
        st.pyplot(fig)
        
        progress_bar.progress(100)
        status_text.text("Complete!")
        
        # Display statistics
        st.write("### UHI Prediction Statistics")
        st.write(f"Mean UHI Value: {prediction_df['UHI Values Prediction'].mean():.4f}")
        st.write(f"Max UHI Value: {prediction_df['UHI Values Prediction'].max():.4f}")
        st.write(f"Min UHI Value: {prediction_df['UHI Values Prediction'].min():.4f}")



        # Provide download button for predictions
        csv = prediction_df.to_csv(index=False)
        st.download_button(
            label="Download UHI Predictions as CSV",
            data=csv,
            file_name="uhi_predictions.csv",
            mime="text/csv"
        )




if map_data and map_data.get("last_active_drawing"):
    try:
        # All processing steps go here
        pass
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please ensure all required data files are present:")
        st.write("- sentinel_2_summer.tiff")
        st.write("- sentinel_1_summer.tiff")
        st.write("- landsat_thermal_july_28.tiff")
        st.write("- landsat_non_thermal_summer.tiff")
        st.write("- lidar_elevation_dsm.tiff")
        st.write("- lidar_natural_elevation_dtm.tiff")
        st.write("- lidar_elevation_hag.tiff")
        st.write("- rf_model_joblib.md")
else:
    st.info("👆 Draw a rectangle on the map to begin UHI prediction")
